import collections
import copy
from functools import partial
import gc
import itertools
import logging
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import time
import numpy as np
import omegaconf
import optax
from tqdm import tqdm

import hydra

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
import equinox as eqx

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

import torch

from model.timeseries_decoder import TimeseriesPatchedDecoder
from model.vit import VIT
import utils.jax_utils as ju

import mpx

from dataset_loaders.dataset_utils import init_tf_dataloader_timeseries, init_torch_dataloader
from execution_guard import execution_guard


PREFETCH_TO_DEVICE = False

# from: https://flax-linen.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device, but adapted
def prefetch_to_device(iterator, size, sharding):
    """Shard and prefetch batches on device.

    This utility takes an iterator and returns a new iterator which fills an on
    device prefetch buffer. Eager prefetching can improve the performance of
    training loops significantly by overlapping compute and data transfer.

    This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
    necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
    location that isn't free yet so they don't block. Instead those allocators OOM.

    Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
    dimension is sharded across devices.

    size: the size of the prefetch buffer.

    If you're training on GPUs, 2 is generally the best choice because this
    guarantees that you can overlap a training step on GPU with a data
    prefetch step on CPU.

    devices: the list of devices to which the arrays should be prefetched.

    Defaults to the order of devices expected by ``jax.pmap``.

    Yields:
    The original items from the iterator where each ndarray is now sharded to
    the specified devices.
    """
    queue = collections.deque()

    def _prefetch(xs):
        return jax.device_put(xs, sharding)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1) 


def world_info_from_env():
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def filtered_device_put(tree, sharding):
    dynamic, static = eqx.partition(tree, eqx.is_array)
    dynamic = jax.device_put(dynamic, sharding)
    return eqx.combine(dynamic, static)


def get_best_model_name(pruning_ratio: float) -> str:
    """
    Returns the name of the best model for a given pruning ratio.
    """
    return f"best_{int(round(pruning_ratio*100)):03d}"

def get_model_name(pruning_ratio: float, epoch: int) -> str:
    """
    Returns the name of the model for a given pruning ratio and epoch.
    """
    return f"m_{int(round(pruning_ratio*100)):03d}_{epoch:03d}"


def log(wandb, data):
    if wandb is not None:
        wandb.log(data)

def print_console(cfg, data):
    if cfg.print_console:
        print(data)


class Trainer:
    def __init__(self, cfg: omegaconf.DictConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.wandb = None
        
        self._setup_distributed()
        self._load_dataset()
        self._setup_jax_distributed()
        self._setup_sharding()
        self._init_model()

        # Create jitted versions
        self.__make_step_jitted = eqx.filter_jit(self._make_step)
        self.__batched_loss_acc_jitted = eqx.filter_jit(self._batched_loss_acc_wrapper)
        

    def _setup_distributed(self):
        """Set up distributed training with torch."""
        if self.cfg.do_distributed_training:
            local_rank, global_rank, world_size = world_info_from_env()
            self.logger.info(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}")

            torch.distributed.init_process_group(world_size=world_size, rank=global_rank,)
            self.logger.info("Successfully initialized torch distributed process group.")
            
            self.local_rank = local_rank
            self.global_rank = global_rank
            self.world_size = world_size
        else: 
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0
    
    def _load_dataset(self):
        """Load dataset."""
        self.logger.info("Loading dataset...")
        self.data_source = hydra.utils.instantiate(self.cfg.dataset)
        
        if self.cfg.use_pytorch_dataloader:
            self.train_dataset, self.train_sampler = init_torch_dataloader(
                self.data_source.get_train_data_source(), 
                self.cfg.batch_size, 
                self.cfg.num_workers, 
                self.cfg.do_distributed_training, 
                self.world_size
            )
            self.val_dataset, self.val_sampler = init_torch_dataloader(
                self.data_source.get_val_data_source(), 
                self.cfg.eval_batch_size, 
                self.cfg.num_workers, 
                self.cfg.do_distributed_training, 
                self.world_size
            )
            self.test_dataset, self.test_sampler = init_torch_dataloader(
                self.data_source.get_test_data_source(), 
                self.cfg.eval_batch_size, 
                self.cfg.num_workers, 
                self.cfg.do_distributed_training, 
                self.world_size
            )
        else:
            if self.cfg.do_distributed_training:
                raise NotImplementedError("Distributed training is not implemented for TensorFlow dataloaders yet.")
            
            self.train_dataset = init_tf_dataloader_timeseries(
                data_source=self.data_source.get_train_data_source(),
                batch_size=self.cfg.batch_size,
                num_epochs=self.cfg.epochs,
                num_features=self.data_source.num_features,
                prediction_length=self.data_source.prediction_length,
                context_length=self.data_source.context_length,
                seed=self.cfg.seed,
                num_workers=self.cfg.num_workers,
            )
            
            self.val_dataset = init_tf_dataloader_timeseries(
                data_source=self.data_source.get_val_data_source(),
                batch_size=self.cfg.batch_size,
                num_epochs=self.cfg.epochs,
                num_features=self.data_source.num_features,
                prediction_length=self.data_source.prediction_length,
                context_length=self.data_source.context_length,
                seed=self.cfg.seed,
                num_workers=self.cfg.num_workers,
            ) 
            
            self.test_dataset = init_tf_dataloader_timeseries(
                data_source=self.data_source.get_test_data_source(),
                batch_size=self.cfg.batch_size,
                num_epochs=self.cfg.epochs,
                num_features=self.data_source.num_features,
                prediction_length=self.data_source.prediction_length,
                context_length=self.data_source.context_length,
                seed=self.cfg.seed,
                num_workers=self.cfg.num_workers,
            )
    
    def _setup_jax_distributed(self):
        """Set up JAX distributed after data loading."""
        if self.cfg.do_distributed_training:
            jax.distributed.initialize(f"{os.environ['MASTER_ADDR']}:10000", local_device_ids=[0, 1, 2, 3])
            self.logger.info("Successfully initialized jax distributed process group.")
            self.logger.info("process id = %d", jax.process_index())
            self.logger.info("global devices = %s", jax.devices())
            self.logger.info("local devices = %s", jax.local_devices())
    
    def _setup_sharding(self):
        """Set up sharding for JAX."""
        devices = jax.devices(backend="gpu")
        self.logger.info(f"Found devices: {devices}")
        mesh = jax.make_mesh((len(devices), ), ("batch",), devices=devices)
        self.num_gpu_devices = len(devices)
        self.batch_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("batch"))
        self.replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())

        if PREFETCH_TO_DEVICE:
            self.train_dataset = prefetch_to_device(self.train_dataset, 2, self.batch_sharding)
            self.val_dataset = prefetch_to_device(self.val_dataset, 2, self.batch_sharding)
            self.test_dataset = prefetch_to_device(self.test_dataset, 2, self.batch_sharding)
    
    def _init_model(self):
        """Initialize model without loading or pruning."""
        key = jax.random.PRNGKey(self.cfg.seed)
        key, subkey = jax.random.split(key)
        
        if self.cfg.model.type == "timeseries_decoder":
            self.base_model = TimeseriesPatchedDecoder(
                input_dim=self.data_source.num_features, 
                cfg=self.cfg, 
                key=subkey
            )
        else:
            self.base_model = VIT(self.cfg, subkey)
        
        self.logger.info(f"Model created successfully. \n model size: {self.base_model.get_number_of_parameters()}")
        self.key = key
    
    def _setup_optimizer(self, model):
        """Set up optimizer for training."""
        # optimizer strategy from https://arxiv.org/abs/2106.10270
        duration_linear_schedule = self.cfg.optimizer.warmup_epochs * int(self.data_source.length_train / self.cfg.batch_size - 1e-5)
        linear_schedule = optax.linear_schedule(
            init_value=self.cfg.optimizer.learning_rate * 0.01,
            end_value=self.cfg.optimizer.learning_rate,
            transition_steps=duration_linear_schedule,
        )
        duration_cosine_schedule = self.cfg.epochs * int(self.data_source.length_train / self.cfg.batch_size - 1e-5)
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=self.cfg.optimizer.learning_rate,
            decay_steps=duration_cosine_schedule,
            alpha=0.0,
        )
        learning_rate_schedule = optax.join_schedules(
            schedules=[linear_schedule, cosine_schedule],
            boundaries=[duration_linear_schedule],
        )

        clip_transform = optax.clip_by_global_norm(1.0)
        adam_transform = optax.scale_by_adam()
        weight_decay = optax.add_decayed_weights(self.cfg.optimizer.weight_decay)
        learning_rate_transform = optax.scale_by_learning_rate(learning_rate_schedule)

        optimizer_adam = optax.chain(clip_transform, adam_transform, weight_decay, learning_rate_transform)
        optimizer_state_adam = optimizer_adam.init(eqx.filter(model, eqx.is_array))
        optimizer_state_adam = filtered_device_put(optimizer_state_adam, self.replicated_sharding)

        # calculate learning rate for sgd after switching
        optimizer_sgd = None
        optimizer_state_sgd = None
        if self.cfg.optimizer.epoch_switch_to_sgd < self.cfg.epochs:
            current_time = self.cfg.optimizer.epoch_switch_to_sgd * int(self.data_source.length_train / self.cfg.batch_size - 1e-5)
            learning_rate_scaling = 0.5 * (1 + jnp.cos(jnp.pi * current_time / duration_cosine_schedule))
            duration_cosine_schedule = (self.cfg.epochs-self.cfg.optimizer.epoch_switch_to_sgd) * int(self.data_source.length_train / self.cfg.batch_size - 1e-5)
            cosine_schedule_sgd = optax.cosine_decay_schedule(
                init_value=self.cfg.optimizer.learning_rate * learning_rate_scaling,
                decay_steps=duration_cosine_schedule,
                alpha=0.0,
            )
            ams_grad_transform = optax.scale_by_amsgrad()
            learning_rate_transform_sgd = optax.scale_by_learning_rate(cosine_schedule_sgd)
            optimizer_sgd = optax.chain(clip_transform, ams_grad_transform, weight_decay, learning_rate_transform_sgd)
            optimizer_state_sgd = optimizer_sgd.init(eqx.filter(model, eqx.is_array))
            optimizer_state_sgd = filtered_device_put(optimizer_state_sgd, self.replicated_sharding)
        
        return optimizer_adam, optimizer_state_adam, optimizer_sgd, optimizer_state_sgd
    
    def _init_wandb(self):
        """Initialize wandb logging."""
        self.cfg.wandb_log = self.cfg.wandb_log and self.global_rank == 0
        if self.cfg.wandb_log:
            import wandb

            wnb_cfg = omegaconf.OmegaConf.to_container(self.cfg, resolve=True)
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=wnb_cfg,
                job_type="train",
                name=self.cfg.experiment_name_wandb,
            )

            wandb.define_metric("epoch")
            wandb.define_metric("val_loss", step_metric="epoch")
            wandb.define_metric("acc_eval", step_metric="epoch")
            wandb.define_metric("acc_test", step_metric="epoch")
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("loss_batch")
            wandb.define_metric("test_loss")
            wandb.define_metric("test_loss_message_loss")
            wandb.define_metric("acc_test_message_loss")
            wandb.define_metric("test_loss_zero_shot")
            wandb.define_metric("message_loss")
            wandb.define_metric("timings/data_loading_time")
            wandb.define_metric("timings/batch_data_loading_time")
            wandb.define_metric("timings/step_time")
            wandb.define_metric("timings/mean_step_time")
            
            self.wandb = wandb
        
        os.makedirs("checkpoints", exist_ok=True)
        self.logger.info("Init WANDB")
    
    def _predict_batch(self, model: eqx.Module, 
                    batch: dict,
                    weight_scalings: dict | None,
                    activation_scalings: dict | None,
                    inference: bool, 
                    key: PRNGKeyArray) -> Array:
        if type(model) == TimeseriesPatchedDecoder:
            subkeys = jax.random.split(key, len(batch["target"]))
            pred = jax.vmap(model, (0, 0 if batch["padding_mask"] is not None else None, None, None, None, 0))(batch["target"][:, 0:self.cfg.dataset.context_length, ...], 
                                                                                                            batch["padding_mask"], 
                                                                                                            weight_scalings, 
                                                                                                            activation_scalings, 
                                                                                                            inference, 
                                                                                                            subkeys)
            
            return pred
        elif type(model) == VIT:
            subkeys = jax.random.split(key, len(batch["image"]))
            pred = jax.vmap(model, in_axes=(0, None, None, None, 0))(batch["image"],
                                    weight_scalings, 
                                    activation_scalings, 
                                    inference, 
                                    subkeys)
            return pred
        else:
            raise NotImplementedError(f"Model type {type(model)} not implemented in predict_batch.")
    
    def _batched_loss_acc_wrapper(self, model, batch, key, inference, weight_regularization=0):
        """
        Wrapper function to calculate the loss and accuracy for a batch of data.
        """
        batch = eqx.filter_shard(batch, self.batch_sharding)
        model = eqx.filter_shard(model, self.replicated_sharding)
        pred, recorded_activations = self._predict_batch(model, batch, None, None, inference, key)
        target = batch["target"]
        
        losses = jax.vmap(model.loss)(pred, target)
        acc = jax.vmap(model.acc)(pred, target)

        loss = mpx.force_full_precision(jnp.mean, losses.dtype)(losses)
        acc = mpx.force_full_precision(jnp.mean, losses.dtype)(acc)

        loss = eqx.filter_shard(loss, self.replicated_sharding)
        acc = eqx.filter_shard(acc, self.replicated_sharding)

        params, _ = eqx.partition(model, eqx.is_array)
        params = jax.tree_util.tree_leaves(params)
        params = jax.tree_util.tree_map(lambda x: x.flatten(), params)
        params = jnp.concatenate(params).flatten()
        
        loss = loss + weight_regularization * mpx.force_full_precision(jnp.sum, jnp.float32)(jnp.abs(params))
        return loss, acc
    
    def _make_step(self, model: eqx.Module, 
                optimizer,
                optimizer_state: PyTree, 
                batch: dict,
                loss_scaling: mpx.DynamicLossScaling | None,
                key: PRNGKeyArray,
                weight_regularization: float = 0,
                ) -> tuple[eqx.Module, PyTree, Float, PRNGKeyArray]:
        
        self.logger.info("Recompile make_step")
        
        batch = eqx.filter_shard(batch, self.batch_sharding)
        model = eqx.filter_shard(model, self.replicated_sharding)
        loss_scaling = eqx.filter_shard(loss_scaling, self.replicated_sharding)
        optimizer_state = eqx.filter_shard(optimizer_state, self.replicated_sharding)
        
        # calculate padding as described in TimesFM
        def calculate_padding(key_):
            r = jax.random.randint(key_, shape=(1,), minval=0, maxval=model.input_patch_length)
            indexes = jnp.arange(self.cfg.dataset.context_length)
            padding_mask  = jnp.where(indexes < r, 1.0, 0.0)
            return padding_mask

        if self.cfg.add_padding:
            key, subkeys = ju.get_subkeys(key, len(batch["target"]))
            padding_masks_batched = jax.vmap(calculate_padding)(subkeys)

            batch["padding_mask"] = padding_masks_batched
        else:
            batch["padding_mask"] = None


        if self.cfg.num_gradient_accumulation_steps == 1:
            # calculate loss and gradients
            (loss_value, acc), loss_scaling, grads_finite, grads = mpx.filter_value_and_grad(
                lambda m, b, k, i: self._batched_loss_acc_wrapper(m, b, k, i, weight_regularization), 
                scaling=loss_scaling, has_aux=True, use_mixed_precision=self.cfg.train_mixed_precision)(
                    model, batch, key, False)

            # optimizer step
            model, optimizer_state = mpx.optimizer_update(model=model,
                optimizer=optimizer, 
                optimizer_state=optimizer_state, 
                grads=grads, 
                grads_finite=grads_finite
            )
        else:
            size_minibatch = len(batch["target"]) // self.cfg.num_gradient_accumulation_steps
            if len(batch["target"]) % self.cfg.num_gradient_accumulation_steps != 0:
                raise ValueError(f"Batch size {len(batch['target'])} must be divisible by num_gradient_accumulation_steps {self.cfg.num_gradient_accumulation_steps}.")
            grads_accumulated = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else None, model)
            grads_accumulated = eqx.filter_shard(grads_accumulated, self.replicated_sharding)

            # because scan makes all statics in the scaling to weeak types otherwise leading to two times recombiling
            loss_scaling_dynamic, loss_scaling_static = eqx.partition(loss_scaling, eqx.is_array)

            def single_gradient_step(carry, batch):
                key_ = carry[0]
                grads_accumulated_ = carry[1]
                loss_scaling_dynamic_ = carry[2]
                loss_scaling_ = eqx.combine(loss_scaling_dynamic_, loss_scaling_static)
                loss_value = carry[3]
                key_, subkey = jax.random.split(key_)
                # calculate loss and gradients
                (loss_value_, acc), loss_scaling_, grads_finite, grads = mpx.filter_value_and_grad(
                    lambda m, b, k, i: self._batched_loss_acc_wrapper(m, b, k, i, weight_regularization), 
                    scaling=loss_scaling_, has_aux=True, use_mixed_precision=self.cfg.train_mixed_precision)(model, batch, subkey, False)
                
                grads_accumulated_ = eqx.apply_updates(grads_accumulated_, grads)
                loss_value += loss_value_.astype(jnp.float32) / self.cfg.num_gradient_accumulation_steps
                
                loss_scaling_dynamic_, _ = eqx.partition(loss_scaling_, eqx.is_array)
                return (key_, grads_accumulated_, loss_scaling_dynamic_, loss_value), loss_value_
        
            # we do it such that each mini-minibatch is processed on all gpus.
            batch_reshape = jax.tree_util.tree_map(lambda x: x.reshape((self.num_gpu_devices, -1) + x.shape[1:]), batch)
            batch_reshape = jax.tree_util.tree_map(lambda x: x.reshape((self.num_gpu_devices, self.cfg.num_gradient_accumulation_steps, size_minibatch // self.num_gpu_devices) + x.shape[2:]), batch_reshape)
            batch_reshape = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch_reshape)
            batch_reshape = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (self.cfg.num_gradient_accumulation_steps, size_minibatch) + x.shape[3:]), batch_reshape)

            carry, _ = jax.lax.scan(single_gradient_step, (key, grads_accumulated, loss_scaling_dynamic, np.zeros((), dtype=jnp.float32)), batch_reshape)

            key = carry[0]
            grads_accumulated = carry[1]
            loss_scaling = eqx.combine(carry[2], loss_scaling_static)
            loss_value = carry[3]

            grads_acccumulated = jax.tree_util.tree_map(lambda x: x / self.cfg.num_gradient_accumulation_steps if eqx.is_array(x) else x, grads_accumulated)
            grads_finite = mpx.all_finite(grads_acccumulated)
            # optimizer step
            model, optimizer_state = mpx.optimizer_update(model, optimizer, optimizer_state, grads_acccumulated, grads_finite)

        model = eqx.filter_shard(model, self.replicated_sharding)
        loss_scaling = eqx.filter_shard(loss_scaling, self.replicated_sharding)
        optimizer_state = eqx.filter_shard(optimizer_state, self.replicated_sharding)
        
        return model, optimizer_state, loss_scaling, loss_value
    
    def _train_epoch(self, model: eqx.Module, 
                    optimizer,
                    optimizer_state: PyTree,
                    num_batches: Int,
                    loss_scaling: mpx.DynamicLossScaling | None,
                    key: PRNGKeyArray,
                    epoch: int,
                    weight_regularization: float = 0) -> tuple[eqx.Module, PyTree, PRNGKeyArray]:
        if self.cfg.do_distributed_training:
            self.train_sampler.set_epoch(epoch)
        
        loss_value = 0
        num_datapoints = 0
        
        data_loading_time = 0
        conversion_time = 0
        training_time = 0

        model = model.change_dropout_probability(self.cfg.model.partial_layer_dropout_prob, 
                                                 self.cfg.model.partial_layer_dropout_prob, 
                                                 self.cfg.model.partial_layer_dropout_prob, 
                                                 0.0)

        start_time = time.time()
        train_dataset_iterator = iter(self.train_dataset)
        gc.disable()

        
        for idx in tqdm(range(num_batches), disable=self.global_rank != 0):
            start_time = time.time()
            batch = next(train_dataset_iterator)

            c_start_time = time.time()
            if type(batch["target"]) is not np.ndarray:
                batch = {k: np.asarray(v) for k, v in batch.items()}

            if self.cfg.do_distributed_training:
                batch = {k: jax.make_array_from_process_local_data(self.batch_sharding, v) for k, v in batch.items()}
            
            conversion_time += time.time() - c_start_time
            if not PREFETCH_TO_DEVICE:
                batch = jax.device_put(batch, self.batch_sharding)
            batch_data_loading_time = time.time() - start_time
            data_loading_time += batch_data_loading_time

            start_time = time.time()
            key, subkey = jax.random.split(key)

            loss_batch = 0
            model, optimizer_state, loss_scaling, loss_batch = self.__make_step_jitted(
                model=model, 
                optimizer=optimizer,
                optimizer_state=optimizer_state,
                batch=batch, 
                loss_scaling=loss_scaling,
                key=subkey,
                weight_regularization=weight_regularization,
            )
            

            if jnp.isfinite(loss_batch):
                loss_value += loss_batch.astype(jnp.float32) * len(batch["target"])
                num_datapoints += len(batch["target"])
            step_time = time.time() - start_time

            if idx > 0:
                training_time += step_time
            log(self.wandb, {"timings/data_loading_time": data_loading_time, "timings/batch_data_loading_time": batch_data_loading_time, "timings/step_time": step_time, "timings/mean_step_time": training_time / (max(idx, 1)), "loss_batch": loss_batch})

        
        self.logger.info(f"Data loading time: {data_loading_time}, Training time: {training_time}, Conversion time: {conversion_time}")
        gc.collect()
        gc.enable()
        return model, optimizer_state, loss_scaling, (loss_value+5e-3) / (num_datapoints+1e-3), training_time
    
    def _eval_epoch(self, model: eqx.Module,
                epoch: Int,
                num_batches: Int,
                key: PRNGKeyArray,
                dataset_iterator,
                sampler=None) -> Float:
        if self.cfg.do_distributed_training and sampler is not None:
            sampler.set_epoch(epoch)
        loss_value = 0
        acc = 0
        num_datapoints = 0
        idx = 0
        
        for idx in tqdm(range(num_batches), disable=self.global_rank != 0):
            batch = next(dataset_iterator)
            if type(batch["target"]) is not np.ndarray:
                batch = {k: np.asarray(v) for k, v in batch.items()}
            if self.cfg.do_distributed_training:
                batch = {k: jax.make_array_from_process_local_data(self.batch_sharding, v) for k, v in batch.items()}
            elif not PREFETCH_TO_DEVICE:
                batch = jax.device_put(batch, self.batch_sharding)
            key, subkey = jax.random.split(key)
            batch["padding_mask"] = None
            if self.cfg.train_mixed_precision:
                model, batch = mpx.cast_to_half_precision((model, batch))
            
            loss_temp, acc_temp = self.__batched_loss_acc_jitted(model, batch, subkey, True, 0)
            
            if jnp.isfinite(loss_temp):
                loss_value += loss_temp.astype(jnp.float32) * len(batch["target"])
                acc += acc_temp * len(batch["target"])
                num_datapoints += len(batch["target"])
                

            if idx < 10 and epoch % self.cfg.checkpoint_interval == self.cfg.checkpoint_interval - 1 and self.cfg.print_console and self.cfg.save_examples:
                batch["padding_mask"] = None
                pred, recorded_activations = self._predict_batch(model, batch, None, None, False, key)
                
                # reversible input normalization
                if type(pred) is tuple:
                    mean, std = pred[1], pred[2]
                    prediction = pred[0][0, -1, ...]
                else:
                    prediction = pred[0, -1, ...]
                
                target = batch["target"][0, -200:, ...]
                x_axis = np.arange(target.shape[0])
                import matplotlib.pyplot as plt

                if prediction.shape[-1] == 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(x_axis[-self.cfg.dataset.prediction_length:], prediction[:, 0], label="Prediction")
                    plt.plot(x_axis, target[:, 0], label="Target")
                    plt.legend()
                    plt.title(f"Epoch {epoch} - Prediction vs Target")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Values")
                    plt.grid()
                    plt.savefig(f"checkpoints/prediction_vs_target_epoch{epoch}_{idx}.png")
                    plt.close()
                else:
                    num_subplots = prediction.shape[-1]
                    fig, axes = plt.subplots(int(num_subplots / 3 - 1e-3) + 1, 3, figsize=(10, 10))
                    axes = axes.flatten()
                    for j in range(num_subplots):
                        axes[j].plot(prediction[:, j], label="Prediction")
                        axes[j].plot(target[:, j], label="Target")
                        axes[j].legend()
                        axes[j].set_title(f"Epoch {epoch} - Prediction vs Target (Feature {j})")
                        axes[j].set_xlabel("Time Steps")
                        axes[j].set_ylabel("Values")
                        axes[j].grid()
                    plt.tight_layout()
                    plt.savefig(f"checkpoints/prediction_vs_target_epoch{epoch}_{idx}.png")
                    plt.close()
        
        return (loss_value+5e-3) / (num_datapoints + 1e-3), (acc) / (num_datapoints + 1e-3)
    
    def train(self, pruning_step: int = 0):
        """
        Train the model for a specific pruning step.
        
        Args:
            pruning_step: The current pruning step (0 for initial training)
        """
        self._init_model()
        # Load or initialize model based on pruning step
        model = self.base_model
        self.cfg.pruning_step = pruning_step
        
        if pruning_step > 0 and not self.cfg.do_finetuning:
            # model = model.load_model(get_best_model_name(self.cfg.per_step_pruning_ratio*(pruning_step-1)))
            for _ in range(pruning_step):
                model = model.prune_step(self.cfg.per_step_pruning_ratio)
        elif self.cfg.do_finetuning:
            model = model.load_model(f"{self.cfg.root_dir}/foundation_model/{self.cfg.model.num_transformer_blocks}_{self.cfg.model.num_features_attention}_{self.cfg.model.num_features_residual}/best_000", absolute_path=True)
            self.cfg.optimizer.learning_rate = self.cfg.optimizer.learning_rate * 0.1

            if pruning_step > 0:
                model = model.prune_step(self.cfg.per_step_pruning_ratio * pruning_step)

        model = model.change_dropout_probability(0.0, 0.0, 0.0, 0.0)
        model = filtered_device_put(model, self.replicated_sharding)
        
        # Setup mixed precision
        if self.cfg.train_mixed_precision:
            loss_scaling = mpx.DynamicLossScaling(loss_scaling=mpx.FLOAT16_MAX, min_loss_scaling=jnp.ones((), dtype=jnp.float32) * 1.0, period=2000)
            loss_scaling = filtered_device_put(loss_scaling, self.replicated_sharding)
        else:
            loss_scaling = None
        
        # Setup optimizer
        optimizer_adam, optimizer_state_adam, optimizer_sgd, optimizer_state_sgd = self._setup_optimizer(model)
        
        # Initialize wandb
        self._init_wandb()
        
        weight_regularization = 0
        
        # Zero-shot evaluation
        num_evals = 10
        test_loss = 0
        for i in range(num_evals):
            self.key, subkey = jax.random.split(self.key)
            test_dataset_iterator = iter(self.test_dataset)
            loss_, acc_ = self._eval_epoch(model=model, 
                                    num_batches=int(self.data_source.length_test / self.cfg.eval_batch_size - 1e-5),
                                    epoch=0,
                                    key=subkey,
                                    dataset_iterator=test_dataset_iterator,
                                    sampler=self.test_sampler if self.cfg.use_pytorch_dataloader else None)
            test_loss += loss_
        log(self.wandb, {"test_loss_zero_shot": test_loss / num_evals, "message_loss": self.cfg.model.partial_layer_dropout_prob})

        training_times = []
        best_val_loss = 1e6
        optimizer_state = optimizer_state_adam
        
        for epoch in range(self.cfg.epochs):
            # Switch optimizer if needed
            if epoch == self.cfg.optimizer.epoch_switch_to_sgd:
                optimizer_state = optimizer_state_sgd
                current_optimizer = optimizer_sgd
            else:
                current_optimizer = optimizer_adam
            
            # Train
            model, optimizer_state, loss_scaling, train_loss, training_time = self._train_epoch(
                model=model, 
                optimizer=current_optimizer,
                optimizer_state=optimizer_state,
                num_batches=int(self.data_source.length_train / self.cfg.batch_size - 1e-5),
                loss_scaling=loss_scaling,
                key=self.key,
                epoch=epoch,
                weight_regularization=weight_regularization
            )
            
            # Track training time
            if epoch > 0:
                training_times.append(training_time)
                print_console(self.cfg, f"mean training time: {np.mean(training_times)}, max training time: {np.max(training_times)}, min training time: {np.min(training_times)}")
            if loss_scaling is not None:
                print(loss_scaling.loss_scaling)
            
            # Evaluate
            val_dataset_iterator = iter(self.val_dataset)
            val_loss, acc = self._eval_epoch(
                model=model, 
                epoch=epoch,
                num_batches=int(self.data_source.length_val / self.cfg.eval_batch_size - 1e-5),
                key=self.key,
                dataset_iterator=val_dataset_iterator,
                sampler=self.val_sampler if self.cfg.use_pytorch_dataloader else None
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model(get_best_model_name(self.cfg.per_step_pruning_ratio*pruning_step))
            
            # Log
            log(self.wandb, {"train_loss": train_loss, "val_loss": val_loss, "acc_eval": acc * 100, "epoch": epoch})
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")

            # Checkpointing
            if epoch % self.cfg.checkpoint_interval == self.cfg.checkpoint_interval - 1:
                model.save_model(get_model_name(self.cfg.per_step_pruning_ratio*pruning_step, epoch))

        # Final test with best model
        model = model.load_model(get_best_model_name(self.cfg.per_step_pruning_ratio*pruning_step))
        model = model.change_dropout_probability(self.cfg.model.partial_layer_dropout_prob, 
                                                 self.cfg.model.partial_layer_dropout_prob, 
                                                 self.cfg.model.partial_layer_dropout_prob, 
                                                 0.0)
        
        # Test with different message loss values
        num_evals = 10
        test_loss = 0
        for i in range(num_evals):
            self.key, subkey = jax.random.split(self.key)
            test_dataset_iterator = iter(self.test_dataset)
            loss_, acc_ = self._eval_epoch(model=model, 
                                    num_batches=int(self.data_source.length_test / self.cfg.eval_batch_size - 1e-5),
                                    epoch=epoch,
                                    key=subkey,
                                    dataset_iterator=test_dataset_iterator,
                                    sampler=self.test_sampler if self.cfg.use_pytorch_dataloader else None)
            test_loss += loss_
        log(self.wandb, {"test_loss": test_loss / num_evals})
        
        for idx in range(0, 11):
            message_loss = 0.01 * idx
            model = model.change_dropout_probability(message_loss, message_loss, message_loss, 0.0)
            test_loss = 0
            test_acc = 0
            num_evals = 10
            for i in range(num_evals):
                self.key, subkey = jax.random.split(self.key)
                test_dataset_iterator = iter(self.test_dataset)
                loss_, acc_ = self._eval_epoch(model=model, 
                                        num_batches=int(self.data_source.length_test / self.cfg.eval_batch_size - 1e-5),
                                        epoch=epoch,
                                        key=subkey,
                                        dataset_iterator=test_dataset_iterator,
                                        sampler=self.test_sampler if self.cfg.use_pytorch_dataloader else None)
                test_loss += loss_
                test_acc += acc_
            log(self.wandb, {"test_loss_message_loss": test_loss / num_evals, "acc_test_message_loss": test_acc / num_evals, "message_loss": message_loss})
        
        if self.wandb is not None:
            self.wandb.finish()