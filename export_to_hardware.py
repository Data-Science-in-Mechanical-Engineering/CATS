import hydra
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import omegaconf
import torch
from tqdm import tqdm
import pickle
import subprocess
import tempfile
import sys

from dataset_loaders.dataset_utils import init_torch_dataloader
from model import quantization
from model.dropout import PartialLayerDropoutMask
from model.timeseries_decoder import TimeseriesPatchedDecoder
from model.vit import VIT
from trainer import get_best_model_name
from utils import export_utils
from utils.splitting_utils import get_neuron_slices, get_weight_slices
import os
import logging

# Convert scalings to pure Python types (nested dicts with floats)
def convert_to_python(obj):
    """Recursively convert to pure Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python(item) for item in obj]
    elif hasattr(obj, '__array__'):  # JAX or NumPy array
        arr = np.array(obj)
        if arr.size == 1:  # Scalar
            return float(arr.item())
        return arr
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj
    
def run_export(model, X, scalings_weight, scalings_activation, cfg, path_output):
    # Convert all JAX/Equinox data to pure NumPy
    model_weight_dict_numpy = jax.tree.map(lambda x: np.array(x), model.get_weight_dict())
    pruning_state_dict = jax.tree.map(lambda x: np.array(x), model.get_pruning_state_dict())
    example_input_numpy = np.array(X)
    
    scalings_weight_python = convert_to_python(scalings_weight)
    scalings_activation_python = convert_to_python(scalings_activation)
    # o_split_input_python = [int(x) for x in model.decoder.multi_head_attention_layers[0].o_split_input]
    
    # Prepare data for export
    export_data = {
        'name': 'decoder',
        'weight_dict': model_weight_dict_numpy,
        'pruning_dict': pruning_state_dict,
        'num_heads': int(model.decoder.multi_head_attention_layers[0].num_heads),
        'o_split_input': model.decoder.multi_head_attention_layers[0].o_split_input,
        'scalings_weight': scalings_weight_python,
        'scalings_activation': scalings_activation_python,
        'num_devices': int(cfg.num_devices),
        'example_input': example_input_numpy,
        'num_bits': 8,
        'attention_only': False,
        'path_output': path_output
    }
    
    # Save to pickle file
    pickle_dir = "/data/distributed_transformer/export_temp"
    os.makedirs(pickle_dir, exist_ok=True)
    pickle_path = os.path.join(pickle_dir, f"export_data_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio * cfg.pruning_step*100)}.pkl")
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(export_data, f)
    
    # Call the standalone export script via subprocess
    script_path = os.path.join(os.path.dirname(__file__), "export_standalone.py")
    result = subprocess.run(
        [sys.executable, script_path, pickle_path],
        capture_output=True,
        text=True
    )
    
    # Print output
    # if result.stdout:
    #     print(result.stdout)
    # if result.stderr:
    #     print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        if result.stdout:
            print("stdout:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Export script failed with return code {result.returncode}")
        sys.exit(1)


@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
def main(cfg: omegaconf.OmegaConf):
    logger = logging.getLogger(__name__)
    if False:
        jax.config.update('jax_platform_name', 'cpu')

        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, 3)
        
        model = TimeseriesPatchedDecoder(1, cfg, keys[0], apply_padding=False)
        print(f"Number parameters: {model.get_number_of_parameters()}")
        
        input_dim = cfg.model.input_patch_length*1
        X = 2 * jax.random.uniform(keys[2], (cfg.dataset.context_length // cfg.model.input_patch_length, input_dim)) - 1.0
        X_flattened = jnp.array([X.flatten()]).T
        
        # Example usage
        if cfg.pruning_ratio > 0.0:
            model = model.prune_step(cfg.pruning_ratio)
        # model = model.change_dropout_probability(0.1, 0.1, 0.1, 0.0)

        # def _print(x):
        #     if isinstance(x, PartialLayerDropoutMask):
        #         assert x.dropout_prob_mode1 == 0.1
        #         assert x.dropout_prob_mode2 == 0.1
        #         assert x.dropout_prob_mode3 == 0.1
        #     return x

        # jax.tree_map(_print, model, is_leaf=lambda x: isinstance(x, PartialLayerDropoutMask))

        # Y, record_activations = jax.vmap(residual_block, (0, None, None, None, None, None))(X, None, None, True, keys[0], True)
        Y, record_activations = model(X_flattened, None, None, None, True, keys[0])
        record_weights = model.get_weight_dict()

        scalings_activation = quantization.calculate_scalings(record_activations, num_bits=8)
        scalings_weight = quantization.calculate_scalings(record_weights, num_bits=8)

        path_output = f"/data/distributed_transformer/code_export/code_{cfg.num_devices}_{int(cfg.pruning_ratio*100)}"
        os.makedirs(path_output, exist_ok=True)

        export_utils.export_transformer("decoder", 
                                        transformer=model.decoder, 
                                        scalings_weight=scalings_weight, 
                                        scalings_activation=scalings_activation, 
                                        num_devices=cfg.num_devices, 
                                        example_input=X,
                                        num_bits=8,
                                        attention_only=True,
                                        path_output=path_output)

        # test
        Y, _ = model(X_flattened, None, scalings_weight, scalings_activation, True, keys[0])

        if type(Y) is tuple:
            Y = Y[0]

        print(quantization.quantize_forward(Y, scalings_activation["out"]["sum"], rounding=True))

    else:
        jax.config.update('jax_platform_name', 'cpu')
        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, 3)
        torch.manual_seed(1)
        np.random.seed(1)

        path_output = f"/data/distributed_transformer/code_export/cat_vs_dog/code_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio * cfg.pruning_step*100)}"
        os.makedirs(path_output, exist_ok=True)
        
        logger.info("Loading model...")
        model = VIT(cfg, keys[0])
        if cfg.per_step_pruning_ratio > 0.0:
            model = model.prune_step(cfg.per_step_pruning_ratio)

        model_path = "/data/distributed_transformer/cat_vs_dog_pruning2/cat_vs_dog/12_384_384_False/11/checkpoints"
        model_name = f"best_{int(cfg.per_step_pruning_ratio * cfg.pruning_step*100):03d}"
        #model = model.load_model(f"{model_path}/{model_name}", absolute_path=True)
        logger.info(f"Model loaded, number parameters: {model.get_number_of_parameters()}")
        exit()
        logger.info("Starting calibration...")
        # load dataset
        data_source = hydra.utils.instantiate(cfg.dataset)
        train_dataset, train_sampler = init_torch_dataloader(data_source.get_train_data_source(), 64, cfg.num_workers, cfg.do_distributed_training, 1)
        val_dataset, val_sampler = init_torch_dataloader(data_source.get_val_data_source(), 64, cfg.num_workers, cfg.do_distributed_training, 1)
        test_dataset, test_sampler = init_torch_dataloader(data_source.get_test_data_source(), 2, cfg.num_workers, cfg.do_distributed_training, 1)
        @eqx.filter_jit
        def predict_batch(model,
                          batch,
                          weight_scalings,
                          activation_scalings,
                          inference, 
                          key):
            if type(model) == TimeseriesPatchedDecoder:
                subkeys = jax.random.split(key, len(batch["target"]))
                pred = jax.vmap(model, (0, 0 if batch["padding_mask"] is not None else None, None, None, None, 0))(batch["target"][:, 0:cfg.dataset.context_length, ...], 
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

        train_dataset_iterator = iter(train_dataset)
        scalings_activation = None
        for i in tqdm(range(10), desc="Calibration"):
            batch = next(train_dataset_iterator)
            if type(batch["target"]) is not np.ndarray:
                batch = {k: np.asarray(v) for k, v in batch.items()}
            Y, record_activations = predict_batch(model, 
                                                 batch,
                                                 None,
                                                 None,
                                                 True,
                                                 keys[0])
            scalings_activation_temp = quantization.calculate_scalings(record_activations, num_bits=8)
            if scalings_activation is None:
                scalings_activation = scalings_activation_temp
            else:               
                scalings_activation = jax.tree_util.tree_map(lambda x, y: jnp.minimum(x, y), scalings_activation, scalings_activation_temp)
        
        scalings_weight = quantization.calculate_scalings(model.get_weight_dict(), num_bits=8)
        logger.info("Calibration done.")
        
        # Example usage
        val_dataset_iterator = iter(val_dataset)  # iter(test_dataset)  # iter(val_dataset)
        batch = next(val_dataset_iterator)
        batch = {k: np.asarray(v) for k, v in batch.items()}
        Y, _ = predict_batch(model, 
                            batch,
                            scalings_weight,
                            scalings_activation,
                            True,
                            keys[0])
        
        # Plot 8x8 grid of images with predictions and true classes
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        axes = axes.flatten()

        num_images = min(64, len(batch["image"]))
        predictions = jnp.argmax(Y, axis=-1) if len(Y.shape) > 1 else Y
        correct = jnp.sum(predictions == batch["target"])
        accuracy = correct / num_images
        logger.info(f"Accuracy on batch: {accuracy:.2%} ({correct}/{num_images})")
        for idx in range(num_images):
            ax = axes[idx]
            img = torch.tensor(batch["image"][idx])
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            
            pred_class = int(predictions[idx])
            true_class = int(batch["target"][idx])
            
            color = 'green' if pred_class == true_class else 'red'
            ax.imshow(img)
            ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}\n{quantization.quantize_forward(Y[idx, ...], scalings_activation["out"]["sum"], rounding=True)}", color=color, fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(num_images, 64):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(path_output, 'predictions_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predictions grid to {os.path.join(path_output, 'predictions_grid.png')}")
        print(batch["image"][0, ...].shape)
        print("----------------------------------------------------")
        Y_, _ = model(jnp.array(batch["image"][0, ...]), scalings_weight, scalings_activation, True, keys[0], True)
        print(quantization.quantize_forward(Y_[idx, ...], scalings_activation["out"]["sum"], rounding=True))

        X = model.patchify_input(jnp.array(batch["image"][0, ...]))
        #exit()
        print(X.shape)
        X = jnp.concatenate((X, model.decoder.cls_token), axis=0)

        run_export(model, X, scalings_weight, scalings_activation, cfg, path_output)
            
        logger.info("Export done.")




if __name__ == "__main__":
    main()
