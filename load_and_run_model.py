import jax 
import jax.numpy as jnp
import hydra
import mpx
import numpy as np
from tqdm import tqdm

import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

import matplotlib.pyplot as plt

from model.timeseries_decoder import TimeseriesPatchedDecoder
from dataset_loaders.Monash import Monash

from torch.utils.data import Dataset, DataLoader, default_collate
from dataset_loaders.utsdataset import UTSDataSource
from dataset_loaders.ICD import ICDDataSource
from dataset_loaders.ETT import ETT
from dataset_loaders.ECL import ECLDataSource
from dataset_loaders.traffic import TrafficDataSource


@eqx.filter_jit
def predict_batch(model: eqx.Module, 
                  batch: dict,
                  weight_scalings: dict | None,
                  activation_scalings: dict | None,
                  inference: bool, 
                  key: PRNGKeyArray) -> Array:
    subkeys = jax.random.split(key, len(batch["target"]))
    print("b")
    pred = jax.vmap(model, (0, 0 if batch["padding_mask"] is not None else None, None, None, None, 0))(batch["target"][:, 0:512, ...], 
                                                                                                       batch["padding_mask"], 
                                                                                                       weight_scalings, 
                                                                                                       activation_scalings, 
                                                                                                       inference, 
                                                                                                       subkeys)
    
    return pred

@eqx.filter_jit
def loss(model, batch, key):
    """
    
    Wrapper function to calculate the loss and accuracy for a batch of data.

    Args:
        model (eqx.Module): The model to be evaluated.
        batch (dict): The batch of data.
        batch_sharding (jax.sharding.NamedSharding): Sharding for the batch.
        replicated_sharding (jax.sharding.NamedSharding): Replicated sharding.
        denormalize (callable): Function to denormalize the data.
        key (PRNGKeyArray): Random key for JAX operations.
        loss_scaling (DynamicLossScaling | None): Loss scaling object for mixed precision training. If None, then the training is not mixed precision.

    """
    # model_half = model
    pred, recorded_activations = predict_batch(model, batch, None, None, False, key)
    target = batch["target"]
    
    losses = jax.vmap(model.loss)(pred, target)
    loss = mpx.force_full_precision(jnp.mean, losses.dtype)(losses)
    return loss, losses, pred


@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
def load_and_run_model(cfg):
    # data_source = UTSDataSource(dataset_base_path="/data/datasets", 
    #                             subset_name=r'UTSD-1G', 
    #                             flag='val',
    #                             split=0.9, context_length=512, prediction_length=96,
    #                             scale=True, stride=1)

    # data_source = ICDDataSource(dataset_base_path="/data/datasets", 
    #                            context_length=512, 
    #                            prediction_length=96, 
    #                            name="ICD")

    # data_source = ETT(variant="h1",
    #                   context_length=512,
    #                   prediction_length=96,
    #                   dataset_base_path="/data/datasets",
    #                   name="ETT")

    # data_source = ECLDataSource(context_length=512,
    #                             prediction_length=96,
    #                             dataset_base_path="/data/datasets",
    #                             name="ECL")

    data_source = TrafficDataSource(context_length=512,
                            prediction_length=96,
                            dataset_base_path="/data/datasets",
                            name="traffic")
    print(len(data_source.val_data_source))
    # dataset_train = TorchDatasetWrapper(data_source.get_train_data_source())
    dataloader = DataLoader(data_source.val_data_source, 
                            batch_size=1024, 
                            shuffle=True, 
                            num_workers=1, 
                            prefetch_factor=None,
                            in_order=True)  #, collate_fn=numpy_collate)


    # load model
    model = TimeseriesPatchedDecoder(input_dim=1, cfg=cfg, key=jax.random.PRNGKey(0))
    foundation_model = model.load_model("/data/distributed_transformer/test_utsd/UTSD/20250715091640/checkpoints/best_000", absolute_path=True)
    model = model.load_model("/data/distributed_transformer/traffic_messageloss/traffic/12_128_256_True/0/checkpoints/best_000", absolute_path=True)

    # model = model.prune_step(0.2)

    foundation_model = foundation_model.change_dropout_probability(0.01, 0.01, 0.01, 0)
    model = model.change_dropout_probability(0.01, 0.01, 0.01, 0)


    idx = 0
    key = jax.random.PRNGKey(0)

    dataloader = iter(dataloader)
    for idx in tqdm(range(100)):
        batch_complete = next(dataloader)
        batch_complete = {k: np.asarray(v)[:, :, :] for k, v in batch_complete.items()}

        predictions = []
        foundation_predictions = []
        for i in range(batch_complete["target"].shape[-1]):
            batch = {k: v[:, :, i:i+1] for k, v in batch_complete.items()}
            key, subkey = jax.random.split(key)
            batch["padding_mask"] = None

            _, _, pred = loss(model, batch, subkey)
            _, _, pred_foundational = loss(foundation_model, batch, subkey)

            if type(pred) is tuple:
                mean, std = pred[1], pred[2]
                print(pred[0].shape, mean.shape, std.shape)
                prediction = pred[0][:, -1, ...]
            else:
                prediction = pred[:, -1, ...]

            if type(pred_foundational) is tuple:
                mean_foundational, std_foundational = pred_foundational[1], pred_foundational[2]
                print(pred_foundational[0].shape, mean_foundational.shape, std_foundational.shape)
                foundation_prediction = pred_foundational[0][:, -1, ...]
            else:
                foundation_prediction = pred_foundational[:, -1, ...]

            
            foundation_predictions.append(foundation_prediction)
            predictions.append(prediction)
        
        for i in tqdm(range(3)):
            if batch_complete["target"].shape[-1] == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                prediction = predictions[0][i, :, ...].squeeze()
                foundation_prediction = foundation_predictions[0][i, :, ...].squeeze()
                target_i = batch_complete["target"][i, :, 0].squeeze()
                x_axis = np.arange(target_i.shape[0])
                ax.plot(x_axis[-cfg.dataset.prediction_length:], prediction, label="Prediction")
                ax.plot(x_axis[-cfg.dataset.prediction_length:], foundation_prediction, label="Zeor Shot Prediction")
                ax.plot(x_axis, target_i, label="Target")
                ax.set_title(f"Prediction vs Target {i}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Values")
                ax.legend()
            else:
                fig, axs = plt.subplots(2, batch_complete["target"].shape[-1] // 2, figsize=(18, 6))
                axs = axs.flatten()
                for j, ax in enumerate(axs):
                    prediction = predictions[j][i, :, ...]
                    print(prediction.shape)
                    target_i = batch_complete["target"][i, :, j]
                    x_axis = np.arange(target_i.shape[0])
                    ax.plot(x_axis[-cfg.dataset.prediction_length:], prediction, label="Prediction")
                    ax.plot(x_axis, target_i, label="Target")
                    ax.set_title(f"Prediction vs Target {i}")
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Values")
                    ax.legend()
        plt.show()
    
        # plt.figure(figsize=(10, 6))
        # plt.hist(np.asarray(batch["target"]).flatten(), bins=30, edgecolor='black')
        # plt.title("Target Histogram")
        # plt.xlabel("Target Value")
        # plt.ylabel("Frequency")
        # plt.grid(True)
        # plt.show()


        # plt.figure(figsize=(10, 6))
        # plt.hist(np.asarray(losses), bins=30, edgecolor='black')
        # plt.title("Loss Histogram")
        # plt.xlabel("Loss Value")
        # plt.ylabel("Frequency")
        # plt.grid(True)
        # plt.show()

        # pred = pred[0]
        # print(loss_value)

        # print(pred[0, ...])
        # print(batch["target"][0, ...])


if __name__ == "__main__":
    load_and_run_model()