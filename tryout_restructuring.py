import jax
import jax.numpy as jnp
import mpx

def my_func(x):
    my_const = jnp.zeros((42), dtype=jnp.bfloat16)
    x = x+my_const
    print(x.dtype)
    return x

my_func(jnp.zeros((42, )))
mpx.cast_function(my_func, dtype=jnp.bfloat16)(jnp.zeros((42, )))

# import jax 
# import jax.numpy as jnp
# import hydra
# from tqdm import tqdm

# import equinox as eqx
# from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray 

# from model.timeseries_decoder import TimeseriesPatchedDecoder

# from munkres import Munkres

# import einshape as es

# import numpy as np
# import matplotlib.pyplot as plt

# from dataset_loaders.Monash import Monash


# def plot_matrix(matrix):
#     # Reorder center of mass using sorted indices
#     # center_of_mass_sorted = jnp.array(center_of_mass)[sorted_indices]
    
#     # Create heatmap and overlay center of mass
#     plt.figure(figsize=(8, 6))    
#     # plt.scatter(jnp.arange(len(sorted_indices)), center_of_mass_sorted, 
#     #             color='red', marker='o', label='Center of Mass')
#     plt.legend()
#     plt.imshow(matrix, cmap='gray_r', aspect='auto')
#     plt.colorbar(label='Weight value')
#     plt.title('Heatmap of absolute weights')
#     plt.xlabel('Sorted indices')
#     plt.ylabel('Input dimension')
#     plt.show()


# def reorder(w, num_devices):
#     w_tiled = es.jax_einshape("(na)b->nab", jnp.abs(w), n=num_devices, m=num_devices)

#     w_sum = jnp.sum(w_tiled, axis=1, keepdims=True)

#     w_cost = es.jax_einshape("nab->(na)b", w_sum, n=num_devices, m=num_devices)

#     w_cost = jnp.repeat(w_cost, w_cost.shape[1] // w_cost.shape[0], axis=0)

#     # plot_matrix(w_cost)

#     m = Munkres()
#     indexes = m.compute((-np.abs(w_cost)).tolist())
#     indexes = [i[1] for i in indexes]

#     # plot_matrix(w_cost[:, indexes])

#     return w[:, indexes], indexes


# @eqx.filter_jit
# def predict_batch(model: eqx.Module, 
#                   batch: dict,
#                   weight_scalings: dict | None,
#                   activation_scalings: dict | None,
#                   inference: bool, 
#                   key: PRNGKeyArray) -> Array:
#     subkeys = jax.random.split(key, len(batch["input"]))
#     print("b")
#     pred = jax.vmap(model, (0, 0 if batch["padding_mask"] is not None else None, None, None, None, 0))(batch["input"], 
#                                                                                                        batch["padding_mask"], 
#                                                                                                        weight_scalings, 
#                                                                                                        activation_scalings, 
#                                                                                                        inference, 
#                                                                                                        subkeys)
    
#     return pred


# @hydra.main(config_path="parameters", config_name="main", version_base="1.1")
# def load_and_run_model(cfg):
#     # load dataset
#     data_source = Monash(name="london_smart_meters_dataset",
#                      dataset_base_path="/data/datasets",
#                      context_length=512,
#                      prediction_length=96,
#                      stride=255
#                      )
    
#     test_dataset = data_source.init_dataloader_test(data_source.test_data_source, 1, 100, seed=1)


#     # load model
#     model = TimeseriesPatchedDecoder(input_dim=1, cfg=cfg, key=jax.random.PRNGKey(0))
#     # model = model.load_model("/data/distributed_transformer/test_syntheticV2/SyntheticV2/0/checkpoints/best_000", absolute_path=True)

#     model_restructured = model.restructure()
    
#     # model_restructured.save_model("/data/distributed_transformer/test_syntheticV2/SyntheticV2/0/checkpoints/best_000_restructured", absolute_path=True)

#     idx = 0
#     key = jax.random.PRNGKey(0)

#     for idx in tqdm(range(100)):
#         batch = next(test_dataset)
#         key, subkey = jax.random.split(key)
#         batch["padding_mask"] = None

#         pred, recorded_activations = predict_batch(model, batch, None, None, True, key)
#         pred_restructured, _ = predict_batch(model_restructured, batch, None, None, True, key)

#         # reversible input normalization
#         if type(pred) is tuple:
#             prediction = pred[0][0, -1, ...]
#             prediction_restructured = pred_restructured[0][0, -1, ...]
#         else:
#             prediction = pred[0, -1, ...]
#             prediction_restructured = pred_restructured[0, -1, ...]

#         target = batch["target"][0, -200:, ...]
#         x_axis = np.arange(target.shape[0])
#         import matplotlib.pyplot as plt

#         prediction_sorted = np.sort(prediction[:, 0], axis=-1)
#         prediction_restructured_sorted = np.sort(prediction_restructured[:, 0], axis=-1)

#         plt.figure(figsize=(10, 6))
#         plt.plot(prediction_sorted, label="Sorted Prediction")
#         plt.plot(prediction_restructured_sorted, label="Restructured Sorted Prediction", linestyle="--")
#         plt.title("Comparison of Sorted Predictions")
#         plt.xlabel("Index")
#         plt.ylabel("Sorted Value")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         # if prediction.shape[-1] == 1:
#         #     plt.figure(figsize=(10, 6))
#         #     plt.plot(x_axis[-cfg.dataset.prediction_length:], prediction[:, 0], label="Prediction")
#         #     plt.plot(x_axis[-cfg.dataset.prediction_length:], prediction_restructured[:, 0],
#         #                 label="Restructured Prediction", linestyle="--")
#         #     plt.plot(x_axis, target[:, 0], label="Target")
#         #     plt.legend()
#         #     plt.title("Prediction vs Target Comparison")
#         #     plt.xlabel("Time Steps")
#         #     plt.ylabel("Values")
#         #     plt.grid()
#         #     plt.show()
#         # else:
#         #     raise NotImplementedError("Multi-dimensional predictions are not supported in this example.")

#     # num_devices = 8

#     # indexes = None
    
#     # for layer_name in weights:
#     #     for k2 in weights[layer_name]:
#     #         w = jnp.abs(weights[layer_name][k2]["weights"])

#     #         if indexes is not None:
#     #             w = w[indexes, :]

#     #         t = jnp.ones((w.shape[0] // num_devices, w.shape[1] // num_devices))

#     #         t = jnp.block(
#     #             [[t if i == j else jnp.zeros_like(t) for j in range(num_devices)]
#     #             for i in range(num_devices)]
#     #         )

#     #         # t = 1 - t

#     #         print((t*w).sum())

#     #         w, indexes = reorder(w, num_devices)

#     #         # num_steps = 4
#     #         # for i in range(num_steps):
#     #         #     w = reorder(w, num_devices)
#     #         #     w = w.T

#     #         # if num_steps % 2 == 1:
#     #             # w = w.T

#     #         print((t*w).sum())

#     #         plot_matrix(w)



# if __name__ == "__main__":
#     import jax
#     import jax.numpy as jnp
#     import mpx

#     def my_func(x):
#         x = x+1
#         print(x.dtype)
#         return x

#     my_func(jnp.zeros((42, )))
#     mpx.cast_function(my_func(jnp.zeros((42, ))), dtype=jnp.bfloat16)


#     # n = 32
#     # x = jnp.zeros((n * 4, 2))

#     # x = jnp.reshape(x, (4, n, 2))

#     # def body_fun(carry, x):
#     #     return carry, jnp.mean(x)
    
#     # result, _ = jax.lax.scan(body_fun, (jnp.zeros((n, 2)), 1), x)
#     # print(result)
    
#     # load_and_run_model()