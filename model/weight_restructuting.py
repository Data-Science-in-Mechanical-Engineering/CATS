import jax 
import jax.numpy as jnp
import equinox as eqx

from munkres import Munkres

import einshape as es

from utils.splitting_utils import get_neuron_slices


def reorder(w, num_devices):
    w_tiled = es.jax_einshape("(na)b->nab", jnp.abs(w), n=num_devices, m=num_devices)

    w_sum = jnp.sum(w_tiled, axis=1, keepdims=True)

    w_cost = es.jax_einshape("nab->(na)b", w_sum, n=num_devices, m=num_devices)

    w_cost = jnp.repeat(w_cost, w_cost.shape[1] // w_cost.shape[0], axis=0)

    # plot_matrix(w_cost)

    m = Munkres()
    indexes = m.compute((-jnp.abs(w_cost)).tolist())
    indexes = jnp.array([i[1] for i in indexes])

    # plot_matrix(w_cost[:, indexes])

    return w[:, indexes], indexes


def restructure_layernorm(layer_norms, input_permutation):
    layer_norm_weights = [l.weight for l in layer_norms]
    layer_norm_biases = [l.bias for l in layer_norms]

    layer_norm_weight = jnp.concatenate(layer_norm_weights, axis=0)
    layer_norm_bias = jnp.concatenate(layer_norm_biases, axis=0)

    feature_dim = layer_norm_weight.shape[0]
    num_devices = len(layer_norms)

    layer_norm_weight = layer_norm_weight[input_permutation]
    layer_norm_bias = layer_norm_bias[input_permutation]

    new_layer_norms = []
    slices = get_neuron_slices(num_neurons=feature_dim, num_devices=num_devices)
    for i in range(num_devices):
        w_temp = layer_norm_weight[slices[i]:slices[i+1]]
        b_temp = layer_norm_bias[slices[i]:slices[i+1]]

        new_layer_norms.append(eqx.tree_at(lambda pt: (pt.weight, pt.bias), 
                                            layer_norms[i], 
                                            (w_temp, b_temp)))
        
    return new_layer_norms