"""
Implements the basics of transformers.
All modules are for non-batched inputs. To create calls for batched data use vmap.
"""


import copy
import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
import omegaconf
import einshape as es

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

import utils.jax_utils as ju
from model.pruning import InterDevicePruningMask, ContainsPruning
from model.dropout import ContainsDropout, PartialLayerDropoutMask
from model.layernorm import PartialLayerNorm
from model.quantization import quantize_backward, quantize_forward, quantize_forward_backward
from model.weight_restructuting import reorder, restructure_layernorm

from utils.splitting_utils import get_neuron_slices, get_slice_indices_from_split, get_weight_slices, split_neurons

import mpx



def init_weights(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    lim = 1 / jnp.sqrt(shape[0])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def linear_layer(x: Array, W: Array, b: Array) -> Array:
    return x @ W + b


def calculate_mask(shape: tuple[int, ...], 
                   pruning: InterDevicePruningMask, 
                   partial_layer_dropout: PartialLayerDropoutMask,
                   d_key,
                   split_output=None) -> Array:
    mask = pruning.get_mask(shape, split_output)
    mask *= partial_layer_dropout.get_mask(shape, d_key, split_output)
    mask = mask.astype(dtype=jnp.float16)
    return mask


def apply_layer_norms(x, layer_norms):
    input_splits = get_slice_indices_from_split(split_neurons(len(layer_norms), len(x)))
    for i in range(len(layer_norms)):
        x = x.at[input_splits[i]:input_splits[i+1]].set(layer_norms[i](x[input_splits[i]:input_splits[i+1]]))
    return x



class SavableModel:
    def save_model(self, filename, absolute_path=False):
        """Save the model to a file.
        
        Returns:
            The filename of the saved model.
        """
        if not absolute_path:
            eqx.tree_serialise_leaves(f"checkpoints/{filename}.eqx", self)
            pruning_dict = self.get_pruning_state_dict()
            with open(f"checkpoints/{filename}.pkl", "wb") as f:
                pickle.dump(pruning_dict, f)
        else:
            eqx.tree_serialise_leaves(f"{filename}.eqx", self)
            pruning_dict = self.get_pruning_state_dict()
            with open(f"{filename}.pkl", "wb") as f:
                pickle.dump(pruning_dict, f)

    def load_model(self, filename, absolute_path=False):
        """Load the model from a file.
        
        Returns:
            The filename of the loaded model.
        """
        if not absolute_path:
            with open(f"checkpoints/{filename}.pkl", "rb") as f:
                pruning_dict = pickle.load(f)
            model = eqx.tree_deserialise_leaves(f"checkpoints/{filename}.eqx", self)
        else:
            with open(f"{filename}.pkl", "rb") as f:
                pruning_dict = pickle.load(f)
            model = eqx.tree_deserialise_leaves(f"{filename}.eqx", self)
        model = model.set_pruning_state_dict(pruning_dict)
        return model



class DenseLayer(eqx.Module):
    weights: Array
    bias: Array

    def __init__(self, 
                 input_dim: Int, 
                 output_dim: Int, 
                 key: PRNGKeyArray):
        key, subkey = jax.random.split(key)
        self.weights = init_weights(subkey, (input_dim, output_dim))
        self.bias = jnp.zeros((output_dim,))

    def __call__(self, inputs: Array, weight_scalings: dict | None, activation_scalings: dict | None, mask: Array | None) -> Array:
        # Apply mask if provided
        weights = self.weights
        if mask is not None:
            weights = weights * mask
        
        if weight_scalings is None:
            return inputs @ weights + self.bias

        # weights = quantize_forward(weights, weight_scalings["weights"])
        weights = quantize_forward_backward(weights, weight_scalings["weights"])
        inputs = quantize_forward_backward(inputs, activation_scalings)
        bias = self.bias  # quantize_forward_backward(self.bias, weight_scalings["weights"] * activation_scalings)
        # inputs = quantize_forward(inputs, activation_scalings)
        # bias = quantize_forward(self.bias, weight_scalings["weights"] * activation_scalings)
        result = inputs @ weights + bias
        # result = quantize_backward(result, weight_scalings["weights"] * activation_scalings)

        return result
        # return quantize_backward(result, activation_scalings *  weight_scalings["weights"]) + self.bias

    
    def get_output_for_device(self, inputs,  weight_scalings: dict | None, activation_scalings: dict | None, mask: Array | None, device_index: Int, output_slices: list) -> Array:
        weights = self.weights
        if mask is not None:
            weights = weights * mask

        weight_partial = weights[:, output_slices[device_index]:output_slices[device_index+1]]
        bias_partial = self.bias[output_slices[device_index]:output_slices[device_index+1]]

        if weight_scalings is None:
            return inputs @ weight_partial + bias_partial

        weight_partial = quantize_forward(weight_partial, weight_scalings["weights"])
        inputs = quantize_forward(inputs, activation_scalings)
        bias_partial = quantize_forward(bias_partial, weight_scalings["weights"] * activation_scalings)
        result = inputs @ weight_partial + bias_partial
        result = quantize_backward(result, weight_scalings["weights"] * activation_scalings)
        return result
        # return quantize_backward(result, activation_scalings * weight_scalings["weights"]) + bias_partial
    
    def get_weight_dict(self) -> dict:
        """Returns the weights of the layer as a dictionary."""
        return {
            "weights": self.weights,
            "bias": self.bias
        }

    def get_number_of_parameters(self) -> int:
        """Returns the number of parameters in the layer."""
        return self.weights.size + self.bias.size
    
    def reorder_inputs(self, input_permutation):
        if input_permutation is not None:
            new_weights = self.weights[input_permutation, :]
            return eqx.tree_at(lambda pt: pt.weights, self, new_weights)
        return self
    
    def reorder_outputs(self, output_permutation):  
        """Reorders the outputs of the layer according to the output permutation."""
        if output_permutation is not None:
            new_weights = self.weights[:, output_permutation]
            new_bias = self.bias[output_permutation]
            return eqx.tree_at(lambda pt: (pt.weights, pt.bias), self, (new_weights, new_bias))
        return self
    
    def restructure(self, num_devices, input_permutation=None):
        """Restructures the layer by permuting the weights according to the input permutation."""
        if input_permutation is not None:
            new_weights = self.weights[input_permutation, :]
        else:
            new_weights = self.weights

        
        new_weights, perm = reorder(new_weights, num_devices)
        assert new_weights.shape[0] == self.weights.shape[0], f"Layer {self} has a different input dimension after restructuring. Expected {self.weights.shape[0]}, got {new_weights.shape[0]}."
        assert new_weights.shape[1] == self.weights.shape[1], f"Layer {self} has a different output dimension after restructuring. Expected {self.weights.shape[1]}, got {new_weights.shape[1]}."
        new_bias = self.bias[perm]
        return eqx.tree_at(lambda pt: (pt.weights, pt.bias), self, (new_weights, new_bias)), perm
    


class ResidualBlock(eqx.Module, ContainsPruning, ContainsDropout):
    """A residual block module that applies a series of dense layers, dropout, and layer normalization 
    with an optional transformed residual connection.

    i.e. y = LayerNorm(MLP(x) + Residual(x)), where Resiudual(x) = x if transform_residual is False and Residual(x) = DenseLayer(x) if transform_residual is True

    Attributes:
        layers (list[eqx.Module]): A list of dense layers that process the input sequentially.
        residual_layer (eqx.Module | None): An optional dense layer for transforming the residual 
            connection. If `None`, the residual connection is an identity mapping.
        layernorm (eqx.nn.LayerNorm): A layer normalization module applied to the output.
        dropout (eqx.nn.Dropout): A dropout module applied to the output for regularization.
    """

    layers: list[DenseLayer]
    pruning_masks: list[InterDevicePruningMask]
    use_residual: bool
    residual_layer: DenseLayer | None
    dropout: eqx.nn.Dropout

    partial_layer_dropout: list[PartialLayerDropoutMask]

    disable_pruning_partial_layer_dropout: bool   # if the disable them, then the training is much more memory efficient, as we can use normal layernorm.

    num_devices: Int

    partial_layer_norm: PartialLayerNorm

    activation_fn: callable

    def __init__(self, 
                 input_dim: Int,
                 output_dim: Int,
                 feature_dim: Int,
                 dropout_rate: Float,
                 partial_layer_dropout_prob_mode1: Float,
                 partial_layer_dropout_prob_mode2: Float,
                 partial_layer_dropout_prob_mode3: Float,
                 key: PRNGKeyArray,
                 transform_residual: bool, # use dense layer instead of identity for residual connection
                 num_devices: Int,
                 disable_pruning_partial_layer_dropout: bool,
                 prune_completely: bool = False,
                 use_residual: bool = True,
                 use_layernorm: bool = True,
                 num_layers: Int = 2,
                 activation="relu"):
        """
        Args:
            input_dim (Int): The dimensionality of the input features.
            output_dim (Int): The dimensionality of the output features.
            feature_dim (Int): The dimensionality of the intermediate features in the dense layers.
            dropout_rate (Float): The dropout rate for regularization.
            key (PRNGKeyArray): A JAX random key for initializing the layers.
            transform_residual (bool): If `True`, applies a dense layer to transform the residual 
                connection. If `False`, uses an identity mapping for the residual connection.
            num_layers (Int, optional): The number of dense layers in the block. Must be at least 2. 
                Defaults to 2.
        """

        # init layers
        layers = []
        pruning_masks = []
        key, subkey = jax.random.split(key)
        layers.append(DenseLayer(input_dim, 
                                    feature_dim,
                                    subkey))
        pruning_masks.append(InterDevicePruningMask(input_dim, num_devices))
        if num_layers >= 2:
            for _ in range(num_layers - 2):
                key, subkey = jax.random.split(key)
                layers.append(DenseLayer(feature_dim, 
                                    feature_dim,
                                    subkey))
                pruning_masks.append(InterDevicePruningMask(feature_dim, num_devices))
            
            key, subkey = jax.random.split(key)
            layers.append(DenseLayer(feature_dim, 
                                output_dim, 
                                subkey))
            pruning_masks.append(InterDevicePruningMask(feature_dim, num_devices))
        # print(prune_completely)
        if prune_completely:
            for l, p in zip(layers, pruning_masks):
                p.prune_step(jnp.array([l.weights]), 1.0)
                # print(p.pruning_sub_mat)
        
        self.pruning_masks = pruning_masks
        self.layers = layers

        self.partial_layer_dropout = [PartialLayerDropoutMask(num_devices, 
                                                                  dropout_prob_mode1=partial_layer_dropout_prob_mode1, 
                                                                  dropout_prob_mode2=partial_layer_dropout_prob_mode2, 
                                                                  dropout_prob_mode3=partial_layer_dropout_prob_mode3) for _ in range(num_layers)]

        assert transform_residual or input_dim == output_dim, "input_dim must be equal to output_dim if transform_residual is False"
        if transform_residual:
            self.residual_layer = DenseLayer(input_dim, output_dim, key)
        else:
            self.residual_layer = None

        # init layernorm and dropout
        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.num_devices = num_devices
        if use_layernorm:
            self.partial_layer_norm = PartialLayerNorm(input_dim)
        else:
            self.partial_layer_norm = None

        if activation == "relu":
            self.activation_fn = jax.nn.relu
        elif activation == "tanh":
            self.activation_fn = jax.nn.tanh
        elif activation == "gelu":
            self.activation_fn = jax.nn.gelu

        self.use_residual = use_residual

        self.disable_pruning_partial_layer_dropout = disable_pruning_partial_layer_dropout

    def __call__(self, inputs: Array, weight_scalings: dict, activation_scalings: dict, inference: bool, key: PRNGKeyArray, record_activations=False) -> Array:
        """
        Applies the residual block to the input tensor. The input is passed through the dense 
        layers, dropout, and layer normalization, with a residual connection added.
        Args:
            inputs (Array): The input tensor to the residual block.
            inference (bool): Whether the model is in inference mode. If `True`, disables dropout.
            key (PRNGKeyArray): A JAX random key for dropout.
        Returns:
            Array: The output tensor after applying the residual block.
        """

        # just a comfort thing so we do not have to check if activation_scalings is None
        if activation_scalings is None:
            activation_scalings = {f"{i}": None for i in range(len(self.layers)+1)}
            activation_scalings["in"] = None
            activation_scalings["residual"] = None
            activation_scalings["sum"] = None
            weight_scalings = {f"{i}": None for i in range(len(self.layers))}
            weight_scalings["residual"] = None
        
        recorded_activations = {f"{i}": None for i in range(len(self.layers) + 1)}
        recorded_activations["residual"] = None
        recorded_activations["sum"] = None
        # resample the dropout masks
        d_keys = []
        for d in self.partial_layer_dropout:
            key, key2 = jax.random.split(key)
            d_keys.append(d.sample(key2))
        
        recorded_activations["in"] = inputs

        # first layer 
        if self.partial_layer_norm is not None:
            # inputs_after_layernorm = apply_layer_norms(inputs, self.layer_norms)
            # recorded_activations["0"] = inputs_after_layernorm

            # mask = calculate_mask(self.layers[0].weights.shape, 
            #                   self.pruning_masks[0], 
            #                   self.partial_layer_dropout[0], 
            #                   d_keys[0])
            # outputs = self.layers[0](inputs_after_layernorm, mask=mask, weight_scalings=weight_scalings["0"], activation_scalings=activation_scalings["0"])
            
            # if we disable the masks, we can calculate the layernorm the normal way.
            if self.disable_pruning_partial_layer_dropout:
                inputs = quantize_forward_backward(inputs, activation_scalings["in"])
                inputs_after_layernorm = mpx.force_full_precision(self.partial_layer_norm.normal_layer_norm, return_dtype=inputs.dtype)(inputs)
                recorded_activations["0"] = inputs_after_layernorm
                outputs = self.layers[0](inputs_after_layernorm, weight_scalings["0"], activation_scalings["0"], mask=None)
            else:
                # if not, we have to do the complicated and also memory inefficient way.
                mask = calculate_mask((self.layers[0].weights.shape[0], self.num_devices), 
                                self.pruning_masks[0], 
                                self.partial_layer_dropout[0], 
                                d_keys[0])
                inputs = quantize_forward_backward(inputs, activation_scalings["in"])
                inputs_after_layernorm = mpx.force_full_precision(self.partial_layer_norm.partial_layer_norm, return_dtype=inputs.dtype)(inputs, mask)
                recorded_activations["0"] = inputs_after_layernorm
                outputs = self.partial_layer_norm.partial_layer_normed_data_through_dense(inputs_after_layernorm, weight_scalings["0"], activation_scalings["0"], self.layers[0])
        else:
            mask = calculate_mask(self.layers[0].weights.shape, 
                              self.pruning_masks[0], 
                              self.partial_layer_dropout[0], 
                              d_keys[0])
            recorded_activations["0"] = inputs
            outputs = self.layers[0](inputs, mask=mask, weight_scalings=weight_scalings["0"], activation_scalings=activation_scalings["0"])
            # if activation_scalings["1"] is not None:
            #     print(quantize_forward(outputs, activation_scalings["1"]))
        
        if len(self.layers) >= 2:
            key, subkey = jax.random.split(key)
            # outputs = self.dropout(outputs, inference=inference, key=subkey)
            # jax.debug.print("outputs: {}", jnp.sort(outputs)[0:8])
            outputs = self.activation_fn(outputs)

            for i, (layer, pruning_mask, d, d_key) in enumerate(zip(self.layers[1:-1], 
                                                                    self.pruning_masks[1:-1], 
                                                                    self.partial_layer_dropout[1:-1], 
                                                                    d_keys[1:-1])):
                mask = calculate_mask(layer.weights.shape, 
                                    pruning_mask, 
                                    d, 
                                    d_key)
                recorded_activations[f"{i+1}"] = outputs
                key, subkey = jax.random.split(key)
                # outputs = self.dropout(outputs, inference=inference, key=subkey)
                outputs = self.activation_fn(layer(outputs, mask=mask, weight_scalings=weight_scalings[f"{i+1}"], activation_scalings=activation_scalings[f"{i+1}"]))

            mask = calculate_mask(self.layers[-1].weights.shape, 
                                self.pruning_masks[-1], 
                                self.partial_layer_dropout[-1], 
                                d_keys[-1])
            recorded_activations[f"{len(self.layers) - 1}"] = outputs
            

            outputs = self.layers[-1](outputs, 
                                    mask=mask, 
                                    weight_scalings=weight_scalings[f"{len(self.layers) - 1}"], 
                                    activation_scalings=activation_scalings[f"{len(self.layers) - 1}"])
            recorded_activations[f"{len(self.layers)}"] = outputs

            outputs = self.dropout(outputs, inference=inference, key=key)

            if self.residual_layer is not None:
                # the devices transmit the first input only once. Hence pruning/dropout for the first and residual layer is the same
                residual = 0
                mask = calculate_mask(self.residual_layer.weights.shape, 
                        self.pruning_masks[0], 
                        self.partial_layer_dropout[0],
                        d_keys[0])
                residual = self.residual_layer(inputs, weight_scalings["residual"], activation_scalings["in"],  mask)
                recorded_activations["residual"] = residual
                outputs = quantize_forward_backward(outputs, activation_scalings[f"{len(self.layers)}"])
                residual = quantize_forward_backward(residual, activation_scalings["residual"])
                # if activation_scalings["residual"] is not None:
                #     print(quantize_forward(residual, activation_scalings["residual"]))
                outputs = outputs + residual
                recorded_activations["sum"] = outputs
            else:
                # if the input and output dimensions are the same, we do not need pruning as the devices
                # already have the same splits
                outputs = quantize_forward_backward(outputs, activation_scalings[f"{len(self.layers)}"])
                inputs = quantize_forward_backward(inputs, activation_scalings["in"])
                recorded_activations["residual"] = inputs
                outputs = outputs + inputs 
                recorded_activations["sum"] = outputs
            
            # outputs = quantize_forward_backward(outputs, activation_scalings["sum"])
            # no layernorm, as we put this into the input of the attention layer, because of message loss.
            return outputs, recorded_activations
        else:
            recorded_activations["sum"] = outputs
            # outputs = quantize_forward_backward(outputs, activation_scalings["sum"])
            return outputs, recorded_activations
        
    def get_weight_dict(self) -> dict:
        """Returns the weights of the residual block as a dictionary."""
        weight_dict = {}
        for i, layer in enumerate(self.layers):
            weight_dict[f"{i}"] = layer.get_weight_dict()
        if self.residual_layer is not None:
            weight_dict["residual"] = self.residual_layer.get_weight_dict()
        if self.partial_layer_norm is not None:
            weight_dict["partial_layer_norm"] = self.partial_layer_norm.get_weight_dict()
        return weight_dict
    

    def restructure(self, input_permutation):
        if self.partial_layer_norm is not None:
            new_partial_layer_norm = self.partial_layer_norm.restructure(input_permutation)
        
        if self.residual_layer is not None:
            new_residual_layer = self.residual_layer.reorder_inputs(input_permutation)
        else:
            new_residual_layer = None
            # assert new_layer.weights.shape[0] == l.weights.shape[0], f"Layer {l} has a different input dimension after restructuring. Expected {l.weights.shape[0]}, got {new_layer.weights.shape[0]}."
            # assert new_layer.weights.shape[1] == l.weights.shape[1], f"Layer {l} has a different output dimension after restructuring. Expected {l.weights.shape[1]}, got {new_layer.weights.shape[1]}."

        if self.residual_layer is not None:
            new_layers = []
            perm = input_permutation
            for l in self.layers:
                new_layer, perm = l.restructure(num_devices=self.num_devices, input_permutation=perm)
                new_layers.append(new_layer)
            new_residual_layer = new_residual_layer.reorder_outputs(perm)
            if self.partial_layer_norm is not None:
                return eqx.tree_at(lambda pt: (pt.layers, pt.residual_layer, pt.partial_layer_norm), self, (new_layers, new_residual_layer, new_partial_layer_norm)), perm
            else:
                return eqx.tree_at(lambda pt: (pt.layers, pt.residual_layer), self, (new_layers, new_residual_layer)), perm
        else:
            new_layers = []
            perm = input_permutation
            for l in self.layers[:-1]:
                new_layer, perm = l.restructure(num_devices=self.num_devices, input_permutation=perm)
                new_layers.append(new_layer)

            new_layer = self.layers[-1].reorder_inputs(perm)
            new_layer = new_layer.reorder_outputs(input_permutation)  # because of residual connection
            perm = input_permutation
            new_layers.append(new_layer)
            if self.partial_layer_norm is not None:
                return eqx.tree_at(lambda pt: (pt.layers, pt.partial_layer_norm), self, (new_layers, new_partial_layer_norm)), perm
            else:
                return eqx.tree_at(lambda pt: pt.layers, self, new_layers), perm

    
    def get_pruning_weights(self):
        return (jnp.array([l.weights]) for l in self.layers)

    def where_pruning_masks(self, pt):
        return tuple(pt.pruning_masks)
    
    def where_dropout_masks(self, pt):
        return tuple(pt.partial_layer_dropout)
    
    def get_number_of_parameters(self) -> int:
        """Returns the number of parameters in the residual block."""
        num_params = 0
        for layer in self.layers:
            num_params += layer.get_number_of_parameters()
        if self.residual_layer is not None:
            num_params += self.residual_layer.get_number_of_parameters()
        return num_params
    

def apply_rotary_positional_encoding(inputs: Array):
    """
    Applies rotary positional encoding to the input tensor. (https://arxiv.org/pdf/2104.09864)
    Args:
        inputs (Array): The input tensor to apply positional encoding to. Shape is (n, d).
    """
    assert inputs.shape[1] % 2 == 0, "inputs.shape[1] must be divisible by 2 for rotary positional encoding"
    d = inputs.shape[1]

    # notation: _ denotes a list of the formula symbols used in the paper
    i_ = jnp.arange(inputs.shape[1] // 2) + 1   # 1, 2, 3, ..., d/2
    phi_ = 10_000 ** (-2 * ((i_-1) / d)) # phi_1, phi_2, ..., phi_d/2

    def _rotary_encoding(m):
        """
        rotary encopding for a single feature at position m (i.e., inputs[m, :])
        """
        x = inputs[m, :]
        x1, x2 = x[0:d:2], x[1:d:2]
        
        cos_mphi_ = jnp.cos(m * phi_)   # cos(m*phi_1), cos(m*phi_2), ..., cos(m*phi_d/2)
        sin_mphi_ = jnp.sin(m * phi_)   # sin(m*phi_1), sin(m*phi_2), ..., sin(m*phi_d/2)

        odd_rows_ = x1 * cos_mphi_ - x2 * sin_mphi_   # x1[1] * cos(m*phi_1) - x2[1] * sin(m*phi_1), x1[2] * cos(m*phi_2) - x2[2] * sin(m*phi_2), ..., x1[d/2] * cos(m*phi_d/2) - x2[d/2] * sin(m*phi_d/2)
        even_rows_ = x1 * sin_mphi_ + x2 * cos_mphi_

        result = jnp.ones((len(x),))
        result = result.at[0:d:2].set(odd_rows_)
        result = result.at[1:d:2].set(even_rows_)

        return result
    
    # apply rotary encoding to all positions
    return jax.vmap(_rotary_encoding)(jnp.arange(inputs.shape[0]))


def rope_positional_attention_scores(q, k):
    q = apply_rotary_positional_encoding(q)
    k = apply_rotary_positional_encoding(k)
    attention_scores = q @ k.T / jnp.sqrt(q.shape[-1])
    return attention_scores


def rbf_positional_attention_scores(q, k, length_scale):
    qk = q @ k.T / jnp.sqrt(q.shape[-1])

    indexes = jnp.arange(q.shape[0])

    def rbf_kernel(x1, x2, length_scale):
        return jnp.exp(-0.5 * jnp.square((x1 - x2) / length_scale))
    

    
    calc_encodings = jax.vmap(rbf_kernel, in_axes=(0, None, None))

    encodings = calc_encodings(indexes, indexes, length_scale)

    return (encodings) * qk

def init_kernel_hyperparams(key: PRNGKeyArray) -> PyTree:
    return {
        "periodic_kernel_weight": jax.random.uniform(key, (1,), minval=0.0, maxval=1.0),
        "periodic_length_scale": jax.random.uniform(key, (1,), minval=0.0, maxval=1.0),
        "rational_quadratic_kernel_weight": jax.random.uniform(key, (1,), minval=0.0, maxval=10.0),
        "rational_quadratic_length_scale": jax.random.uniform(key, (1,), minval=0.0, maxval=10.0)
    }

def kernel_positional_attention_scores(q, k, hyperparams):  
    def periodic_kernel(x1, x2, length_scale):
        return jnp.exp(-2 * jnp.square(jnp.sin(jnp.pi * jnp.abs(x1 - x2) / length_scale)))
    
    def rational_quadratic_kernel(x1, x2, length_scale):
        return (1 + jnp.square(x1 - x2) / (2 * length_scale)) ** (-length_scale)
    
    def kernel(x1, x2):
        return (hyperparams["periodic_kernel_weight"] * periodic_kernel(x1, x2, hyperparams["periodic_length_scale"]) + hyperparams["rational_quadratic_kernel_weight"] * rational_quadratic_kernel(x1, x2, hyperparams["rational_quadratic_length_scale"])) / (hyperparams["periodic_kernel_weight"] + hyperparams["periodic_kernel_weight"])
    
    indexes = jnp.arange(q.shape[0])
    calc_encodings = jax.vmap(kernel, in_axes=(0, None))

    return (calc_encodings(indexes, indexes) + 1) * (q @ k.T / jnp.sqrt(q.shape[-1]))



class MultiHeadAttentionBlock(eqx.Module, ContainsPruning, ContainsDropout):
    """MultiHeadAttentionBlock is a module that implements a multi-head attention mechanism, 
    commonly used in transformer architectures. It performs the following operations:
    1. Computes query (Q), key (K), and value (V) projections for each attention head.
    2. Applies scaled dot-product attention with optional causal masking to prevent 
        attending to future positions and attention dropout.
    3. Concatenates the outputs of all attention heads and projects them back to the 
        original feature dimension.
    4. Adds a residual connection and applies layer normalization.
    Attributes:
         weight_qs (Array): Weights for the query projections, shaped as 
              (num_heads, feature_dim, inner_dim).
         weight_ks (Array): Weights for the key projections, shaped as 
              (num_heads, feature_dim, inner_dim).
         weight_vs (Array): Weights for the value projections, shaped as 
              (num_heads, feature_dim, inner_dim).
         bias_qs (Array): Biases for the query projections, shaped as 
              (num_heads, inner_dim).
         bias_ks (Array): Biases for the key projections, shaped as 
              (num_heads, inner_dim).
         bias_vs (Array): Biases for the value projections, shaped as 
              (num_heads, inner_dim).
         weight_o (Array): Weights for the output projection, shaped as 
              (feature_dim, feature_dim).
         bias_o (Array): Biases for the output projection, shaped as 
              (feature_dim,).
         layernorm (eqx.nn.LayerNorm): Layer normalization applied after the residual 
              connection.
         dropout (eqx.nn.Dropout): Dropout applied to the attention scores.
    """

    dense_qs: DenseLayer   # (num_heads * feature_dim, inner_dim), i.e., w_q of head 2 is dense_qs.weight[feature_dim:2*feature_dim, :]
    dense_ks: DenseLayer
    dense_vs: DenseLayer
    pruning_attention: InterDevicePruningMask   # query, keys and values have the same pruning mask

    dense_o: DenseLayer
    pruning_o: InterDevicePruningMask

    partial_layer_dropout_attention: PartialLayerDropoutMask
    partial_layer_dropout_o: PartialLayerDropoutMask
    partial_layer_norm: PartialLayerNorm

    disable_pruning_partial_layer_dropout: bool   # if the disable them, then the training is much more memory efficient, as we can use normal layernorm.

    encoding_hyperparams: PyTree

    attention_scores: str
    num_heads: int
    feature_dim: int
    slice_heads: list[int]
    o_split_input: list[int]

    num_devices: int

    dropout: eqx.nn.Dropout

    def __init__(self, feature_dim: int, 
                 num_heads: int, 
                 dropout_rate: float, 
                 partial_layer_dropout_prob_mode1: float,
                 partial_layer_dropout_prob_mode2: float,
                 partial_layer_dropout_prob_mode3: float,
                 key: PRNGKeyArray, 
                 disable_pruning_partial_layer_dropout: bool,
                 num_devices: Int, 
                 attention_scores: str = "rotary"):
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.num_devices = num_devices
        self.partial_layer_dropout_attention = PartialLayerDropoutMask(num_devices, 
                                                                       dropout_prob_mode1=partial_layer_dropout_prob_mode1, 
                                                                       dropout_prob_mode2=partial_layer_dropout_prob_mode2, 
                                                                       dropout_prob_mode3=partial_layer_dropout_prob_mode3)
        
        head_split = split_neurons(num_devices, num_heads)   # every head is associated to one device.
        self.o_split_input = [h * feature_dim // num_heads for h in head_split]
        self.slice_heads = get_slice_indices_from_split(head_split)
        self.partial_layer_dropout_o = PartialLayerDropoutMask(num_devices,
                                                                dropout_prob_mode1=partial_layer_dropout_prob_mode1,
                                                                dropout_prob_mode2=partial_layer_dropout_prob_mode2,
                                                                dropout_prob_mode3=partial_layer_dropout_prob_mode3,
                                                                split_input=self.o_split_input,
                                                                )

        key, subkey = jax.random.split(key)
        self.dense_qs = DenseLayer(feature_dim, feature_dim, subkey)

        key, subkey = jax.random.split(key)
        self.dense_ks = DenseLayer(feature_dim, feature_dim, subkey)

        key, subkey = jax.random.split(key)
        self.dense_vs = DenseLayer(feature_dim, feature_dim, subkey)

        self.pruning_attention = InterDevicePruningMask(feature_dim, num_devices)

        key, subkey = jax.random.split(key)
        self.dense_o = DenseLayer(feature_dim, feature_dim, subkey)

        self.pruning_o = InterDevicePruningMask(feature_dim, num_devices, self.o_split_input)

        key, subkeys = ju.get_subkeys(key, num_heads)

        if attention_scores == "rotary":
            self.encoding_hyperparams = None
        elif attention_scores == "rbf":
            self.encoding_hyperparams = jax.vmap(init_weights, (0, None))(subkeys, (1,))
        elif attention_scores == "kernel":
            self.encoding_hyperparams =  jax.vmap(init_kernel_hyperparams)(subkeys)
        elif attention_scores == "vanilla":
            self.encoding_hyperparams = None
        elif attention_scores == "relative_learned":
            self.encoding_hyperparams = jax.vmap(init_weights, (0, None))(subkeys, (feature_dim // num_heads,))
        else:
            raise ValueError("Unknown attention_scores type")

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.partial_layer_norm = PartialLayerNorm(feature_dim)
        self.disable_pruning_partial_layer_dropout = disable_pruning_partial_layer_dropout

        self.attention_scores = attention_scores

    @staticmethod
    def attention(q: Array,
                  k: Array,
                  v: Array,
                  dropout: eqx.nn.Dropout,
                  key: PRNGKeyArray, 
                  inference: bool, 
                  mask: Array,
                  encoding_hyperparams: Array,
                  attention_scores,
                  activation_scalings) -> Array:
        """
        Computes the scaled dot-product attention for a single attention head. With rotary position encoding.
        Args:
            q (Array): Query tensor, shaped as (seq_len, inner_dim).
            k (Array): Key tensor, shaped as (seq_len, inner_dim).
            v (Array): Value tensor, shaped as (seq_len, inner_dim).
            dropout (eqx.nn.Dropout): Dropout module for regularization of attention scores.
            key (PRNGKeyArray): Random key for dropout.
            inference (bool): Whether the model is in inference mode (dropout disabled).
            mask (Array): Attention mask, shaped as (seq_len, seq_len).
            encoding_hyperparams (Array): Hyperparameters for the positional encoding.
            attention_scores (str): Type of positional encoding to use.
        Returns:
            Array: Output of the attention mechanism, shaped as 
            (seq_len, inner_dim)."""
        
        if attention_scores == "rotary":
            attention_scores = rope_positional_attention_scores(q, k)
        elif attention_scores == "rbf":
            attention_scores = rbf_positional_attention_scores(q, k, encoding_hyperparams)
        elif attention_scores == "kernel":
            attention_scores = kernel_positional_attention_scores(q, k, encoding_hyperparams)
        elif attention_scores == "vanilla":
            if activation_scalings["q"] is not None:
                q = quantize_forward(q, activation_scalings["q"])
                k = quantize_forward(k, activation_scalings["k"])

            attention_scores = q @ k.T

            if activation_scalings["q"] is not None:
                attention_scores = quantize_backward(attention_scores, activation_scalings["q"] * activation_scalings["k"])
                    
                attention_scores /= jnp.sqrt(q.shape[-1])
            else:
                attention_scores = attention_scores / jnp.sqrt(q.shape[-1])
        elif attention_scores == "relative_learned":
            attention_scores = q @ k.T / jnp.sqrt(q.shape[-1])
            attention_scores = q @ k.T / jnp.sqrt(q.shape[-1]) * (1 + encoding_hyperparams @ encoding_hyperparams.T)
        else:
            raise ValueError("Unknown attention_scores type")
        attention_scores_ = jnp.where(mask, attention_scores, -jnp.inf)
        attention_scores_ = mpx.force_full_precision(jax.nn.softmax, attention_scores_.dtype)(attention_scores_, axis=-1)
        attention_scores_ = dropout(attention_scores_, inference=inference, key=key)
        v = quantize_forward_backward(v, activation_scalings["v"])
        # print(attention_scores_ @ v)
        return attention_scores_ @ v


    def __call__(self, inputs: Array, weight_scalings: dict | None, activation_scalings: dict | None, inference: bool, key: PRNGKeyArray, use_causal_attention: bool) -> Array:
        """
        Applies the multi-head attention block to the input sequence.
        Args:
            inputs (Array): Input tensor, shaped as (seq_len, feature_dim).
            inference (bool): Whether the model is in inference mode.
            key (PRNGKeyArray): Random key for dropout and other stochastic operations.
            use_causal_attention (bool): Whether to use causal masking for attention.
        Returns:
            Array: Output tensor, shaped as (seq_len, feature_dim).
        """
        # just a comfort thing so we do not have to check if activation_scalings is None
        if activation_scalings is None:
            activation_scalings = {f"in": None, f"layer_norm": None, f"q": None, "k": None, "v": None, "qkv": None, "heads": None, "o":None, "sum": None}
            weight_scalings = copy.deepcopy(activation_scalings)

        recorded_activations = {f"in": None, f"layer_norm": None, f"q": None, "k": None, "v": None, "qkv": None, "heads": None, "o":None, "sum": None}

        key, key2 = jax.random.split(key)
        attention_d_key = self.partial_layer_dropout_attention.sample(key2)
        # mask = calculate_mask((self.dense_qs.weights.shape[0], self.dense_qs.weights.shape[1]),
        #                       self.pruning_attention,
        #                       self.partial_layer_dropout_attention,
        #                       attention_d_key, 
        #                       self.o_split_input)

        mask = calculate_mask((self.dense_qs.weights.shape[0], self.num_devices),
                              self.pruning_attention,
                              self.partial_layer_dropout_attention,
                              attention_d_key)

        recorded_activations["in"] = inputs    
        # if activation_scalings["in"] is not None:
        #     print("in")  
        #     print(quantize_forward(inputs[0:2, :], activation_scalings["in"], rounding=True))
        inputs = quantize_forward_backward(inputs, activation_scalings["in"])

        # if activation_scalings["in"] is not None:
        #      print(quantize_forward(inputs, activation_scalings["in"]))


        if self.disable_pruning_partial_layer_dropout:
            inputs_after_layernorm = jax.vmap(mpx.force_full_precision(self.partial_layer_norm.normal_layer_norm, return_dtype=inputs.dtype), in_axes=(0,))(inputs)
            recorded_activations["layer_norm"] = inputs_after_layernorm

            qs = self.dense_qs(inputs_after_layernorm, weight_scalings["q"], activation_scalings["layer_norm"], mask=None)
            ks = self.dense_ks(inputs_after_layernorm, weight_scalings["k"], activation_scalings["layer_norm"], mask=None)
            vs = self.dense_vs(inputs_after_layernorm, weight_scalings["v"], activation_scalings["layer_norm"], mask=None)
        else:
            inputs_after_layernorm = jax.vmap(mpx.force_full_precision(self.partial_layer_norm.partial_layer_norm, return_dtype=inputs.dtype), in_axes=(0, None))(inputs, mask)

            # print(inputs_after_layernorm[0:2, :])
            # if activation_scalings["in"] is not None:
            #     print(quantize_forward(inputs_after_layernorm[:, 0, :], activation_scalings["layer_norm"]))
            recorded_activations["layer_norm"] = inputs_after_layernorm
            qs = jax.vmap(self.partial_layer_norm.partial_layer_normed_data_through_dense, in_axes=(0, None, None, None, None))(
                inputs_after_layernorm, weight_scalings["q"], activation_scalings["layer_norm"], self.dense_qs, self.o_split_input)
            ks = jax.vmap(self.partial_layer_norm.partial_layer_normed_data_through_dense, in_axes=(0, None, None, None, None))(
                inputs_after_layernorm, weight_scalings["k"], activation_scalings["layer_norm"], self.dense_ks, self.o_split_input)
            vs = jax.vmap(self.partial_layer_norm.partial_layer_normed_data_through_dense, in_axes=(0, None, None, None, None))(
                inputs_after_layernorm, weight_scalings["v"], activation_scalings["layer_norm"], self.dense_vs, self.o_split_input)




        # inputs_after_layernorm = jax.vmap(apply_layer_norms, (0, None))(inputs, self.layer_norms)
        # qs = jax.vmap(self.dense_qs, (0, None, None, None))(inputs_after_layernorm, weight_scalings["q"], activation_scalings["q"], mask)
        # ks = jax.vmap(self.dense_ks, (0, None, None, None))(inputs_after_layernorm, weight_scalings["k"], activation_scalings["k"], mask)
        # vs = jax.vmap(self.dense_vs, (0, None, None, None))(inputs_after_layernorm, weight_scalings["v"], activation_scalings["v"], mask)

        # qs, ks, vs, inputs_after_layernorm = jax.vmap(self.calculate_attention_input_single_with_layernorm, (0, None, None, None, None))(inputs, weight_scalings["q"], activation_scalings["q"], self.dense_qs, mask)

        # reshape such that the first dimension is the head, the second the time and the third the features
        # The nth num_heads / num_devices heads belong to the nth device
        # As we combine the heads in the last dimension, for the input, the first num_heads / num_devices * num_features
        # belong to the first device, and so on. This is important as we apply the same pruning mask as for dense layers.
        # When reshaping, we need to take these chunkks (i.e., result[:, 0: num_heads / num_devices * num_features]) and put them into
        # the coresponding heads (i.e. to result[0:num_heads / num_devices, 0: num_features]).
        # This is exactly what the einshape call does.
        # result = es.jax_einshape("n(hf)->hnf", result, h=self.num_heads)

        # print(qs.shape)

        qs = es.jax_einshape("n(hf)->hnf", qs, h=self.num_heads)
        ks = es.jax_einshape("n(hf)->hnf", ks, h=self.num_heads)
        vs = es.jax_einshape("n(hf)->hnf", vs, h=self.num_heads)

        # inputs_after_layernorm = jax.vmap(self.partial_layer_norm.partial_layer_norm, (0, None))(inputs, mask)
        # recorded_activations["layer_norm"] = inputs_after_layernorm   

        
        # mask = calculate_mask((self.dense_qs.weights.shape[0], self.dense_qs.weights.shape[1]),
        #                 self.pruning_attention,
        #                 self.partial_layer_dropout_attention,
        #                 attention_d_key,
        #                 self.o_split_input)
        # qs = self.calculate_attention_input(inputs_after_layernorm, weight_scaling=weight_scalings["q"], activation_scaling=activation_scalings["layer_norm"], dense=self.dense_qs, mask=mask)
        # ks = self.calculate_attention_input(inputs_after_layernorm, weight_scaling=weight_scalings["k"], activation_scaling=activation_scalings["layer_norm"], dense=self.dense_ks, mask=mask)
        # vs = self.calculate_attention_input(inputs_after_layernorm, weight_scaling=weight_scalings["v"], activation_scaling=activation_scalings["layer_norm"], dense=self.dense_vs, mask=mask)
        
        recorded_activations["layer_norm"] = inputs_after_layernorm   
        recorded_activations["q"] = qs
        recorded_activations["k"] = ks
        recorded_activations["v"] = vs

        # if activation_scalings["q"] is not None:
        #     # print(inputs_after_layernorm[0:2, :])
        #     # print(inputs)
        #     # print(activation_scalings["layer_norm"])
        #     print(quantize_forward(inputs_after_layernorm[0:2, :], activation_scalings["layer_norm"]))
        #     print("qs:")
        #     print(quantize_forward(qs[0, 0:2, :], activation_scalings["q"], rounding=False))
        #     print("ks:")
        #     print(quantize_forward(ks[0, 0:2, :], activation_scalings["k"]))
        #     print("vs:")
        #     print(quantize_forward(vs[0, 0:2, :], activation_scalings["v"]))

        keys = jax.random.split(key, self.num_heads)
        
        # if causal attention make mask lower triangular to prevent looking into the future
        if use_causal_attention:
            mask = jnp.tril(jnp.ones((len(inputs), len(inputs))))
        else:
            mask = jnp.ones((len(inputs), len(inputs)))

        outputs = jax.vmap(self.attention, in_axes=(0, 0, 0, None, 0, None, None, 0, None, None))(
            qs, 
            ks,
            vs,
            self.dropout,
            keys,
            inference, 
            mask,
            self.encoding_hyperparams,
            self.attention_scores,
            activation_scalings)

        recorded_activations["qkv"] = outputs
        # reshape outputs (concatenate heads)
        # The first num_heads / num_devices * num_features belong to the first device, and so on.
        outputs = es.jax_einshape("hnf->n(hf)", outputs)

        key, key2 = jax.random.split(key)
        output_key = self.partial_layer_dropout_o.sample(key2)
        mask = calculate_mask(self.dense_o.weights.shape,
                              self.pruning_o,
                              self.partial_layer_dropout_o,
                              output_key)
        recorded_activations["heads"] = outputs
        # if activation_scalings["o"] is not None:
        #     print("heads:")
        #     print(quantize_forward(outputs[0:2, :], activation_scalings["heads"]))
        outputs = self.dense_o(outputs, weight_scalings=weight_scalings["o"], activation_scalings=activation_scalings["heads"], mask=mask)
        
        # residual connection
        # here again, we dont need pruning/dropout (cf. DenseLayer)
        recorded_activations["o"] = outputs
        outputs = quantize_forward_backward(outputs, activation_scalings["o"])
    

        key, key2 = jax.random.split(key)
        outputs = self.dropout(outputs, inference=inference, key=key2)

        inputs = quantize_forward_backward(inputs, activation_scalings["in"])
        # if activation_scalings["o"] is not None:
        #     print(outputs[0:2, :])
        #     print(quantize_forward(outputs[0:2, :], activation_scalings["o"]))
        # note that inputs_after_layernorm has a different shape caused by inter-device-pruning.
        # now only get the entrances in it that belong to the device and put in one vector (cf layernorm)
        # inputs_after_layernorm = jax.vmap(self.partial_layer_norm.reshape_back)(inputs_after_layernorm)
        outputs += inputs # inputs_after_layernorm

        recorded_activations["sum"] = outputs

        # outputs = quantize_forward_backward(outputs, activation_scalings["sum"])

        return outputs, recorded_activations

    def get_weight_dict(self) -> dict:
        """Returns the weights of the multi-head attention block as a dictionary."""
        weight_dict = {}
        weight_dict["q"] = self.dense_qs.get_weight_dict()
        weight_dict["k"] = self.dense_ks.get_weight_dict()
        weight_dict["v"] = self.dense_vs.get_weight_dict()
        weight_dict["o"] = self.dense_o.get_weight_dict()
        weight_dict["partial_layer_norm"] = self.partial_layer_norm.get_weight_dict()
        return weight_dict
    
    def restructure(self, input_permutation):
        new_partial_layer_norm = self.partial_layer_norm.restructure(input_permutation)

        new_dense_qs = self.dense_qs.reorder_inputs(input_permutation)
        new_dense_ks = self.dense_ks.reorder_inputs(input_permutation)
        new_dense_vs = self.dense_vs.reorder_inputs(input_permutation)
        
        new_dense_o = self.dense_o.reorder_outputs(input_permutation)   # because of residual connection
        perm = input_permutation

        return eqx.tree_at(lambda pt: (pt.dense_qs, pt.dense_ks, pt.dense_vs, pt.dense_o, pt.partial_layer_norm), 
                           self, 
                           (new_dense_qs, new_dense_ks, new_dense_vs, new_dense_o, new_partial_layer_norm)), perm
    
    def get_pruning_weights(self):
        return jnp.array([self.dense_qs.weights, self.dense_ks.weights, self.dense_vs.weights]), jnp.array([self.dense_o.weights])

    def where_pruning_masks(self, pt):
        return pt.pruning_attention, pt.pruning_o  

    def where_dropout_masks(self, pt):
        return pt.partial_layer_dropout_attention, pt.partial_layer_dropout_o
        
    def get_number_of_parameters(self) -> int:
        """Returns the number of parameters in the multi-head attention block."""
        num_params = 0
        num_params += self.dense_qs.get_number_of_parameters()
        num_params += self.dense_ks.get_number_of_parameters()
        num_params += self.dense_vs.get_number_of_parameters()
        num_params += self.dense_o.get_number_of_parameters()
        return num_params


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkeys = ju.get_subkeys(key, 5)
    a = jax.vmap(DenseLayer, (None, None, 0))(10, 20, subkeys)

    x = jax.random.normal(key, (1, 5))
    print(jax.vmap(a, (0, None))(x, None))