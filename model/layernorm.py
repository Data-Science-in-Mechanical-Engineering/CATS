import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import PRNGKeyArray, Int, Float, Array
from utils.splitting_utils import get_neuron_slices, get_slice_indices_from_split, get_weight_slices, split_neurons

import einshape as es


class PartialLayerNorm(eqx.Module):
    rescale: Array
    bias: Array 

    def __init__(self, input_dim: Int):
        self.rescale = jnp.ones((input_dim,), dtype=jnp.float32)  # jnp.ones((input_dim,), dtype=jnp.float32)
        self.bias = jnp.zeros((input_dim,), dtype=jnp.float32)  # jnp.zeros((input_dim,), dtype=jnp.float32)

    def pruned_layer_norm(self, x, mask):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        mask = mask.astype(jnp.float32)
        x_masked = x * mask
        summed = jnp.sum(x_masked, axis=-1)
        mean = summed / jnp.sum(mask, axis=-1)
        mean = jnp.expand_dims(mean, axis=-1)
        x_masked -= mean

        x_masked *= mask
        std = jnp.sqrt(jnp.sum(x_masked ** 2, axis=-1) / (jnp.sum(mask, axis=-1) - 1))
        x_masked /= std + 1e-7

        return ((x_masked * self.rescale + self.bias) * mask).astype(orig_dtype)   # also mask
    
    def partial_layer_norm(self, x: Array, mask: Array) -> Array:
        """
        Applies layer normalization to the input tensor `x` using the provided `mask`.
        As every device only receives a part of x, we need to apply layer normalization for each device separately.


        Args:
            x (Array): The input tensor to be normalized.
            mask (Array): The mask tensor indicating which elements to include in the normalization. Has to have shape (len(x), ...).

        Returns:
            Array: The layer-normalized tensor. This now has shape (mask.shape(0), len(x)). Here layer normalization is applied to each entrance seperately.
        """
        assert len(x.shape) == 1, "Input tensor must be 2D"
        assert mask.shape[0] == x.shape[0], "Mask must have the same first dimension as input tensor"

        x_tiled = jnp.tile(x, (mask.shape[1], 1))

        mask = mask.T

        # we only calculate the mean and std for the masked data. Hence, we multiply the data with the mask. 
        # and calculate the "number of elements" we calculate the mean and std for by summing the mask.
        x_masked = x_tiled * mask

        def norm(x_masked_, mask_):
            summed = jnp.sum(x_masked_)
            mean = summed / (jnp.sum(mask_) + 1e-7)
            x_masked_ -= mean
            # print("mmmmmmmmmmmmmm")
            # print(mean)

            x_masked_ *= mask_
            std = jnp.sqrt(jnp.sum(x_masked_ ** 2) / (jnp.sum(mask_) - 1) + 1e-7)
            # print(std)
            x_masked_ /= (std + 1e-7)

            x_masked_ = (x_masked_ + self.bias) * self.rescale
            return x_masked_

        x_masked = jax.vmap(norm, in_axes=(0, 0))(x_masked, mask)

        return x_masked

    def reshape_back(self, x,):
        num_devices = x.shape[0]
        slice_1 = get_neuron_slices(x.shape[1], num_devices)
        indexes_0 = []
        indexes_1 = []
        for i in range(num_devices):
            for j in range(slice_1[i], slice_1[i + 1]):
                indexes_0.append(i)
                indexes_1.append(j)
        return x[indexes_0, indexes_1]
    

    def normal_layer_norm(self, x: Array) -> Array:
        mean = jnp.mean(x)
        std = jnp.std(x)
        x_normalized = (x - mean) / (std + 1e-7)
        x_normalized = x_normalized * self.rescale + self.bias
        return x_normalized


    def partial_layer_normed_data_through_dense(self, x, weight_scaling, activation_scaling, dense, split_output=None):
        """
        Wrapper to put the output of the layer norm through a dense layer. This is needed, because the layer norm is applied to the data of each device separately.
        x[i, :] is the data device i has (entrances it does not have are 0)) normalized. 
        Normally, every device would now put this data through its part of the dense layer. However, this is difficult to implement here in an efficient way. Therefore, we do a little trick.

        We put this data through the entire dense layer for every device (in reality every device would do this only for parts of the layer).
        Then, for every device, we only take the data the device would have in reality and return this as one vector.
        (this is then the combined data the devices have after calculating the dense layer.).
        """
        
        # result = jax.vmap(dense, (0, None, None, None))(x, weight_scaling, activation_scaling, None)
        num_devices = x.shape[0]

        if split_output is None:
            output_slices = get_neuron_slices(dense.weights.shape[1], num_devices)
        else:
            output_slices = get_slice_indices_from_split(split_output)


        partial_results = []
        for i in range(num_devices):
            partial_results.append(dense.get_output_for_device(x[i, :],  weight_scaling, activation_scaling, mask=None, device_index=i, output_slices=output_slices))
        return jnp.concatenate(partial_results, axis=0)
        

        # we do not care about all results. Currently we caculate what would happen, if each device would put the data it has
        # through the entire layer (x[i, :] is the data device i has (entrances it does not have are 0)). 
        # However, in reality, they only put it through parts of the layer.
        # What we need is the following (1 index needed, 0 not):
        # Assume num_devices = 3, feature_dim = 8
        # results_needded = [1, 1, 1, 0, 0, 0, 0, 0,
        #                    0, 0, 0, 1, 1, 1, 0, 0
        #                    0, 0, 0, 0, 0, 0, 1, 1]    
        # at the end of the nested loop: indexes_0 = [0, 0, 0, 1, 1, 1, 2, 2], indexes_1 = [0, 1, 2, 3, 4, 5, 6, 7] 
        # if split_output is None:     
        #     slice_1 = get_neuron_slices(result.shape[1], num_devices)
        # else:
        #     slice_1 = get_slice_indices_from_split(split_output)
        # indexes_0 = []
        # indexes_1 = []
        # for i in range(num_devices):
        #     for j in range(slice_1[i], slice_1[i + 1]):
        #         indexes_0.append(i)
        #         indexes_1.append(j)

        # return result[indexes_0, indexes_1]
    
    def get_weight_dict(self) -> dict:
        """Returns the weights of the partial layer norm as a dictionary."""
        return {
            "rescale": self.rescale,
            "bias": self.bias,
        }

    def restructure(self, input_permutation):
        return eqx.tree_at(lambda pt: (pt.rescale, pt.bias), self, (self.rescale[input_permutation], self.bias[input_permutation]))


if __name__ == "__main__":
    # Example usage
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([[1, 0], 
                      [0, 1], 
                      [1, 1], 
                      [0, 1]])
    result = partial_layer_norm(x, mask)
    print(result)