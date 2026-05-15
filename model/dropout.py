from abc import ABC, abstractmethod
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from utils.splitting_utils import get_slice_indices_from_split, get_weight_slices, split_neurons

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray


class ContainsDropout(ABC):
    @abstractmethod
    def where_dropout_masks(self, pt):
        """
        Returns the location of the pruning masks in the model pt. Has to have the same order as the associated weights in get_pruning_weights.
        """
        pass

    def change_own_masks(self, partial_layer_dropout_prob_mode1, partial_layer_dropout_prob_mode2, partial_layer_dropout_prob_mode3, unstructured_dropout_prob):
        dropout_masks = self.where_dropout_masks(self)
        new_masks = []
        for m in dropout_masks:
            n = PartialLayerDropoutMask(m.num_devices, 
                                        put_on_last_node=m.put_on_last_node,
                                        dropout_prob_mode1=partial_layer_dropout_prob_mode1, 
                                        dropout_prob_mode2=partial_layer_dropout_prob_mode2, 
                                        dropout_prob_mode3=partial_layer_dropout_prob_mode3,
                                        split_input=m.split_input)
            new_masks.append(n)

        return eqx.tree_at(where=self.where_dropout_masks, pytree=self, replace=new_masks)
    
    def change_dropout_probability(self, partial_layer_dropout_prob_mode1, partial_layer_dropout_prob_mode2, partial_layer_dropout_prob_mode3, unstructured_dropout_prob): 
        # first change probability of own mask
        model = self.change_own_masks(partial_layer_dropout_prob_mode1, partial_layer_dropout_prob_mode2, partial_layer_dropout_prob_mode3, unstructured_dropout_prob)

        # if a leaf or list is of type ContainsDropout, also change it
        for attr_name in dir(model):
            sub = getattr(model, attr_name)
            if isinstance(sub, ContainsDropout):
                sub = sub.change_dropout_probability(partial_layer_dropout_prob_mode1, partial_layer_dropout_prob_mode2, partial_layer_dropout_prob_mode3, unstructured_dropout_prob)
                model = eqx.tree_at(lambda x: getattr(x, attr_name), model, sub)
            elif isinstance(sub, list):
                for i in range(len(sub)):
                    if isinstance(sub[i], ContainsDropout):
                        sub[i] = sub[i].change_dropout_probability(partial_layer_dropout_prob_mode1, partial_layer_dropout_prob_mode2, partial_layer_dropout_prob_mode3, unstructured_dropout_prob)
                        model = eqx.tree_at(lambda x: getattr(x, attr_name)[i], model, sub[i])
        return model


def sample_key_mode1(dropout_prob, num_devices, rng_key: PRNGKeyArray):
    """
        The mask key is a num_devices x num_devices matrix with 1s and 0s, where 1 means that the device at the first dimension communicates with the other device at the second devices.
    """
    return jnp.astype(jax.random.bernoulli(rng_key, jnp.ones((num_devices, num_devices)) * (1 - dropout_prob)), jnp.float32)


def sample_key_mode2(dropout_prob, num_devices, rng_key: PRNGKeyArray):
    key = jnp.astype(jax.random.bernoulli(rng_key, jnp.ones((num_devices,)) * (1 - dropout_prob)), jnp.float32)
    return jnp.tile(key, reps=(num_devices, 1))


def sample_key_mode3(dropout_prob, num_devices, rng_key: PRNGKeyArray):
    key = jnp.astype(jax.random.bernoulli(rng_key, jnp.ones((num_devices,)) * (1 - dropout_prob)), jnp.float32)
    return jnp.tile(key, reps=(num_devices, 1)).T

    

class PartialLayerDropoutMask(eqx.Module):
    """Abstract class for partial layer dropout. It features an abstract method sample, which can be called to sample
    a new mask."""
    put_on_last_node: bool
    num_devices: int
    dropout_prob_mode1: float
    dropout_prob_mode2: float
    dropout_prob_mode3: float

    split_input: list | None


    def __init__(self, num_devices: int, put_on_last_node: bool=False, dropout_prob_mode1=0.0, dropout_prob_mode2=0.0, dropout_prob_mode3=0.0, split_input=None):
        super().__init__()
        assert 0.0 <= dropout_prob_mode1 <= 1.0, f"Invalid dropout probability: {dropout_prob_mode1}"
        assert 0.0 <= dropout_prob_mode2 <= 1.0, f"Invalid dropout probability: {dropout_prob_mode2}"
        assert 0.0 <= dropout_prob_mode3 <= 1.0, f"Invalid dropout probability: {dropout_prob_mode3}"
        
        self.put_on_last_node = put_on_last_node
        self.num_devices = num_devices
        self.dropout_prob_mode1 = dropout_prob_mode1
        self.dropout_prob_mode2 = dropout_prob_mode2
        self.dropout_prob_mode3 = dropout_prob_mode3

        self.split_input = split_input

    @eqx.filter_jit
    def sample(self, rng_key: PRNGKeyArray):
        """Samples a new mask key. The mask key is a num_devices x num_devices matrix with 1s and 0s, where 1 means that the device at the first dimension communicates with the other device at the second devices."""
        mode1_key, mode2_key, mode3_key = jax.random.split(rng_key, num=3)
        key = sample_key_mode1(self.dropout_prob_mode1, self.num_devices, mode1_key)
        key *= sample_key_mode2(self.dropout_prob_mode2, self.num_devices, mode2_key)
        key *= sample_key_mode3(self.dropout_prob_mode3, self.num_devices, mode3_key)
        return key


    @eqx.filter_jit
    def _calculate_mask(self, shape, mask_key, split_output, num_mask_repeat=1):
        """Calculates the mask for the given shape and mask key. 
        The mask key is a num_devices x num_devices matrix with 1s and 0s, where 1 means that the device at the first dimension communicates with the other device at the second devices.
        The mask is a binary matrix with the same shape as the input"""
        assert shape[1] % num_mask_repeat == 0
        
        mask_shape = (shape[0], shape[1] // num_mask_repeat)
        slice_input, slice_output = get_weight_slices(self.num_devices, mask_shape, self.put_on_last_node)
        
        if self.split_input is not None:
            slice_input = get_slice_indices_from_split(self.split_input)
            split_input = self.split_input
        else:
            split_input = split_neurons(self.num_devices, mask_shape[0])
        
        if split_output is not None:
            slice_output = get_slice_indices_from_split(split_output)
        else:
            split_output = split_neurons(self.num_devices, mask_shape[1], put_on_last_node=self.put_on_last_node)
        
        # mask = jnp.ones(mask_shape)
        num_key_repeat_i = slice_input[1]
        num_key_repeat_o = slice_output[1]

        mask = jnp.fill_diagonal(mask_key, 1.0, inplace=False)
        mask = jnp.tile(mask, reps=(num_key_repeat_i, num_key_repeat_o))

        def _calculate_indexes(slice):
            indexes = []
            for i in range(self.num_devices):
                for j in range(slice[i]):
                    indexes.append(j * self.num_devices + i)
            return indexes

        indexes_input = _calculate_indexes(split_input)
        indexes_output = _calculate_indexes(split_output)

        mask = mask[indexes_input, :]
        mask = mask[:, indexes_output]

        return mask    


    def get_mask(self, shape, mask_key, split_output, num_mask_repeat=1):
        # use if statement here instead of _calculate_mask to save on compile time?
        # actually, shouldn't make a difference
        if self.dropout_prob_mode1 == 0.0 and self.dropout_prob_mode2 == 0.0 and self.dropout_prob_mode3 == 0.0:
            # if no dropout, return a mask of ones (to speedup training and compilation)
            return jnp.tile(jnp.ones(shape), reps=(num_mask_repeat, 1))
        else:
            return self._calculate_mask(shape, mask_key, split_output, num_mask_repeat=num_mask_repeat)


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(100)
    rng_key, dropout_key = jax.random.split(rng_key)
    w = jnp.ones((9, 24))
    w2 = jnp.ones((8, 9))
    dm1 = PartialLayerDropoutMask(num_devices=4, dropout_prob_mode1=0.5, dropout_prob_mode2=0.0, dropout_prob_mode3=0.0)

    print("Mode 1 --------------------------")
    dropout_key = jax.random.split(dropout_key, num=10)
    for i in range(10):
        mask_key = dm1.sample(dropout_key[i])

        print(dm1.get_mask(w.shape, mask_key))
        print(dm1.get_mask(w2.shape, mask_key))
        print("")

    dm2 = PartialLayerDropoutMask(num_devices=4, dropout_prob_mode1=0.0, dropout_prob_mode2=0.5, dropout_prob_mode3=0.0)

    print("Mode 2 --------------------------")
    for i in range(10):
        mask_key = dm2.sample(dropout_key[i])
        print(dm2.get_mask(w.shape, mask_key))
        print(dm2.get_mask(w2.shape, mask_key))
        print("")

    dm3 = PartialLayerDropoutMask(num_devices=4, dropout_prob_mode1=0.0, dropout_prob_mode2=0.0, dropout_prob_mode3=0.5)
    # dm31 = PartialLayerDropoutMode3Mask(num_devices=4, weight=w, dropout_prob=0.5, num_mask_repeat=1)

    print("Mode 3 --------------------------")
    for i in range(10):
        mask_key = dm3.sample(dropout_key[i])
        print(dm3.get_mask(w.shape, mask_key))
        print(dm3.get_mask(w2.shape, mask_key))
        print("")
        # print(dm31(dropout_key[i]))
        # print("------")
