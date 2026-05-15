import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

@jax.custom_jvp
def quantization_round(x):
    return jnp.round(x)

@quantization_round.defjvp
def quantization_round_jvb(primals, tangents):
    return quantization_round(primals[0]), tangents[0]

def quantization_range(num_bits):
    return 2**(num_bits-1) - 1


def calculate_scalings(records: dict, num_bits=8):
    """
    Calculate the scaling factors for quantization.
    """
    bit_range = quantization_range(num_bits)
    def _calc_scaling(x):
        return bit_range / jnp.max(jnp.abs(x))
    
    return jax.tree_util.tree_map(_calc_scaling, records)

def calculate_cmsis_nn_quant_params(scale, num_bits=8):
    """
    Calculate the parameters for CMSIS-NN quantization.
    """
    bit_range = quantization_range(num_bits)
    shift = 0
    while scale >= 1:
        scale /= 2
        shift += 1
    
    # float to q31
    multiplier = int(scale * (2**31))

    return multiplier, shift

def calculate_scaling_from_cmsis_nn_quant_params(cmsis_nn_quant_params):
    """
    Calculate the scaling factor from CMSIS-NN quantization parameters.
    """
    multiplier, shift = cmsis_nn_quant_params
    scaling = (2**shift / 2**31) * multiplier   # scale from q31 to float
    return scaling

def quantize_forward_cmsis_nn(x, cmsis_nn_quant_params, num_bits=8):
    multiplier, shift = cmsis_nn_quant_params
    scaling = (2**shift / 2**31) * multiplier   # scale from q31 to float
    return quantize_forward(x, scaling, num_bits)

def quantize_backward_cmsis_nn(x, cmsis_nn_quant_params, num_bits=8):
    multiplier, shift = cmsis_nn_quant_params
    scaling = (2**shift / 2**31) * multiplier   # scale from q31 to float
    return quantize_backward(x, scaling, num_bits)


def quantize_forward(x, scaling, num_bits=8, rounding=True):
    bit_range = quantization_range(num_bits)
    if rounding:
        return jnp.clip(quantization_round(x * scaling), a_min=-bit_range - 1, a_max=bit_range)
    else:
        return jnp.clip((x * scaling), a_min=-bit_range - 1, a_max=bit_range)


def quantize_backward(x, scaling, num_bits=8):
    if scaling is None:
        return x
    return x / scaling 


def quantize_forward_backward(x, scaling, num_bits=8):
    if scaling is None:
        return x
    return quantize_backward(quantize_forward(x, scaling, num_bits), scaling, num_bits)
