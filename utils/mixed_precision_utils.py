"""
Tools for mixer precision training. taken and proted from jmp. The reason for this is that current version of jmp is not compatible with equinox
"""

import functools
import jax
import jax.numpy as jnp
import equinox as eqx

def cast_tree(tree, dtype):
    """Cast a pytree to a given dtype."""
    def _cast(x):
        if eqx.is_array(x):
            return x.astype(dtype)
        else:
            return x
    return jax.tree_util.tree_map(_cast, tree)


def cast_to_float32(x):
    """Cast to float32."""
    return cast_tree(x, jnp.float32)

def cast_to_float16(x):
    """Cast to float16."""
    return cast_tree(x, jnp.float16)

def all_finite(tree) -> jnp.ndarray:
    """Returns a scalar ndarray indicating whether the input arrays are finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = map(jnp.isfinite, leaves)
        leaves = map(jnp.all, leaves)
        return jnp.stack(list(leaves)).all()
    
def force_full_precision(func, return_dtype):
    def wrapper(*args, **kwargs):
        args_full_precision = []
        for arg in args:
            if eqx.is_array(arg):
                args_full_precision.append(arg.astype(jnp.float32))
            else:
                args_full_precision.append(arg)
        args_full_precision = tuple(args_full_precision)

        kwargs_full_precision = {}
        for key, value in kwargs.items():
            if eqx.is_array(value):
                kwargs_full_precision[key] = value.astype(jnp.float32)
            else:
                kwargs_full_precision[key] = value

        results = func(*args_full_precision, **kwargs_full_precision)

        if type(results) == tuple:
            results_converted = []
            for r in results:
                if eqx.is_array(r):
                    results_converted.append(r.astype(return_dtype))
                else:
                    results_converted.append(r)
            return tuple(results_converted)
        elif eqx.is_array(results):
            return results.astype(return_dtype)
        return results
        
    return wrapper

    
def select_tree(pred: jnp.ndarray, a, b):
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    def _select_leaf(x1, x2):
        if eqx.is_array(x1):
            return jax.lax.select(pred, x1, x2)
        else:
            return x1

    return jax.tree_util.tree_map(_select_leaf, a, b)

class DynamicLossScale(eqx.Module):
    loss_scale: jnp.ndarray
    min_loss_scale: jnp.ndarray
    counter: jnp.ndarray
    factor: int
    period: int

    def __init__(self, loss_scale: jnp.ndarray, min_loss_scale: jnp.ndarray, factor: int = 2, period: int = 2000, counter=None):
        self.loss_scale = loss_scale
        self.min_loss_scale = min_loss_scale
        self.factor = factor
        self.period = period
        if counter is None:
            self.counter = jnp.zeros((1,), dtype=jnp.int32)
        else:
            self.counter = counter

    def scale(self, tree):
        return jax.tree_util.tree_map(lambda x: x * self.loss_scale[0], tree)

    def unscale(self, tree):
        inv_loss_scale = 1 / self.loss_scale
        inv_loss_scale = inv_loss_scale.astype(jnp.float32)   # cast to float32, so the result is float32 (otherwise the whole scaling point would be senseless)
        return jax.tree_util.tree_map(lambda x: x * inv_loss_scale[0], tree)
    
    def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
        """Returns the next state dependent on whether grads are finite."""
        assert grads_finite.ndim == 0, "Expected boolean scalar"

        first_finite = lambda a, b: jax.lax.select(jnp.isfinite(a).all(), a, b)

        loss_scale = jax.lax.select(
            grads_finite,

            # When grads are finite increase loss scale periodically.
            jax.lax.select(
                self.counter == (self.period - 1),
                first_finite(self.loss_scale * self.factor,
                            self.loss_scale),
                self.loss_scale),

            # If grads are non finite reduce loss scale.
            jnp.maximum(self.min_loss_scale, self.loss_scale / self.factor))

        loss_scale = jnp.clip(loss_scale, a_min=self.min_loss_scale, a_max=jnp.ones((1,), dtype=jnp.float32) * int((2 - 2**(-10)) * 2**15))

        counter = ((self.counter + 1) % self.period) * grads_finite

        return DynamicLossScale(
            loss_scale=loss_scale,
            counter=counter,
            period=self.period,
            factor=self.factor,
            min_loss_scale=self.min_loss_scale)

