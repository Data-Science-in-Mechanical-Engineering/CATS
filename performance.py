from functools import partial
import time
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import tqdm

import equinox as eqx


@register_pytree_node_class
class RegisteredSpecial2():
    
    def __init__(self, layers, activation):
        self.layers = layers
        self.activation = activation

    def __repr__(self):
        return "RegisteredSpecial2"

    def tree_flatten(self):
        children = (self.layers,)
        aux_data = (self.activation, )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data[0])
    
class Layer(eqx.Module):
    weight: jax.Array

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return self.weight @ x
    
class Model(eqx.Module):
    layers: list
    activation: callable
    mask = list

    def __init__(self, layers, activation):
        self.layers = layers
        self.activation = activation

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
            x = self.activation(x)
        return x


# @partial(jax.jit, static_argnames=("model_aux", "flag1"))
@eqx.filter_jit
def do_smth_static(model, x, flag1):
    # model = RegisteredSpecial2.tree_unflatten(model_aux, model_child)
    def call_model(model, x):
        for i in range(20):
            # x = model.layers[i]@x
            x = model(x)
            if flag1:
                x = model.activation(x)
        return jnp.mean(x)
    
    def call_model_batch(model, x):
        x = jax.vmap(call_model, in_axes=(None, 0))(model, x)
        return jnp.mean(x)
    v, grad = eqx.filter_value_and_grad(call_model_batch, has_aux=False)(model, x)
    model = eqx.apply_updates(model, grad)
    # model_child, model_aux = model.tree_flatten()
    return model, v

@jax.jit
def do_smth(model, x):
    flag1 = True
    activation1 = jax.nn.relu
    for i in range(10):
        x = model[i]@x
        if flag1:
            x = activation1(x)
    return model, x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    layers = {}

    for i in range(20):
        key, subkey = jax.random.split(key)
        layers[i] = jax.random.normal(subkey, (100, 100))

    model = RegisteredSpecial2(layers, jax.nn.relu)

    model = Model([Layer(layers[i]) for i in layers], jax.nn.relu)

    max_time = 0
    step_times_static = []
    activation1 = jax.nn.relu
    for i in tqdm.tqdm(range(1000)):
        start_time = time.time()
        x = jnp.zeros((100, 100, 100))
        # model_child, model_aux = model.tree_flatten()
        model, v = do_smth_static(model, x, True)
        # model = RegisteredSpecial2.tree_unflatten(model_aux, model_child)
        step_time = time.time() - start_time
        step_times_static.append(step_time)

    step_times = []
    # for i in tqdm.tqdm(range(50000)):
    #     start_time = time.time()
    #     x = jnp.zeros((1000,1))
    #     model, x = do_smth(model, x)
    #     step_time = time.time() - start_time
    #     step_times.append(step_time)

    print(model)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(step_times_static[1:], label="do_smth_static")
    plt.plot(step_times[1:], label="do_smth")
    plt.xlabel("Step")
    plt.ylabel("Time (seconds)")
    plt.title("Step Time Performance")
    plt.legend()
    plt.grid(True)
    plt.show()
