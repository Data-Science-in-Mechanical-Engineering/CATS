import jax
import jax.numpy as jnp
import time

import equinox as eqx

import gc

from tqdm import tqdm

class TestLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    activation: callable

    def __init__(self, size):
        self.weight = jax.random.normal(jax.random.PRNGKey(0), (size, size))
        self.bias = jax.random.normal(jax.random.PRNGKey(1), (size,))
        self.activation = None  # jax.nn.relu

    def __call__(self, x):
        if self.activation is None:
            return self.weight @ x + self.bias
        else:
            # Apply activation function if defined
            return self.activation(self.weight @ x + self.bias)
    
class TestModel(eqx.Module):
    layers: list

    def __init__(self, size):
        self.layers = [TestLayer(size) for _ in range(10)]

    @eqx.filter_jit
    def __call__(self, x, ):
        for layer in self.layers:
            x = layer(x)
        return x
    
@eqx.filter_jit
def train(model, x, activation=None):
    x = model(x["x"])

    if activation is not None:
        x = activation(x)

    model = jax.tree_util.tree_map(lambda x: x * 0.99 if eqx.is_array(x) else x, model)
    
    return model, x


    
if __name__ == "__main__":
    size = 1000
    model = TestModel(size)
    x = jax.random.normal(jax.random.PRNGKey(2), (size,))
    batch = {"x": jax.random.normal(jax.random.PRNGKey(2), (size,))}
    train(model, batch)

    times = []
    gc.disable()
    for _ in tqdm(range(10000)):
        batch = {"x": jax.random.normal(jax.random.PRNGKey(2), (size,))}
        start_time = time.time()
        model, x = train(model, batch)
        times.append(time.time() - start_time)

    
    import matplotlib.pyplot as plt

    plt.plot(times)
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.title("Model Inference Times")
    plt.show()

