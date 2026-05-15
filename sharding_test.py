import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import equinox as eqx
import numpy as np

@eqx.filter_jit
def func(x, replicated_sharding):
    print("recompile")
    x = eqx.filter_shard(x, replicated_sharding)
    return x

if __name__ == "__main__":
    jax.config.update('jax_num_cpu_devices', 8)
    devices = jax.devices(backend="cpu")
    mesh = jax.make_mesh((len(devices), ), ("batch",),devices=devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, P("batch"))
    
    a = jnp.zeros((800,))
    a = jnp.reshape(a, (8, 100))
    a = jax.device_put(a, batch_sharding)
    a = a.swapaxes(0, 1)
    jax.debug.visualize_array_sharding(a[:, :])
    a = a.swapaxes(0, 1)

    a = jnp.reshape(a, (8, 10, 10))
    a = jnp.reshape(a, (8, 10, 2, 5))
    a = a.swapaxes(0, 1)
    a  = jnp.reshape(a, (10, 80))
    
    jax.debug.visualize_array_sharding(a[:, :])
    
    # replicated_sharding = jax.sharding.NamedSharding(mesh, P())
    # x = jax.numpy.ones((160, 8))
    # x = eqx.filter_shard(x, replicated_sharding)
    # jax.debug.visualize_array_sharding(x)
    
    # x = func(x, replicated_sharding)
    # jax.debug.visualize_array_sharding(x)

    # x = func(x, replicated_sharding)
    # jax.debug.visualize_array_sharding(x)