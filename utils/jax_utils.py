import jax

def get_subkeys(key, num_subkeys):
    key, subkey = jax.random.split(key)
    return key, jax.random.split(key, num_subkeys)