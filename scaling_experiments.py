import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding # For data and model parallelism (explained in more detail later)
from jax.experimental import mesh_utils

import flax.nnx as nnx
import optax

from dataclasses import dataclass
import grain.python as pygrain
import pandas as pd
import tiktoken
import time

# Create a `Mesh` object representing TPU device arrangement.
# For example, for Kaggle TPU v5e-8:
if jax.device_count() == 8:
    mesh = jax.make_mesh((4, 2), ('batch', 'model'))

    ### Alternatively, we could use the 8-way data parallelism with only one line of code change.
    ### JAX enables quick experimentation with different partitioning strategies
    ### like this. We will come back to this point at the end of this tutorial.
    # mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))

### For free-tier Colab TPU, which only has a single TPU core
if jax.device_count() == 1:
    mesh = jax.make_mesh((1, 1), ("batch", "model"))


# Define a triangular mask for causal attention with `jax.numpy.tril` and `jax.numpy.ones`.
def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

class TransformerBlock(nnx.Module):
    """ A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1):
        # Multi-Head Attention (MHA) with `flax.nnx.MultiHeadAttention`.
        # Specifies tensor sharding (depending on the mesh configuration)
        # where we shard the weights across devices for parallel computation.
        self.mha = nnx.MultiHeadAttention(num_heads=num_heads,
                                          in_features=embed_dim,
                                          kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, 'model')),
                                          bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                          rngs=rngs)
        # The first dropout with `flax.nnx.Dropout`.
        self.dropout1 = nnx.Dropout(rate=rate)
        # First layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ('model',)),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                         rngs=rngs)
        # The first linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear1 = nnx.Linear(in_features=embed_dim,
                                  out_features=ff_dim,
                                  kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, 'model')),
                                  bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                  rngs=rngs)
        # The second linear transformation for the feed-forward network with `flax.nnx.Linear`.
        self.linear2 = nnx.Linear(in_features=ff_dim,
                                  out_features=embed_dim,
                                  kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, 'model')),
                                  bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                  rngs=rngs)
        # The second dropout with `flax.nnx.Dropout`.
        self.dropout2 = nnx.Dropout(rate=rate)
        # Second layer normalization with `flax.nnx.LayerNorm`.
        self.layer_norm2 = nnx.LayerNorm(epsilon=1e-6,
                                         num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ('model',)),
                                         bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                         rngs=rngs)


    # Apply the Transformer block to the input sequence.
    def __call__(self, inputs, training: bool = False):
        input_shape = inputs.shape
        _, seq_len, _ = input_shape

        # Instantiate the causal attention mask.
        mask = causal_attention_mask(seq_len)

        # Apply Multi-Head Attention with the causal attention mask.
        attention_output = self.mha(
            inputs_q=inputs,
            mask=mask,
            decode=False
        )
        # Apply the first dropout.
        attention_output = self.dropout1(attention_output, deterministic=not training)
        # Apply the first layer normalization.
        out1 = self.layer_norm1(inputs + attention_output)

        # The feed-forward network.
        # Apply the first linear transformation.
        ffn_output = self.linear1(out1)
        # Apply the ReLU activation with `flax.nnx.relu`.
        ffn_output = nnx.relu(ffn_output)
        # Apply the second linear transformation.
        ffn_output = self.linear2(ffn_output)
        # Apply the second dropout.
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        # Apply the second layer normalization and return the output of the Transformer block.
        return self.layer_norm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nnx.Module):
    """ Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Matimum sequence length.
        vocal_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        # Initialize token embeddings (using `flax.nnx.Embed`).
        # Each unique word has an embedding vector.
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def __call__(self, x):
        # Generate a sequence of positions for the input tokens.
        positions = jnp.arange(0, x.shape[1])[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x)
        # Combine token and positional embeddings.
        return token_embedding + position_embedding

class MiniGPT(nnx.Module):
    """ A miniGPT transformer model, inherits from `flax.nnx.Module`.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        feed_forward_dim (int): Dimensionality of the feed-forward network.
        num_transformer_blocks (int): Number of transformer blocks. Each block contains attention and feed-forward networks.
        rngs (nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """
    # Initialize miniGPT model components.
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, num_heads: int, feed_forward_dim: int, num_transformer_blocks: int, rngs: nnx.Rngs):
        # Initiliaze the `TokenAndPositionEmbedding` that combines token and positional embeddings.
        self.embedding_layer = TokenAndPositionEmbedding(
                    maxlen, vocab_size, embed_dim, rngs=rngs
                )
        # Create a list of `TransformerBlock` instances.
        # Each block processes input sequences using attention and feed-forward networks.
        self.transformer_blocks = nnx.List([TransformerBlock(
            embed_dim, num_heads, feed_forward_dim, rngs=rngs
        ) for _ in range(num_transformer_blocks)])
        # Initialize the output `flax.nnx.Linear` layer producing logits over the vocabulary for next-token prediction.
        self.output_layer = nnx.Linear(in_features=embed_dim,
                                       out_features=vocab_size,
                                       kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, 'model')),
                                       bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('model',)),
                                       rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        # Pass the input tokens through the `embedding_layer` to get token embeddings.
        # Apply each transformer block sequentially to the embedded input, use the `training` flag for the behavior of `flax.nnx.Dropout`.
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        # Pass the output of the transformer blocks through the output layer,
        # and obtain logits for each token in the vocabulary (for next token prediction).
        outputs = self.output_layer(x)
        return outputs

    @nnx.jit
    def sample_from(self, logits):
        logits, indices = jax.lax.top_k(logits, k=top_k)
        logits = nnx.softmax(logits)
        return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index):
        logits = self(padded_tokens)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token

    def generate_text(self, max_tokens, start_tokens):
        generated = []
        print(tokenizer.decode(start_tokens), flush=True, end='')
        for i in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1

            padded_tokens = jnp.array((start_tokens + generated + [0] * (maxlen - len(start_tokens) - len(generated))))[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
            if next_token == tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
              break
            generated.append(next_token)
            # decode and print next_token
            print(tokenizer.decode([next_token]), flush=True, end='')
        return tokenizer.decode(start_tokens + generated)

# Creates the miniGPT model with 4 transformer blocks.
def create_model(rngs):
    return MiniGPT(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim, num_transformer_blocks=4, rngs=rngs)

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
num_transformer_blocks = 8
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 144 * jax.device_count() / 2  # divide by 2 in case of model parallelism
if jax.device_count() == 1:
    batch_size = 144
num_epochs = 1
top_k = 10

@dataclass
class TextDataset:
    data: list
    maxlen: int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Use Tiktoken for tokenization
        encoding = tokenizer.encode(self.data[idx], allowed_special={'<|endoftext|>'})[:self.maxlen]  # Tokenize and truncate
        return encoding + [0] * (self.maxlen - len(encoding))  # Pad to maxlen

def load_and_preprocess_data(file_path, batch_size, maxlen):

    with open(file_path, 'r') as f:
      text = f.read()

    stories = text.split('<|endoftext|>')
    stories = [story+'<|endoftext|>' for story in stories if story.strip()]
    df = pd.DataFrame({'text': stories})
    data = df['text'].dropna().tolist()
    dataset = TextDataset(data, maxlen)

    sampler = pygrain.IndexSampler(
        len(dataset),
        shuffle=False,
        seed=42,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )

    dl = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
    )

    return dl

text_dl = load_and_preprocess_data('/data/datasets/TinyStories-train.txt', batch_size, maxlen)

# Defines the loss function using `optax.softmax_cross_entropy_with_integer_labels`.
def loss_fn(model, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
    return loss, logits

# Define the training step with the `flax.nnx.jit` transformation decorator.
@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(model, grads)

with jax.set_mesh(mesh):

    model = create_model(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    rng = jax.random.PRNGKey(0)

    start_prompt = "Once upon a time"
    start_tokens = tokenizer.encode(start_prompt)[:maxlen]
    print("Initial generated text:")
    generated_text = model.generate_text(maxlen, start_tokens)

    metrics_history = {
        "train_loss": [],
    }

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    step = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in text_dl:
            if len(batch) % len(jax.devices()) != 0:
                continue  # skip the remaining elements
            input_batch = jnp.array(jnp.array(batch).T)
            target_batch = prep_target_batch(input_batch)
            train_step(
                model,
                optimizer,
                metrics,
                jax.device_put(
                    (input_batch, target_batch), NamedSharding(mesh, P("batch", None))
                ),
            )

            if (step + 1) % 200 == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value)
                metrics.reset()

                elapsed_time = time.time() - start_time
                print(
                    f"\n\nStep {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds"
                )
                start_time = time.time()

                print("Generated text:")
                generated_text = model.generate_text(maxlen, start_tokens)

            step += 1

    # Final text generation
    print("Final generated text:")
    generated_text = model.generate_text(maxlen, start_tokens)