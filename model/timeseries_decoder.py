"""
Decoder only model like in TimesFM
"""

import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
import omegaconf

import einshape as es

from model.decoder import Decoder

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

from model.dropout import ContainsDropout
from model.pruning import ContainsPruning
from model.transformer import SavableModel

def _calculate_mean_std(inputs):
    """Calculate mean and std for the inputs."""
    mean = jnp.mean(inputs, keepdims=True, axis=0)
    std = jnp.std(inputs, keepdims=True, axis=0) + 1e-5
    return mean, std
            
def _normalize(inputs, mean, std):
    """Normalize the inputs."""
    return (inputs - mean) / std

def _denormalize(inputs, mean, std):
    """Denormalize the inputs."""
    return inputs * std + mean


class TimeseriesPatchedDecoder(eqx.Module, ContainsPruning, ContainsDropout, SavableModel):
    decoder: Decoder
    input_patch_length: Int
    output_patch_length: Int
    input_dim: Int
    apply_padding: bool
    use_reversible_input_normalization: bool


    def __init__(self, 
                 input_dim: Int,
                 cfg: omegaconf.DictConfig,
                 key: PRNGKeyArray,
                 apply_padding: bool = True):
        
        key, subkey = jax.random.split(key)
        self.input_patch_length = cfg.model.input_patch_length
        self.output_patch_length = cfg.dataset.prediction_length
        self.input_dim = input_dim
        self.apply_padding = apply_padding
        patched_input_dim = cfg.model.input_patch_length * input_dim 
        if self.apply_padding:  # input + padding_mask
            patched_input_dim += cfg.model.input_patch_length
        
        self.decoder = Decoder(input_dim=patched_input_dim,
                               num_features=cfg.model.num_features_attention,
                                 num_features_residual=cfg.model.num_features_residual,
                               output_dim=self.output_patch_length*input_dim,
                               num_heads=cfg.model.num_attention_heads,
                               num_transformer_blocks=cfg.model.num_transformer_blocks,
                               num_layers_residual_block=cfg.model.num_layers_residual_block,
                               activation_function_residual_block=cfg.model.activation_function_residual_block,
                               dropout_rate=cfg.model.dropout_rate,
                               partial_layer_dropout_prob=cfg.model.partial_layer_dropout_prob,
                               attention_scores=cfg.model.attention_scores,
                               add_pos_embeddings=cfg.model.add_pos_embeddings,
                               max_context_length=cfg.model.max_context_length,
                                 num_features_head=cfg.model.num_features_head,
                                 num_layers_head=cfg.model.num_layers_head,
                                 use_cls_token=False,
                                 use_input_output_residual=True,
                                 head_activation=cfg.model.activation_function_head,
                               num_devices=cfg.num_devices,
                               disable_pruning_partial_layer_dropout=cfg.model.disable_pruning_partial_layer_dropout,
                               prune_residual_blocks_completely=cfg.model.prune_residual_blocks_completely,
                               key=subkey)
        self.use_reversible_input_normalization = cfg.model.use_reversible_input_normalization

    def __call__(self, inputs: Array, padding_mask: Array | None, weight_scalings: dict | None, activation_scalings: dict | None, inference: bool, key: PRNGKeyArray) -> Array:
        if self.use_reversible_input_normalization:
            mean, std = _calculate_mean_std(inputs)
            inputs = _normalize(inputs, mean, std)
        
        # padding mask
        if self.apply_padding:
            if padding_mask is None:
                padding_mask = jnp.zeros(inputs.shape[:-1], dtype=jnp.float16)
            
            inputs *= (1-padding_mask[..., None])

        # patch input
        inputs = es.jax_einshape("(np)d->n(pd)", inputs, d=self.input_dim, p=self.input_patch_length)


        if self.apply_padding:
            padding_mask = es.jax_einshape("(np)->np", padding_mask, p=self.input_patch_length)

            inputs = jnp.concatenate([inputs, padding_mask], axis=-1)
        
        outputs, recorded_activations = self.decoder(inputs, 
                                                     weight_scalings=weight_scalings, 
                                                     activation_scalings=activation_scalings, 
                                                     inference=inference, 
                                                     key=key)

        outputs = es.jax_einshape("n(pd)->npd", outputs, d=self.input_dim, p=self.output_patch_length)

        if self.use_reversible_input_normalization:
            outputs = _denormalize(outputs, mean, std)

        if self.use_reversible_input_normalization:
            return (outputs, mean, std), recorded_activations
        else:
            return outputs, recorded_activations
    
    def get_weight_dict(self) -> dict:
        """Returns the weights of the multi-head attention block as a dictionary."""
        return self.decoder.get_weight_dict()
    

    def restructure(self):
        restructured_decoder, _ = self.decoder.restructure(None)
        return eqx.tree_at(lambda x: x.decoder, self, restructured_decoder)

    def loss(self, pred, target):
        """Loss function for the PatchedDecoder model.
        
        Args:
            pred: The predicted output from the model. It is of shape (num_tokens, output_patch_length).
            target: The target output it is of shape (num_tokens*input_patch_length + output_patch_length).
        """

        if self.use_reversible_input_normalization:
            mean, std = pred[1], pred[2]
            pred = pred[0]
            # target = _normalize(target, mean, std)

        # loss for a single token
        def _single_loss(index):
            s = jnp.mean((pred[index, ...] 
                             - target[(index+1)*self.input_patch_length:(index+1)*self.input_patch_length + len(pred[index, :]), ...]
                             ) ** 2)
            return s
        loss = 0
        # jax.debug.print("target= {t}", t=target[0:96,...])
        # jax.debug.print("pred= {p}", p=pred[0][0:10, 0])
        for i in range(len(pred)):
            loss += _single_loss(i) / len(pred)
            # print(len(pred))
            # if i == 0:
            #     jax.debug.print("loss= {l}", l=loss)
        return loss
    
    def acc(self, pred, target):
        return 0
    
    def get_pruning_weights(self):
        return ()
    
    def where_pruning_masks(self, pt):
        return ()
    
    def where_dropout_masks(self, pt):
        return ()

    def get_number_of_parameters(self) -> int:
        """Get the number of parameters in the model.
        
        Returns:
            The number of parameters in the model.
        """
        return self.decoder.get_number_of_parameters()
