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
import mpx



class VIT(eqx.Module, ContainsPruning, ContainsDropout, SavableModel):
    """Vision Transformer model from the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
    """
    decoder: Decoder
    input_patch_length: Int  # in pixels
    # cls_token: Array
    num_channels: Int
    input_dim: Int


    def __init__(self, 
                 cfg: omegaconf.DictConfig,
                 key: PRNGKeyArray,
                 ):
        
        key, subkey = jax.random.split(key)
        self.input_patch_length = cfg.model.input_patch_length
        self.num_channels = cfg.model.num_channels
        input_dim = cfg.model.num_channels * cfg.model.input_patch_length * cfg.model.input_patch_length
        self.input_dim = input_dim

        # self.cls_token = jax.random.normal(key, (1, input_dim))
                
        self.decoder = Decoder(input_dim=input_dim,
                               num_features=cfg.model.num_features_attention,
                               num_features_residual=cfg.model.num_features_residual,
                               output_dim=cfg.model.num_classes,
                               num_heads=cfg.model.num_attention_heads,
                               num_transformer_blocks=cfg.model.num_transformer_blocks,
                               num_layers_residual_block=cfg.model.num_layers_residual_block,
                               activation_function_residual_block=cfg.model.activation_function_residual_block,
                               dropout_rate=cfg.model.dropout_rate,
                               partial_layer_dropout_prob=cfg.model.partial_layer_dropout_prob,
                               attention_scores=cfg.model.attention_scores,
                               add_pos_embeddings=cfg.model.add_pos_embeddings,
                               max_context_length=cfg.model.max_context_length,
                               num_devices=cfg.num_devices,
                               num_features_head=cfg.model.num_features_head,
                               num_layers_head=cfg.model.num_layers_head,
                               head_activation=cfg.model.activation_function_head,
                               use_input_output_residual=False,
                               use_cls_token=True,
                               disable_pruning_partial_layer_dropout=cfg.model.disable_pruning_partial_layer_dropout,
                               prune_residual_blocks_completely=cfg.model.prune_residual_blocks_completely,
                               key=subkey)


    def patchify_input(self, x: Array) -> Array:
        return es.jax_einshape("(np)(mq)c->(nm)(pqc)", x, p=self.input_patch_length, q=self.input_patch_length, c=self.num_channels)
    
    def __call__(self, inputs: Array, weight_scalings: dict | None, activation_scalings: dict | None, inference: bool, key: PRNGKeyArray, exit_exec=False) -> Array:

        # patch input
        inputs = self.patchify_input(inputs)
        # inputs = jnp.zeros(inputs.shape)
        # inputs = jnp.concatenate((inputs, self.cls_token), axis=0)
        outputs, recorded_activations = self.decoder(inputs, 
                                                     weight_scalings=weight_scalings, 
                                                     activation_scalings=activation_scalings, 
                                                     inference=inference, 
                                                     key=key,
                                                     use_causal_attention=False,
                                                     exit_exec=exit_exec)

        # outputs = mpx.force_full_precision(jax.nn.softmax, outputs.dtype)(outputs[-1, :].flatten())
        outputs = outputs[-1, :]
        return outputs, recorded_activations
    
    def get_weight_dict(self) -> dict:
        """Returns the weights of the multi-head attention block as a dictionary."""
        return self.decoder.get_weight_dict()


    def loss(self, pred, target):
        """Loss function for the vit.
        """
        target_one_hot = jax.nn.one_hot(target, num_classes=pred.shape[-1], dtype=pred.dtype)
        loss = -mpx.force_full_precision(jnp.sum, pred.dtype)(target_one_hot * jnp.log(pred + 1e-4))
        return loss
    
    def acc(self, pred, target):
        """Accuracy function for the vit.
        """
        pred = jnp.argmax(pred, axis=-1)
        acc = mpx.force_full_precision(jnp.mean, pred.dtype)(pred == target)
        return acc
    
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
