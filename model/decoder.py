"""
Decoder only model like in TimesFM
"""

import copy
import jax
import jax.numpy as jnp
import equinox as eqx
import omegaconf

import einshape as es
from tqdm import tqdm

from model import quantization
from model.dropout import ContainsDropout
from model.pruning import ContainsPruning
import utils.jax_utils as ju

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

from model.transformer import ResidualBlock, MultiHeadAttentionBlock


class Decoder(eqx.Module, ContainsPruning, ContainsDropout):
    input_residual_block: ResidualBlock
    multi_head_attention_layers: list[MultiHeadAttentionBlock]
    residual_layers: list[eqx.Module]
    output_residual_block: eqx.Module
    pos_embeddings: Array

    cls_token: Array | None

    num_devices: int


    def __init__(self, 
                 input_dim: Int,
                 num_features: Int,
                 num_features_residual: Int,
                 output_dim: Int,
                 num_heads: Int,
                 num_transformer_blocks: Int,
                 num_layers_residual_block: Int,
                 activation_function_residual_block: str,
                 dropout_rate: Float,
                 partial_layer_dropout_prob: Float,
                 attention_scores: str,
                 add_pos_embeddings: bool,
                 max_context_length: Int,
                 num_features_head: Int,
                 num_layers_head: Int,
                 num_devices: Int,
                 use_cls_token: bool,
                head_activation: str,
                use_input_output_residual: bool,
                disable_pruning_partial_layer_dropout: bool,
                prune_residual_blocks_completely: bool,
                 key: PRNGKeyArray):
        
        key, subkey = jax.random.split(key)
        self.num_devices = num_devices

        if use_cls_token:
            self.cls_token = jax.random.normal(key, (1, input_dim))
        else:
            self.cls_token = None

        self.input_residual_block = ResidualBlock(input_dim=input_dim,   # input + padding_mask
                                                 output_dim=num_features, 
                                                 feature_dim=num_features_residual,  #cfg.model.num_features_attention * 2, 
                                                 dropout_rate=dropout_rate,
                                                 partial_layer_dropout_prob_mode1=partial_layer_dropout_prob,
                                                 partial_layer_dropout_prob_mode2=partial_layer_dropout_prob,
                                                 partial_layer_dropout_prob_mode3=partial_layer_dropout_prob, 
                                                 num_layers=num_layers_residual_block,
                                                 transform_residual=True,
                                                 use_layernorm=False,
                                                 disable_pruning_partial_layer_dropout=disable_pruning_partial_layer_dropout,
                                                 key=subkey,
                                                 num_devices=num_devices,
                                                 use_residual=use_input_output_residual,
                                                 activation=activation_function_residual_block
                                                 )
        
        layers = []
        for _ in range(num_transformer_blocks):
            key, subkey = jax.random.split(key)
            layers.append(MultiHeadAttentionBlock(feature_dim=num_features,
                                                  num_heads=num_heads,
                                                  dropout_rate=dropout_rate,
                                                  partial_layer_dropout_prob_mode1=partial_layer_dropout_prob,
                                                  partial_layer_dropout_prob_mode2=partial_layer_dropout_prob,
                                                  partial_layer_dropout_prob_mode3=partial_layer_dropout_prob, 
                                                  disable_pruning_partial_layer_dropout=disable_pruning_partial_layer_dropout,
                                                  key=subkey, 
                                                  attention_scores=attention_scores,
                                                  num_devices=num_devices,))
        self.multi_head_attention_layers = layers

        layers = []
        for _ in range(num_transformer_blocks):
            key, subkey = jax.random.split(key)
            layers.append(ResidualBlock(input_dim=num_features,
                                       output_dim=num_features,
                                       feature_dim=num_features_residual,  # num_features_attention * 2,
                                       dropout_rate=dropout_rate,
                                       partial_layer_dropout_prob_mode1=partial_layer_dropout_prob,
                                       partial_layer_dropout_prob_mode2=partial_layer_dropout_prob,
                                       partial_layer_dropout_prob_mode3=partial_layer_dropout_prob, 
                                       num_layers=num_layers_residual_block,
                                       transform_residual=False,
                                       disable_pruning_partial_layer_dropout=disable_pruning_partial_layer_dropout,
                                       key=subkey,
                                       num_devices=num_devices,
                                       use_residual=True,
                                       prune_completely=prune_residual_blocks_completely,
                                       activation=activation_function_residual_block))    
        self.residual_layers = layers

        key, subkey = jax.random.split(key)
        self.output_residual_block = ResidualBlock(input_dim=num_features,
                                                output_dim=output_dim,
                                                feature_dim=num_features_head,  # num_features_attention * 2,
                                                dropout_rate=dropout_rate * 0.0,
                                                partial_layer_dropout_prob_mode1=partial_layer_dropout_prob,
                                                partial_layer_dropout_prob_mode2=partial_layer_dropout_prob,
                                                partial_layer_dropout_prob_mode3=partial_layer_dropout_prob, 
                                                num_layers=num_layers_head,
                                                transform_residual=True,
                                                disable_pruning_partial_layer_dropout=disable_pruning_partial_layer_dropout,
                                                key=subkey,
                                                num_devices=num_devices,
                                                use_residual=use_input_output_residual,
                                                activation=head_activation)
        
        if add_pos_embeddings:
            self.pos_embeddings = jax.random.normal(key, (max_context_length, num_features))
        else:
            self.pos_embeddings = None
        
    def __call__(self, inputs: Array, weight_scalings: dict | None, activation_scalings: dict | None, inference: bool, key: PRNGKeyArray, use_causal_attention: bool = True, exit_exec=False) -> Array:

        if activation_scalings is None:
            activation_scalings = {f"att_{i}": None for i in range(len(self.multi_head_attention_layers))} \
                                   | {f"res_{i}": None for i in range(len(self.residual_layers))}
            activation_scalings["in"] = None
            activation_scalings["before_pos_embedding"] = None
            activation_scalings["after_pos_embedding"] = None
            activation_scalings["out"] = None
            activation_scalings["pos_enc"] = {"weights": None}
            weight_scalings = copy.deepcopy(activation_scalings)

        recorded_activations = {}

        if self.cls_token is not None:
            inputs = jnp.concatenate((inputs, self.cls_token), axis=0)
        
        # input residual block
        key, subkey = jax.random.split(key)
        # print(inputs.dtype)
        # if activation_scalings["in"] is not None:
        #     print(quantization.quantize_forward(inputs, activation_scalings["in"]["in"], rounding=True))
        # print("residual:")
        # ATTENTION: do not vmap over keys, as the partial layer dropout has to be the same over the time dimension.
        tokens, recorded_activations["in"] = jax.vmap(self.input_residual_block, (0, None, None, None, None))(inputs, 
                                                                                  weight_scalings["in"],
                                                                                  activation_scalings["in"], 
                                                                                  inference, 
                                                                                  subkey)
        # print(tokens.dtype)
        # if activation_scalings["in"] is not None:
        #     print(quantization.quantize_forward(recorded_activations["in"]["in"][:, 0:10], activation_scalings["in"]["0"], rounding=True))
        #     print(quantization.quantize_forward(recorded_activations["in"]["0"][:, 0:10], activation_scalings["in"]["0"], rounding=True))
        #     print("0:")
        #     print(quantization.quantize_forward(recorded_activations["in"]["1"][:, 0:10], activation_scalings["in"]["1"], rounding=False))
        #     print("1:")
        #     print(quantization.quantize_forward(recorded_activations["in"]["2"][:, 0:10], activation_scalings["in"]["2"], rounding=False))
        #     print("out:")
        #     print(quantization.quantize_forward(tokens[:, 0:10], activation_scalings["before_pos_embedding"], rounding=True))
            # if exit_exec:
            #     exit()
        if self.pos_embeddings is not None:
            tokens = quantization.quantize_forward_backward(tokens, activation_scalings["before_pos_embedding"])
            # if activation_scalings["in"] is not None:
            #     print(quantization.quantize_forward(tokens[:, 0:10], activation_scalings["before_pos_embedding"], rounding=True))
            # print("embedd")
            recorded_activations["before_pos_embedding"] = tokens
            pos_emb = quantization.quantize_forward_backward(self.pos_embeddings[:len(tokens), :], weight_scalings["pos_enc"]["weights"])
            # if activation_scalings["in"] is not None:
            #     print(quantization.quantize_forward(pos_emb[:, 0:10], weight_scalings["pos_enc"]["weights"], rounding=True))
            #     print(quantization.quantize_forward(pos_emb[:, 128:128+10], weight_scalings["pos_enc"]["weights"], rounding=True))
            # print("embedd")
            
            tokens += pos_emb
            recorded_activations["after_pos_embedding"] = tokens
            tokens = quantization.quantize_forward_backward(tokens, activation_scalings["after_pos_embedding"])
            # tokens += self.pos_embeddings[:len(tokens), :]
            # if activation_scalings["in"] is not None:
            #     print("after pos embedd")
            #     print(quantization.quantize_forward(tokens[:, 0:10], activation_scalings["after_pos_embedding"], rounding=True))
            #     print(quantization.quantize_forward(tokens[:, 128:128+10], activation_scalings["after_pos_embedding"], rounding=True))
            #     # if exit_exec:
                #     exit()
        
        # if activation_scalings[f"att_{0}"] is not None:
        #     print(quantization.quantize_forward(tokens[0:2, 0:30], activation_scalings[f"att_{0}"]["in"], rounding=True))
        # transformer layers
        for i, (multi_head_attention_layer, residual_layer) in enumerate(zip(self.multi_head_attention_layers, self.residual_layers)):
            key, subkey = jax.random.split(key)
            tokens, recorded_activations[f"att_{i}"] = multi_head_attention_layer(tokens, 
                                                                                weight_scalings=weight_scalings[f"att_{i}"],
                                                                                activation_scalings=activation_scalings[f"att_{i}"],
                                                                                inference=inference,
                                                                                key=subkey,
                                                                                use_causal_attention=use_causal_attention)
            # print(f"Layer {i}:")
            # if activation_scalings[f"att_{i}"] is not None:
            #     print(quantization.quantize_forward(tokens[0:2, :], activation_scalings[f"att_{i}"]["sum"], rounding=True))
            #     # if exit_exec:
                #     exit()

            key, subkey = jax.random.split(key)
            # ATTENTION: do not vmap over keys, as the partial layer dropout has to be the same over the time dimension.
            tokens, recorded_activations[f"res_{i}"] = jax.vmap(residual_layer, (0, None, None, None, None))(tokens, 
                                                               weight_scalings[f"res_{i}"],
                                                               activation_scalings[f"res_{i}"],
                                                               inference,
                                                               subkey)
            # if activation_scalings[f"att_{i}"] is not None:
            #     print(quantization.quantize_forward(tokens[0:2, :], activation_scalings[f"res_{i}"]["sum"], rounding=True))
            # print(tokens.dtype)

        # output residual block
        key, subkey = jax.random.split(key)
        # ATTENTION: do not vmap over keys, as the partial layer dropout has to be the same over the time dimension.
        outputs, recorded_activations[f"out"]  = jax.vmap(self.output_residual_block, (0, None, None, None, None))(tokens, 
                                                                                    weight_scalings["out"],
                                                                                    activation_scalings["out"], 
                                                                                    inference, 
                                                                                    subkey)

        return outputs, recorded_activations
    
    def get_weight_dict(self) -> dict:
        """Returns the weights of the multi-head attention block as a dictionary."""
        weight_dict = {"in": self.input_residual_block.get_weight_dict()}
        weight_dict["pos_enc"] = {"weights": self.pos_embeddings}
        for i, (multi_head_attention_layer, residual_layer) in enumerate(zip(self.multi_head_attention_layers, self.residual_layers)):
            weight_dict[f"att_{i}"] = multi_head_attention_layer.get_weight_dict()
            weight_dict[f"res_{i}"] = residual_layer.get_weight_dict()
        weight_dict["out"] = self.output_residual_block.get_weight_dict()
        return weight_dict
    
    def restructure(self, input_permutation):
        restructured_input_residual_block, perm = self.input_residual_block.restructure(input_permutation)

        restructured_multi_head_attention_layers = []
        restructured_residual_layers = []

        if self.pos_embeddings is not None:
            new_pos_embeddings = self.pos_embeddings[:, perm]

        for i, (multi_head_attention_layer, residual_layer) in tqdm(enumerate(zip(self.multi_head_attention_layers, self.residual_layers)), desc="Restructuring"):
            multi_head_attention_layer, perm = multi_head_attention_layer.restructure(perm)
            residual_layer, perm = residual_layer.restructure(perm)
            restructured_multi_head_attention_layers.append(multi_head_attention_layer)
            restructured_residual_layers.append(residual_layer)

        restructured_output_residual_block, output_permutation = self.output_residual_block.restructure(perm)

        if self.pos_embeddings is not None:
            def where(pt):
                return pt.input_residual_block, pt.multi_head_attention_layers, pt.residual_layers, pt.output_residual_block, pt.pos_embeddings
            return eqx.tree_at(where, self, (restructured_input_residual_block, restructured_multi_head_attention_layers, restructured_residual_layers, restructured_output_residual_block, new_pos_embeddings)), output_permutation
        else:
            def where(pt):
                return pt.input_residual_block, pt.multi_head_attention_layers, pt.residual_layers, pt.output_residual_block
            return eqx.tree_at(where, self, (restructured_input_residual_block, restructured_multi_head_attention_layers, restructured_residual_layers, restructured_output_residual_block)), output_permutation


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
        num_params = 0

        num_params += self.input_residual_block.get_number_of_parameters()
        for m in self.multi_head_attention_layers:
            num_params += m.get_number_of_parameters()
        for m in self.residual_layers:
            num_params += m.get_number_of_parameters()
        num_params += self.output_residual_block.get_number_of_parameters()

        if self.pos_embeddings is not None:
            num_params += self.pos_embeddings.size

        return num_params



