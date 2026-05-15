import jax
import jax.numpy as jnp
import equinox as eqx
import omegaconf

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

from model.decoder import Decoder
from model.transformer import DenseLayer
from model.dropout import ContainsDropout
from model.pruning import ContainsPruning
from model.transformer import SavableModel

class TimeseriesPatchedDecoder(eqx.Module, ContainsPruning, ContainsDropout, SavableModel):
    dense_layers: list[DenseLayer]


    def __init__(self, 
                 input_dim,
                 cfg,
                 key,
                 apply_padding: bool = True):
        
        key, subkey = jax.random.split(key)
        
        dense_layers = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            dense_layer = DenseLayer(input_dim=100, output_dim=100, key=subkey)
            dense_layers.append(dense_layer)
        self.dense_layers = dense_layers

    def __call__(self, inputs: Array, padding_mask: Array | None, weight_scalings: dict | None, activation_scalings: dict | None, inference: bool, key: PRNGKeyArray) -> Array:
        if self.use_reversible_input_normalization:
            mean, std = _calculate_mean_std(inputs)
            # inputs = _normalize(inputs, mean, std)
        
        # padding mask
        if self.apply_padding:
            if padding_mask is None:
                padding_mask = jnp.zeros(inputs.shape[:-1], dtype=jnp.float16)
            
            inputs *= (1-padding_mask[..., None])

        # patch input
        # inputs = es.jax_einshape("(np)d->n(pd)", inputs, d=self.input_dim, p=self.input_patch_length)
        inputs = jnp.zeros((inputs.shape[0]//self.input_patch_length, self.input_patch_length * self.input_dim), dtype=inputs.dtype)


        if self.apply_padding:
            padding_mask = es.jax_einshape("(np)->np", padding_mask, p=self.input_patch_length)

            inputs = jnp.concatenate([inputs, padding_mask], axis=-1)
        
        outputs, recorded_activations = self.decoder(inputs, 
                                                     weight_scalings=weight_scalings, 
                                                     activation_scalings=activation_scalings, 
                                                     inference=inference, 
                                                     key=key)

        # outputs = es.jax_einshape("n(pd)->npd", outputs, d=self.input_dim, p=self.output_patch_length)

        # if self.use_reversible_input_normalization:
        #     outputs = _denormalize(outputs, mean, std)

        if self.use_reversible_input_normalization:
            return (outputs, mean, std)  # , recorded_activations
        else:
            return outputs  #, recorded_activations
    
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
        return 0