import jax
import numpy as np   # important: do NOT use jax. As this makes compilation very slow. We treat the mask as a constant.


from utils.splitting_utils import split_neurons
import equinox as eqx

from abc import ABC, abstractmethod

class ContainsPruning(ABC):
    """
    Every eqx.Module that contains pruning masks should inherit from this class.
    When the module contains other modules that are also from ContainsPruning, they should either be direct attributes or lists of ContainsPruning.
    """
    @abstractmethod
    def get_pruning_weights(self):
        """
        Returns the weights that should be pruned. If multiple weights should be pruned via the same mask, they shjould be combined in a list.
        """
        pass

    @abstractmethod
    def where_pruning_masks(self, pt):
        """
        Returns the location of the pruning masks in the model pt. Has to have the same order as the associated weights in get_pruning_weights.
        """
        pass

    def prune_own_masks(self, pruning_ratio):
        pruning_masks = self.where_pruning_masks(self)
        pruning_weights = self.get_pruning_weights()
        for p, w in zip(pruning_masks, pruning_weights):
            p.prune_step(w, pruning_ratio)

        return eqx.tree_at(self.where_pruning_masks, self, pruning_masks)
    
    def prune_step(self, pruning_ratio): 
        # first prune the own masks
        model = self.prune_own_masks(pruning_ratio)

        # if a leaf is of type Prunable, also prune it
        for attr_name in dir(model):
            sub = getattr(model, attr_name)
            if isinstance(sub, ContainsPruning):
                sub = sub.prune_step(pruning_ratio)
                model = eqx.tree_at(lambda x: getattr(x, attr_name), model, sub)
            elif isinstance(sub, list):
                for i in range(len(sub)):
                    if isinstance(sub[i], ContainsPruning):
                        sub[i] = sub[i].prune_step(pruning_ratio)
                        model = eqx.tree_at(lambda x: getattr(x, attr_name)[i], model, sub[i])
        return model
    
    def get_pruning_state_dict(self):
        """
        Returns the state dict of the pruning masks.
        """
        state_dict = {}
        # get the pruning masks of the submodules and own pruning masks
        for attr_name in dir(self):
            sub = getattr(self, attr_name)
            if isinstance(sub, ContainsPruning):
                state_dict[attr_name] = sub.get_pruning_state_dict()
            elif isinstance(sub, list):
                for i in range(len(sub)):
                    if isinstance(sub[i], ContainsPruning):
                        state_dict[attr_name + str(i)] = sub[i].get_pruning_state_dict()
                    if isinstance(sub[i], InterDevicePruningMask):
                        state_dict[attr_name + str(i)] = sub[i].get_state_dict()
            elif isinstance(sub, InterDevicePruningMask):
                state_dict[attr_name] = sub.get_state_dict()

        return state_dict
    
    def set_pruning_state_dict(self, pruning_state_dict):
        """
        Sets the state dict of the pruning masks.
        """
        # set the pruning masks of the submodules and own pruning masks
        model = self
        for attr_name in dir(self):
            sub = getattr(model, attr_name)
            if isinstance(sub, ContainsPruning):
                sub = sub.set_pruning_state_dict(pruning_state_dict[attr_name])
            elif isinstance(sub, list):
                for i in range(len(sub)):
                    if isinstance(sub[i], ContainsPruning):
                        sub[i] = sub[i].set_pruning_state_dict(pruning_state_dict[attr_name + str(i)])
                    if isinstance(sub[i], InterDevicePruningMask):
                        sub[i].load_state_dict(pruning_state_dict[attr_name + str(i)])
            elif isinstance(sub, InterDevicePruningMask):
                sub.load_state_dict(pruning_state_dict[attr_name])
            else:
                continue

            model = eqx.tree_at(lambda x: getattr(x, attr_name), model, sub)
        return model


class InterDevicePruningMask:
    def __init__(self, input_dims, num_devices, split_input=None):
        """
		Args:
            input_dims: tuple tuple containing the dimensions of the weight mask
            split_input: force a split of the input neurons. This is useful for the first layer, where we do not want to split the input neurons or for the attention layer.
            num_devices: number of nodes
		"""
        if split_input is None:
            self.__split_input = split_neurons(num_devices=num_devices, num_neurons=input_dims)
        else:
            self.__split_input = split_input

        self.__pruning_sub_mat = [np.ones((self.__split_input[i], 1), dtype=np.int32) for i in range(num_devices)]

        self.input_dim = input_dims

        self.__num_devices = num_devices

        self.__part_pruned_neurons = 0.0

        # we need to save this, because we cannot prune the ratio of neurons perfectly all the time. 
        # So we need to save the current ratio of pruned neurons that we want to have, so the pruning stays consistent
        # over multiple steps.
        self.__current_pruning_ratio = 0.0

    # we need those two functions for jax to recognize, when the pruning has changed and it should recompile.
    def __hash__(self):
        res = 1
        for m in self.__pruning_sub_mat:
            for e in m:
                res *= hash(int(e[0]))
        return hash(res)

    def __eq__(self, other):
        eq = True
        for i in range(len(self.__pruning_sub_mat)):
            eq = eq and np.all(self.__pruning_sub_mat[i] == other.pruning_sub_mat[i])
        return eq

    def _calculate_mask(self, dims, split_output):
        """
		
		"""
        if split_output is None:
            split_output = split_neurons(self.__num_devices, dims[1])
        
        assert dims[0] == self.input_dim
        mask = np.zeros(dims)
        current_in_pos = 0
        for ind_node_in in range(len(self.split_input)):
            s_in = self.split_input[ind_node_in]
            current_out_pos = 0
            for ind_node_out in range(len(split_output)):
                s_out = split_output[ind_node_out]
                # do not cut the connections on the same node
                # but cut the connections to the other nodes
                if ind_node_in == ind_node_out:
                    mask[current_in_pos:current_in_pos + s_in, current_out_pos:current_out_pos + s_out] = 1
                else:
                    mask[current_in_pos:current_in_pos + s_in, current_out_pos:current_out_pos + s_out] = np.tile(self.pruning_sub_mat[ind_node_in], (1, s_out))
                current_out_pos += s_out
            current_in_pos += self.split_input[ind_node_in]
        return mask


    def get_mask(self, dims, split_output, num_mask_repeat=1):
        """
        This method has to be overwritten by the subclass. It receives the dimension of the weight and then
        it should generate the weight mask and return it.
        The mask should be calculated based on the current state of the weight (e.g. current pruning step or current
        random sample for dropout). Multiple different masks for the same state are needed e.g. for LSTM-Layers.

        Parameters
        ----------
            dims: dimension of weight
            num_mask_repeat: how many times the number of weights should be repeated. dims[1] has to be a integer multiple of num_mask_repeat
        """
        assert dims[1] % num_mask_repeat == 0, "Number of mask reeats is not equal to the dims."

        single_mask = self._calculate_mask((dims[0], dims[1] // num_mask_repeat), split_output)
        mask = single_mask.repeat(1, num_mask_repeat)

        return mask

    def _calculate_score(self, neuron_weights):
        """calculates the pruning score of a neuron by looking at its weight on the output

		Parameters
		----------
			neuron_weight: torch.Tensor

		Returns
		-------
			score: float

		"""
        score = 0
        for w in neuron_weights:
            score += np.sum(np.abs(w))
        return score
    
    def prune_step(self, weights, pruning_ratio):
        """
		performs a pruning step.

		Parameters
		----------
			weights: Array because the mask can be used for multiple different weights,
					the first dimension is the index of the corresponding. The second and third are then the dimension of the weight.
		"""
        self.__current_pruning_ratio += pruning_ratio
        # for every node prune one neuron
        node_start_neuron = 0
        for node in range(self.__num_devices):
            # for every neuron of the node calculate a score. The neuron with the lowest score is pruned.
            scores = [0 for _ in range(self.__split_input[node])]
            for neuron in range(len(scores)):
                if self.__pruning_sub_mat[node][neuron] == 0:
                    scores[neuron] = 100000000
                else:
                    scores[neuron] = self._calculate_score(weights[:, node_start_neuron + neuron, :])  # because we do xT@W, the weights regarding one input are along a row.

            # calculate how many neurons should be pruned. This is achieved by taking the difference between the number of neurons that are already pruned and the number of neurons that should be pruned in total.
            num_currently_pruned_neurons = len(self.__pruning_sub_mat[node]) - np.sum(self.__pruning_sub_mat[node])
            num_neurons_to_prune_total = int(round((len(self.__pruning_sub_mat[node]) * self.__current_pruning_ratio)))
            if num_neurons_to_prune_total == len(self.__pruning_sub_mat[node]):
                num_neurons_to_prune_total -= 1  # we always need to have at least one neuron left.
            num_neurons_to_prune = max(num_neurons_to_prune_total - num_currently_pruned_neurons, 0)
            if num_neurons_to_prune > 0:
                # prune the neuron with the lowest score
                neuron_to_prune = np.argsort(np.array(scores))[0:num_neurons_to_prune]
                self.__pruning_sub_mat[node][neuron_to_prune] = 0

            node_start_neuron += self.__split_input[node]

    
    @property
    def pruning_sub_mat(self):
        return self.__pruning_sub_mat

    @property
    def split_input(self):
        return self.__split_input

    def load_state_dict(self, pruning_state_dict):
        self.__pruning_sub_mat = pruning_state_dict["pruning_sub_mat"]
        self.__part_pruned_neurons = pruning_state_dict["part_pruned_neurons"]
        self.__current_pruning_ratio = pruning_state_dict["current_pruning_ratio"]
    
    def get_state_dict(self):
        return {"pruning_sub_mat": self.__pruning_sub_mat, "part_pruned_neurons": self.__part_pruned_neurons, "current_pruning_ratio": self.__current_pruning_ratio}


if __name__ == "__main__":
    mask = InterDevicePruningMask(8, 2)
    print(mask.get_mask((8, 12)))

    weight = np.random.randn(1, 8, 12)
    mask.prune_step(weight, 3)

    print(mask.get_mask((8, 12)))
