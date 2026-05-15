
import numpy as np


def split_neurons(num_devices, num_neurons, put_on_last_node=False):
	"""
	Splits the neurons across multiple nodes.

	Returns
	-------
		split: List(int) list containing the number of neurons for each device.
	"""
	split = [num_neurons // num_devices for _ in range(num_devices)]
	i = 0
	while i < num_neurons % num_devices:
		split[i] += 1
		i += 1

	if put_on_last_node:
		split[-1] = num_neurons
		for i in range(len(split)-1):
			split[i] = 0

	return split

def get_slice_indices_from_split(split):
	"""Returns an integer list of slice indices used for mask generation.

	Parameters
	----------
	split : List of integers returned by split_neurons.

	Returns
	-------
		slice_indices : List[int] integer list of slice indices used for mask generation.

	See Also
	-------
		split_neurons

	Examples
	-------
		>>> print(get_slice_indices_from_split([4, 4, 4, 4]))
		[0, 4, 8, 12, 16]

	"""
	slice_indices = [0]
	for split_size in split:
		slice_indices.append(slice_indices[-1] + split_size)
	return slice_indices


def get_neuron_slices(num_neurons, num_devices):
	"""
	Returns an integer list of slice indices used for mask generation.

	Parameters
	----------
	num_devices : Number of physical nodes.
	num_neurons : Number of neurons to be split across the devices.

	Returns
	-------
	slice_indices : List[int] integer list of slice indices used for mask generation.

	See Also
	--------
	split_neurons

	Examples
	-------
		>>> print(get_array_slices(4, 32))
		[0, 8, 16, 24, 32]
	"""
	split = split_neurons(num_devices, num_neurons)
	return get_slice_indices_from_split(split)

def get_weight_slices(num_devices: int, dims, put_on_last_node=False):
	"""

	Parameters
	----------
	num_devices : Number of physical nodes.
	dims : Dimensions of the weight/mask matrix.

	Returns
	-------
	slice_input, slice_output : Indices used for looping over the weight matrix during mask generation.

	Examples
	-------
		>>> print(get_weight_slices(4, (32, 64)))
		([0, 8, 16, 24, 32], [0, 16, 32, 48, 64])
	"""
	split_output = split_neurons(num_devices, dims[1])
	split_input = split_neurons(num_devices, dims[0])

	if put_on_last_node:
		split_output[-1] = dims[0]

		for i in range(len(split_output) - 1):
			split_output[i] = 0

	slice_input = get_slice_indices_from_split(split_input)
	slice_output = get_slice_indices_from_split(split_output)
	return slice_input, slice_output


def split_matrix(weight, num_devices, slice_output=None, put_on_last_node=False):
	"""
	Splits the weight matrix into smaller matrices for each device.

	Parameters
	----------
	weight : Weight matrix to be split.

	Returns
	-------
	split_weight : List of smaller matrices for each device.

	Examples
	-------
		>>> print(split_matrix(np.array([[1, 2], [3, 4], [5, 6]])))
		[array([[1, 2]]), array([[3, 4]]), array([[5, 6]])]
	"""
	if len(weight.shape) == 1:
		weight = np.array([weight])
	if slice_output is None:
		slice_output = get_neuron_slices(weight.shape[1], num_devices)

	result = []
	for i in range(num_devices):
		result.append(weight[:, slice_output[i]:slice_output[i + 1]])

	return result
