import math
import numpy as np
# from tqdm import tqdm
from multiprocessing import Pool, get_context

from hardware_implementation.code_generation.dnni_config import generate_dnni_config
from hardware_implementation.code_generation.dnni_mixer_config import generate_dnni_mixer_config
from hardware_implementation.code_generation.model import generate_model_c
import model.quantization_none_jax as quantization
from utils.splitting_utils import get_slice_indices_from_split, get_weight_slices, split_neurons, split_matrix, get_neuron_slices


def generate_matrix_code(matrix, use_int):
    data = ""
    if len(matrix.shape) == 1:
        for i in range(len(matrix)):
            if use_int:
                data += f"{int(matrix[i])}, "
            else:
                data += f"{float(matrix[i])}f, "
    else:
        for i in range(len(matrix)):
            data += f"{generate_matrix_code(matrix[i], use_int)}"
    return data[0:-1]


def combine_code(codes):
    """
    Combine the code for each device into one string. We are given a list of codes for each device.
    i.e. [["code_1_dev1", code_1_dev2"], 
          ["code_2_dev1", "code_2_dev2"]]
    We want to combine this into one string for each device.
    i.e. ["code_1_dev1\ncode_2_dev1", "code_1_dev2\ncode_2_dev2"]
    """
    c_codes_return = []
    for i in range(len(codes[0])):
        c_codes_return.append("")
        for j in range(len(codes)):
            c_codes_return[i] += codes[j][i]
            c_codes_return[i] += "\n"
    
    return c_codes_return


def get_weight_splits_for_device(weight, bias, num_devices, pruning_sub_mat, split_input, split_output):
    slice_output = get_slice_indices_from_split(split_output)
    slice_input = get_slice_indices_from_split(split_input)
    weight_splits = split_matrix(weight, num_devices, slice_output=slice_output)
    bias_splits = split_matrix(bias, num_devices)
    input_range = [0]
    weight_splits_return = []
    input_range_return = []

    # apply inter-device pruning
    for i in range(num_devices):
        input_range = [0]
        for j in range(num_devices):
            if j == i:
                input_range.append(input_range[-1] + len(pruning_sub_mat[j]))
            else:
                input_range.append(input_range[-1] + np.sum(pruning_sub_mat[j]))
        input_range_return.append(input_range)

        weight = np.zeros((input_range[-1], slice_output[i+1] - slice_output[i]))

        for j in range(num_devices):
            if j == i:
                weight[input_range[j]:input_range[j + 1], :] = weight_splits[i][slice_input[j]:slice_input[j + 1], :]
            else:
                weight_part = weight_splits[i][slice_input[j]:slice_input[j + 1], :]
                weight[input_range[j]:input_range[j + 1], :] = weight_part[pruning_sub_mat[j].flatten()==1, :]
        weight_splits_return.append(weight)
    
    return weight_splits_return, bias_splits, input_range_return

def export_positional_encoding(name, pos_enc: np.array, scaling_weight, scaling_input, scaling_output, num_devices, num_bits=8):
    """
    Export positional encoding to a quantized format.
    """
    c_codes = []
    split_input = split_neurons(num_devices=num_devices, num_neurons=pos_enc.shape[1])
    slices_input = get_slice_indices_from_split(split_input)

    # apply inter-device pruning
    for i in range(num_devices):
        w_q = quantization.quantize_forward(pos_enc[:, slices_input[i]:slices_input[i+1]], scaling_weight, num_bits)
        
        c_code = f"static const int8_t pos_enc_{name}[] = {{ {generate_matrix_code(w_q, True)} }};\n"
        c_code += f"static const cmsis_nn_dims pos_enc_dims_{name} = {{.n={w_q.shape[0]}, .h=1, .w=1, .c={w_q.shape[1]}}};\n"
        c_code += f"static const float input_scaling_{name} = {1 / scaling_input}f;\n"
        c_code += f"static const float pos_enc_scaling_{name} = {1 / scaling_weight}f;\n"
        c_code += f"static const float sum_scaling_{name} = {scaling_output}f;\n"
        c_codes.append(c_code)
    return c_codes

def export_fc(name, weights_dict: dict, pruning_sub_mat: np.array, scaling_weight, scaling_activation_in, scaling_activation_out, num_devices, split_input, split_output, num_bits=8, use_relu=True):
    """
    Export weight to a quantized format.
    """
    # print(f"Exporting FC layer: {name}")
    c_codes = []
    input_feature_size = 0
    output_feature_size = 0

    weight_splits, bias_splits, input_ranges = get_weight_splits_for_device(weights_dict["weights"], weights_dict["bias"], num_devices, pruning_sub_mat, split_input, split_output)

    # apply inter-device pruning
    for i in range(num_devices):
        w_q = quantization.quantize_forward(weight_splits[i], scaling_weight, num_bits)
        b_q = quantization.quantize_forward(bias_splits[i], scaling_activation_in * scaling_weight, num_bits=32)

        scaling = (scaling_activation_out) / (scaling_weight * scaling_activation_in)
        cmsis_nn_quant_params = quantization.calculate_cmsis_nn_quant_params(scaling, num_bits)

        # arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
        #                                        const cmsis_nn_fc_params *fc_params,
        #                                        const cmsis_nn_per_tensor_quant_params *quant_params,
        #                                        const cmsis_nn_dims *input_dims,
        #                                        const int8_t *input_data,
        #                                        const cmsis_nn_dims *filter_dims,
        #                                        const int8_t *filter_data,
        #                                        const cmsis_nn_dims *bias_dims,
        #                                        const int32_t *bias_data,
        #                                        const cmsis_nn_dims *output_dims,
        #                                        int8_t *output_data);
        input_feature_size = max(len(w_q), input_feature_size)
        output_feature_size = max(len(w_q[0]), output_feature_size)

        # we need to transpose, because arm_nn_vec_mat_mult_t_s8 uses the transposed weight in memory (for speed reasons)
        c_code = f"static const int8_t weight_{name}[] = {{ {generate_matrix_code(w_q.T, True)} }};\n"
        c_code += f"static const int32_t bias_{name}[] = {{ {generate_matrix_code(b_q, True)} }};\n"
        c_code += f"static const cmsis_nn_dims weight_dims_{name} = {{.n={w_q.shape[0]}, .h=1, .w=1, .c={w_q.shape[1]}}};\n"
        c_code += f"static const cmsis_nn_dims bias_dims_{name} = {{.n=1, .h=1, .w=1, .c={w_q.shape[1]}}};\n"
        c_code += f"static const cmsis_nn_dims input_dims_{name} = {{.n=LENGTH_TIMESERIES, .h=1, .w=1, .c={w_q.shape[0]}}};\n"
        c_code += f"static const cmsis_nn_fc_params fc_params_{name} = {{.input_offset=0, .filter_offset=0, .output_offset=0, .activation={'RELU' if use_relu else 'LINEAR'}}};\n"
        c_code += f"static const uint32_t input_range_{name}[] = {{ {generate_matrix_code(np.array(input_ranges[i]), True)} }};\n"
        c_code += f"static const uint8_t pruning_{name}[] = {{{generate_matrix_code(pruning_sub_mat[i], True)}}};\n"
        
        c_code += f"static fc_layer_config_t config_{name} = {{\n"
        c_code += f"    .weight=weight_{name},\n"
        c_code += f"    .bias=bias_{name},\n"
        c_code += f"    .weight_dims=&weight_dims_{name},\n"
        c_code += f"    .bias_dims=&bias_dims_{name},\n"
        c_code += f"    .input_dims=&input_dims_{name},\n"
        c_code += f"    .ctx=&ctx,\n"
        c_code += f"    .quant_params={scaling},\n"
        c_code += f"    .fc_params=&fc_params_{name},\n"
        c_code += f"    .input_range=input_range_{name},\n"
        c_code += f"    .input_buffer=input_buffer,\n"
        c_code += f"    .message_assignment=&message_assignment_{name},\n"
        c_code += f"    .pruning=pruning_{name}\n"
        c_code += f"}};\n"

        c_codes.append(c_code)
    return c_codes, [float(np.sum(pruning_sub_mat[i])) for i in range(num_devices)], input_feature_size, output_feature_size


def export_layernorm(name, layernorm_dict, scaling_activation_in, scaling_activation_out, pruning_masks):
    # print(f"Exporting layernorm: {name}")
    c_codes = []
    for i in range(len(pruning_masks)):
        mask = []
        for j in range(len(pruning_masks)):
            if i == j:
                mask.append(np.ones_like(pruning_masks[j]))
            else:
                mask.append(pruning_masks[j])
        mask = np.concatenate(mask, axis=0).flatten()

        scaling = layernorm_dict["rescale"][mask > 0.5] * scaling_activation_out
        c_code = f"static const float layernorm_multiplier_{name}[] = {{ {generate_matrix_code(scaling, False)} }};\n"
        c_code += f"static const float layernorm_bias_{name}[] = {{ {generate_matrix_code(layernorm_dict["bias"][mask > 0.5], False)} }};\n"
        
        c_code += f"static const layernorm_config_t layernorm_config_{name} = {{\n"
        c_code += f"    .multiplier = layernorm_multiplier_{name},\n"
        c_code += f"    .bias = layernorm_bias_{name},\n"
        c_code += f"    .scaling_in = {1 / scaling_activation_in},\n"
        c_code += f"}};\n"
        c_codes.append(c_code)
    return c_codes


def export_residual_block(name, weight_dict: dict, pruning_dict: dict, scalings_weight, scalings_activation, num_devices, num_bits=8):
    # print(f"Exporting residual block: {name}")
    message_sizes = {}
    input_feature_size = 0
    output_feature_size = 0
    # in case we do not transform the residual, this is the size of the residual
    residual_feature_size = split_neurons(num_devices=num_devices, num_neurons=weight_dict["0"]["weights"].shape[0])[0]
    c_codes = []
    layernorm_config = "NULL"
    if "partial_layer_norm" in weight_dict:
        c_code = export_layernorm(name, weight_dict["partial_layer_norm"], scalings_activation["in"], scalings_activation["0"], pruning_dict["pruning_masks0"]["pruning_sub_mat"])
        c_codes.append(c_code)
        layernorm_config = f"&layernorm_config_{name}"

    num_layers = len([k for k in weight_dict.keys() if k.isdigit()])

    for i in range(num_layers):
        c_code, ms, input_feature_size_temp, output_feature_size_temp = export_fc(f"{name}_{str(i)}", 
                                                            weights_dict=weight_dict[f"{i}"], 
                                                            pruning_sub_mat=pruning_dict[f"pruning_masks{i}"]["pruning_sub_mat"], 
                                                            scaling_weight=float(scalings_weight[f"{i}"]["weights"]), 
                                                            scaling_activation_in=float(scalings_activation[f"{i}"]), 
                                                            scaling_activation_out=float(scalings_activation[f"{i+1}"]), 
                                                            num_devices=num_devices,
                                                            split_input=split_neurons(num_devices=num_devices, num_neurons=weight_dict[f"{i}"]["weights"].shape[0]),
                                                            split_output=split_neurons(num_devices=num_devices, num_neurons=weight_dict[f"{i}"]["weights"].shape[1]), 
                                                            num_bits=num_bits,
                                                            use_relu=i<num_layers-1)
        message_sizes[f"{name}_{i}"] = ms
        input_feature_size = max(input_feature_size, input_feature_size_temp)
        output_feature_size = max(output_feature_size, output_feature_size_temp)
        c_codes.append(c_code)
    
    if "residual" in weight_dict:
        c_code, ms, _, residual_feature_size = export_fc(f"{name}_residual", 
                           weights_dict=weight_dict["residual"], 
                           pruning_sub_mat=pruning_dict["pruning_masks0"]["pruning_sub_mat"], 
                           scaling_weight=float(scalings_weight[f"residual"]["weights"]), 
                           scaling_activation_in=float(scalings_activation[f"in"]), 
                           scaling_activation_out=float(scalings_activation["residual"]), 
                           num_devices=num_devices, 
                           split_input=split_neurons(num_devices=num_devices, num_neurons=weight_dict[f"residual"]["weights"].shape[0]),
                           split_output=split_neurons(num_devices=num_devices, num_neurons=weight_dict[f"residual"]["weights"].shape[1]),
                           num_bits=num_bits,
                           use_relu=False)
        message_sizes[f"{name}_residual"] = ms
        c_codes.append(c_code)

    # add config and scalings
    config_code = []
    for i in range(num_devices):
        code = f"static const float input_scaling_{name} = {1 / scalings_activation[f'{num_layers}']}f;\n"
        code += f"static const float residual_scaling_{name} = {1 / scalings_activation['residual']}f;\n"
        code += f"static const float sum_scaling_{name} = {scalings_activation['sum']}f;\n"
        
        # generate config for residual block
        code += f"static fc_layer_config_t *layers_{name}[]={{"
        for j in range(num_layers):
            code += f"&config_{name}_{j}, "
        code += f"}};\n"
        code += f"static residual_block_config_t residual_block_config_{name} = {{\n"
        code += f"    .layers = layers_{name},\n"
        code += f"    .num_layers = {num_layers},\n"
        code += f"    .residual_layer = {f'&config_{name}_residual' if "residual" in weight_dict else 'NULL'},\n"
        code += f"    .input_scaling = input_scaling_{name},\n"
        code += f"    .residual_scaling = residual_scaling_{name},\n"
        code += f"    .sum_scaling = sum_scaling_{name},\n"
        # code += f"    .buffer = buffer,\n"
        code += f"    .residual_buffer = o_buffer,\n"   # residual_buffer,\n"  # reuse q_buffer, as unused during residual block
        code += f"    .layernorm_config = {layernorm_config},\n"
        code += f"}};\n"
        config_code.append(code)
        config_code.append(code)
    c_codes.append(config_code)

    # print(f"Finished exporting residual block {name}.")
    
    return combine_code(c_codes), message_sizes, input_feature_size, output_feature_size, residual_feature_size




def _export_qkv(name, weight, bias, scaling_weight, scaling_activation_in, scaling_activation_out, num_bits=8):
    w_q = quantization.quantize_forward(weight, scaling_weight, num_bits)
    b_q = quantization.quantize_forward(bias, scaling_activation_in * scaling_weight, num_bits=32)

    c_code = f"static const int8_t weight_{name}[] = {{ {generate_matrix_code(w_q.T, True)} }};\n"
    c_code += f"static const int32_t bias_{name}[] = {{ {generate_matrix_code(b_q, True)} }};\n"

    return c_code


def _export_list(element_type, is_pointer, name, elements):
    c_code = f"static const {element_type} {'*' if is_pointer else ''}{name}[] = {{"
    for i in range(len(elements)):
        c_code += f"{elements[i]}, "
    c_code += f"}};\n"
    return c_code


def export_multi_head_attention_block(name, weight_dict: dict, pruning_dict: dict, num_heads, o_split_input, scalings_weight, scalings_activation, num_devices, num_bits=8):
    # print(f"Exporting multi-head attention block: {name}")
    message_sizes = {}
    input_feature_size = 0   
    output_feature_size = 0
    feature_dim = weight_dict["o"]["weights"].shape[0]
    residual_feature_size = split_neurons(num_devices=num_devices, num_neurons=feature_dim)[0]
    q_feature_size = feature_dim // num_heads
    c_codes = []

    def convert_to_float(obj):
        """Recursively convert nested dict values to float."""
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        else:
            return float(obj)

    scalings_weight = convert_to_float(scalings_weight)
    scalings_activation = convert_to_float(scalings_activation)

    head_slices = get_neuron_slices(num_heads, num_devices)

    message_sizes[f"attention_input_{name}"] = [float(np.sum(pruning_dict["pruning_attention"]["pruning_sub_mat"][i])) for i in range(num_devices)]

    wq_split, bq_split, input_range_list = get_weight_splits_for_device(weight=weight_dict["q"]["weights"],
                                                                        bias=weight_dict["q"]["bias"],
                                                                        num_devices=num_devices,
                                                                        pruning_sub_mat=pruning_dict["pruning_attention"]["pruning_sub_mat"],
                                                                        split_input=split_neurons(num_devices=num_devices, num_neurons=weight_dict["q"]["weights"].shape[0]),
                                                                        split_output=o_split_input)
    
    wk_split, bk_split, _ = get_weight_splits_for_device(weight=weight_dict["k"]["weights"],
                                                                        bias=weight_dict["k"]["bias"],
                                                                        num_devices=num_devices,
                                                                        pruning_sub_mat=pruning_dict["pruning_attention"]["pruning_sub_mat"],
                                                                        split_input=split_neurons(num_devices=num_devices, num_neurons=weight_dict["k"]["weights"].shape[0]),
                                                                        split_output=o_split_input)
    
    wv_split, bv_split, _ = get_weight_splits_for_device(weight=weight_dict["v"]["weights"],
                                                                        bias=weight_dict["v"]["bias"],
                                                                        num_devices=num_devices,
                                                                        pruning_sub_mat=pruning_dict["pruning_attention"]["pruning_sub_mat"],
                                                                        split_input=split_neurons(num_devices=num_devices, num_neurons=weight_dict["v"]["weights"].shape[0]),
                                                                        split_output=o_split_input)

    # export W_o
    name_o_layer = f"o_layer_{name}"
    o_layer_c_code, message_sizes[f"o_layer_{name}"], _, output_feature_size = export_fc(name_o_layer, 
                weights_dict=weight_dict["o"], 
                pruning_sub_mat=pruning_dict["pruning_o"]["pruning_sub_mat"],
                scaling_weight=float(scalings_weight["o"]["weights"]), 
                scaling_activation_in=float(scalings_activation["heads"]), 
                scaling_activation_out=float(scalings_activation["o"]), 
                num_devices=num_devices, 
                split_input=o_split_input,
                split_output=split_neurons(num_devices=num_devices, num_neurons=weight_dict["o"]["weights"].shape[1]),
                num_bits=num_bits,
                use_relu=False)
    
    
    o_feature_size = 0
    for i in range(num_devices):
        q_scalings = []
        k_scalings = []
        v_scalings = []
        c_code = ""
        o_feature_size = max(o_feature_size, wv_split[i].shape[1])
        for h in range(head_slices[i], head_slices[i+1]):
            fd = feature_dim // num_heads
            # print(multi_head_attention_block.dense_qs.weights.shape)
            # print(fd)
            # exit(0)
            relative_head_idx = h - head_slices[i]
            input_feature_size = len(wq_split[i][:, relative_head_idx*fd:(relative_head_idx+1)*fd])
            c_code += _export_qkv(f"q_{h-head_slices[i]}_{name}", 
                                 weight=wq_split[i][:, relative_head_idx*fd:(relative_head_idx+1)*fd], 
                                 bias=bq_split[i], 
                                 scaling_weight=scalings_weight["q"]["weights"],
                                 scaling_activation_in=scalings_activation["layer_norm"], 
                                 scaling_activation_out=scalings_activation["q"], 
                                 num_bits=8)
            c_code += _export_qkv(f"k_{h-head_slices[i]}_{name}", 
                                 weight=wk_split[i][:, relative_head_idx*fd:(relative_head_idx+1)*fd], 
                                 bias=bk_split[i], 
                                 scaling_weight=scalings_weight["k"]["weights"],
                                 scaling_activation_in=scalings_activation["layer_norm"], 
                                 scaling_activation_out=scalings_activation["k"], 
                                 num_bits=8)
            c_code += _export_qkv(f"v_{h-head_slices[i]}_{name}", 
                                 weight=wv_split[i][:, relative_head_idx*fd:(relative_head_idx+1)*fd], 
                                 bias=bv_split[i], 
                                 scaling_weight=scalings_weight["v"]["weights"],
                                 scaling_activation_in=scalings_activation["layer_norm"], 
                                 scaling_activation_out=scalings_activation["v"], 
                                 num_bits=8)
            
            q_scalings.append(scalings_activation["q"] / (scalings_weight["q"]["weights"] * scalings_activation["layer_norm"]))
            k_scalings.append(scalings_activation["k"] / (scalings_weight["k"]["weights"] * scalings_activation["layer_norm"]))
            v_scalings.append(scalings_activation["v"] / (scalings_weight["v"]["weights"] * scalings_activation["layer_norm"]))

            w = wq_split[i][:, relative_head_idx*fd:(relative_head_idx+1)*fd]
            
            # scaling = 1 / (scalings_activation["q"] * scalings_activation["k"]) * 1 / math.sqrt(fd)
            # cmsis_nn_quant_params = quantization.calculate_cmsis_nn_quant_params(scaling, num_bits)

            # c_code += f"static const cmsis_nn_per_tensor_quant_params att_quant_params_{h-head_slices[i]}_{name} = {{.multiplier={cmsis_nn_quant_params[0]}, .shift={cmsis_nn_quant_params[1] + 8} }};\n"

            c_code += f"static const float softmax_times_v_scaling_{h-head_slices[i]}_{name} = {scalings_activation['qkv']};\n"
            c_code += f"static const float v_scaling_{h-head_slices[i]}_{name} = {1/scalings_activation['v']};\n"
        
        c_code += f"static const cmsis_nn_dims input_attention_weight_dims_{name} = {{.n={w.shape[0]}, .h=1, .w=1, .c={w.shape[1]}}};\n"
        c_code += f"static const cmsis_nn_dims input_attention_bias_dims_{name} = {{.n=1, .h=1, .w=1, .c={w.shape[1]}}};\n"
        c_code += f"static const cmsis_nn_dims input_attention_input_dims_{name} = {{.n=LENGTH_TIMESERIES, .h=1, .w=1, .c={w.shape[0]}}};\n"
        
        num_elements = head_slices[i+1]-head_slices[i]
        c_code += _export_list("int8_t", True, f"q_weight_list_{name}", [ f"&weight_q_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("int8_t", True, f"k_weight_list_{name}", [ f"&weight_k_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("int8_t", True, f"v_weight_list_{name}", [ f"&weight_v_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("int8_t", True, f"q_bias_list_{name}", [ f"&bias_q_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("int8_t", True, f"k_bias_list_{name}", [ f"&bias_k_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("int8_t", True, f"v_bias_list_{name}", [ f"&bias_v_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("float", False, f"q_quant_params_list_{name}", q_scalings)
        c_code += _export_list("float", False, f"k_quant_params_list_{name}", k_scalings)
        c_code += _export_list("float", False, f"v_quant_params_list_{name}", v_scalings)
        c_code += _export_list("float", False, f"v_scaling_list_{name}", [f"v_scaling_{j}_{name}" for j in range(num_elements)])
        c_code += _export_list("float", False, f"softmax_times_v_scaling_list_{name}", [f"softmax_times_v_scaling_{j}_{name}" for j in range(num_elements)])
        c_code += f"static const uint8_t input_pruning_{name}[] = {{{generate_matrix_code(pruning_dict["pruning_attention"]["pruning_sub_mat"][i], True)}}};\n"

        c_code += export_layernorm(name, weight_dict["partial_layer_norm"], scalings_activation["in"], scalings_activation["layer_norm"], pruning_dict["pruning_attention"]["pruning_sub_mat"])[i]

        c_code += f"static const cmsis_nn_fc_params fc_params_{name} = {{.input_offset=0, .filter_offset=0, .output_offset=0, .activation=LINEAR}};\n"
        c_code += f"static const uint32_t input_range_{name}[] = {{ {generate_matrix_code(np.array(input_range_list[i]), True)} }};\n"

        # # export W_o
        # name_o_layer = f"o_layer_{name}"
        # c, message_sizes[f"o_layer_{name}"], o_feature_size, output_feature_size = export_fc(name_o_layer, 
        #             weights_dict=weight_dict["o"], 
        #             pruning_sub_mat=pruning_dict["pruning_o"]["pruning_sub_mat"],
        #             scaling_weight=float(scalings_weight["o"]["weights"]), 
        #             scaling_activation_in=float(scalings_activation["heads"]), 
        #             scaling_activation_out=float(scalings_activation["o"]), 
        #             num_devices=num_devices, 
        #             split_input=o_split_input,
        #             split_output=split_neurons(num_devices=num_devices, num_neurons=weight_dict["o"]["weights"].shape[1]),
        #             num_bits=num_bits,
        #             use_relu=False)
        # c_code += f"{c[i]}\n"
        c_code += f"{o_layer_c_code[i]}\n"
        
        scaling = 1 / (scalings_activation["q"] * scalings_activation["k"])
        c_code += f"static multi_head_attention_block_config_t mha_config_{name} = {{\n"
        c_code += f"    .layernorm_config = &layernorm_config_{name},\n"
        c_code += f"    .num_heads = {num_elements},\n"
        c_code += f"    .q_weight_list = q_weight_list_{name},\n"
        c_code += f"    .q_bias_list = q_bias_list_{name},\n"
        c_code += f"    .q_quant_params_list = q_quant_params_list_{name},\n"
        c_code += f"    .k_weight_list = k_weight_list_{name},\n"
        c_code += f"    .k_bias_list = k_bias_list_{name},\n"
        c_code += f"    .k_quant_params_list = k_quant_params_list_{name},\n"
        c_code += f"    .v_weight_list = v_weight_list_{name},\n"
        c_code += f"    .v_bias_list = v_bias_list_{name},\n"
        c_code += f"    .v_quant_params_list = v_quant_params_list_{name},\n"
        c_code += f"    .v_scaling_list = v_scaling_list_{name},\n"
        c_code += f"    .softmax_times_v_scaling_list = softmax_times_v_scaling_list_{name},\n"
        c_code += f"    .input_weight_dims = &input_attention_weight_dims_{name},\n"
        c_code += f"    .input_bias_dims = &input_attention_bias_dims_{name},\n"
        c_code += f"    .att_quant_params = {scaling},\n"
        c_code += f"    .inv_sqrt = {1 / math.sqrt(fd)},\n"
        c_code += f"    .input_dims = &input_attention_input_dims_{name},\n"
        c_code += f"    .ctx = &ctx,\n"
        c_code += f"    .fc_params = &fc_params_{name},\n"
        c_code += f"    .input_range = input_range_{name},\n"
        c_code += f"    .input_message_assignment = &message_assignment_attention_input_{name},\n"
        c_code += f"    .input_pruning = input_pruning_{name},\n"
        c_code += f"    .input_buffer = input_buffer,\n"  # = buffer,\n   # here, we need the larger buffer, (see code).
        c_code += f"    .q_buffer = q_buffer,\n"
        c_code += f"    .k_buffer = k_buffer,\n"
        c_code += f"    .v_buffer = v_buffer,\n"
        c_code += f"    .o_buffer = o_buffer,\n"
        c_code += f"    .qk_buffer = qk_buffer,\n"
        c_code += f"    .o_layer_config = &config_o_layer_{name},\n"
        c_code += f"    .input_scaling = {1 / scalings_activation['in']},\n"   # {1 / scalings_activation['layer_norm']},\n"
        c_code += f"    .o_layer_output_scaling = {1 / scalings_activation['o']},\n"
        c_code += f"    .sum_scaling = {scalings_activation['sum']},\n"
        c_code += f"}};\n"

        c_codes.append(c_code)

    return c_codes, message_sizes, input_feature_size, q_feature_size, o_feature_size, output_feature_size, residual_feature_size


# Wrapper functions for parallel execution
def _export_input_residual_wrapper(args):
    """Wrapper for parallel execution of input residual block export."""
    weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits = args
    return export_residual_block(
        name="input_residual",
        weight_dict=weight_dict["in"],
        pruning_dict=pruning_dict["input_residual_block"],
        scalings_weight=scalings_weight["in"],
        scalings_activation=scalings_activation["in"],
        num_devices=num_devices,
        num_bits=num_bits
    )


def _export_positional_encoding_wrapper(args):
    """Wrapper for parallel execution of positional encoding export."""
    weight_dict, scalings_weight, scalings_activation, num_devices, num_bits = args
    return export_positional_encoding(
        name="pos_enc",
        pos_enc=weight_dict["pos_enc"]["weights"],
        scaling_weight=scalings_weight["pos_enc"]["weights"],
        scaling_input=scalings_activation["before_pos_embedding"],
        scaling_output=scalings_activation["after_pos_embedding"],
        num_devices=num_devices,
        num_bits=num_bits
    )


def _export_attention_block_wrapper(args):
    """Wrapper for parallel execution of attention block export."""
    i, weight_dict, pruning_dict, num_heads, o_split_input, scalings_weight, scalings_activation, num_devices, num_bits = args
    return export_multi_head_attention_block(
        name=f"attention_block_{i}",
        weight_dict=weight_dict[f"att_{i}"],
        pruning_dict=pruning_dict[f"multi_head_attention_layers{i}"],
        num_heads=num_heads,
        o_split_input=o_split_input,
        scalings_weight=scalings_weight[f"att_{i}"],
        scalings_activation=scalings_activation[f"att_{i}"],
        num_devices=num_devices,
        num_bits=num_bits
    )


def _export_residual_block_wrapper(args):
    """Wrapper for parallel execution of residual block export."""
    i, weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits = args
    return export_residual_block(
        name=f"residual_block_{i}",
        weight_dict=weight_dict[f"res_{i}"],
        pruning_dict=pruning_dict[f"residual_layers{i}"],
        scalings_weight=scalings_weight[f"res_{i}"],
        scalings_activation=scalings_activation[f"res_{i}"],
        num_devices=num_devices,
        num_bits=num_bits
    )


def _export_output_residual_wrapper(args):
    """Wrapper for parallel execution of output residual block export."""
    weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits = args
    return export_residual_block(
        name="output_residual",
        weight_dict=weight_dict["out"],
        pruning_dict=pruning_dict["output_residual_block"],
        scalings_weight=scalings_weight["out"],
        scalings_activation=scalings_activation["out"],
        num_devices=num_devices,
        num_bits=num_bits
    )


def export_transformer(name, weight_dict: dict, pruning_dict: dict, num_heads, o_split_input, scalings_weight, scalings_activation, num_devices, example_input, num_bits=8, attention_only=False, path_output=None):
    """
    Exports the transformer model to C code.
    Uses multiprocessing to parallelize layer exports for speedup.
    """

    # fist generate all the exported configs and weights for each layer
    layer_data_code = []
    message_sizes = {}
    input_feature_size = 0
    q_feature_size = 0
    o_feature_size = 0
    output_feature_size = 0
    residual_feature_size = 0

    pruning_dict = pruning_dict["decoder"]

    # Convert JAX arrays to NumPy once before parallelization
    # weight_dict = _convert_jax_to_numpy(weight_dict)
    # pruning_dict = _convert_jax_to_numpy(pruning_dict)

    if not attention_only:
        num_residual_layers = len([k for k in weight_dict.keys() if k.startswith("att_")])
        
        # Prepare all tasks for parallel execution
        tasks = []
        
        # Task 0: Input residual block
        tasks.append(('input_residual', (weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits)))
        
        # Task 1: Positional encoding
        tasks.append(('pos_enc', (weight_dict, scalings_weight, scalings_activation, num_devices, num_bits)))
        
        # Tasks 2+: Attention and residual blocks (interleaved)
        for i in range(num_residual_layers):
            tasks.append(('attention', (i, weight_dict, pruning_dict, num_heads, o_split_input, scalings_weight, scalings_activation, num_devices, num_bits)))
            tasks.append(('residual', (i, weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits)))
        
        # Last task: Output residual block
        tasks.append(('output_residual', (weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits)))
        
        # Execute tasks in parallel
        # print(f"Exporting {len(tasks)} layers in parallel...")
        # Use 'spawn' context to avoid JAX/NumPy issues with fork
        with get_context('spawn').Pool() as pool:
            results = []
            for task_type, task_args in tasks:
                if task_type == 'input_residual':
                    results.append(pool.apply_async(_export_input_residual_wrapper, (task_args,)))
                elif task_type == 'pos_enc':
                    results.append(pool.apply_async(_export_positional_encoding_wrapper, (task_args,)))
                elif task_type == 'attention':
                    results.append(pool.apply_async(_export_attention_block_wrapper, (task_args,)))
                elif task_type == 'residual':
                    results.append(pool.apply_async(_export_residual_block_wrapper, (task_args,)))
                elif task_type == 'output_residual':
                    results.append(pool.apply_async(_export_output_residual_wrapper, (task_args,)))
            
            # Collect results in order
            # print("Waiting for parallel tasks to complete...")
            for idx, (task_type, _) in enumerate(tasks):
                # print(f"Collecting result for task {idx+1}/{len(tasks)}: {task_type}")
                result = results[idx].get()
                
                if task_type == 'pos_enc':
                    # Positional encoding only returns c_code
                    layer_data_code.append(result)
                elif task_type in ['input_residual', 'residual', 'output_residual']:
                    # Residual blocks return (c_code, ms, input_fs, output_fs, residual_fs)
                    c_code, ms, input_fs_temp, output_fs_temp, residual_fs_temp = result
                    layer_data_code.append(c_code)
                    message_sizes.update(ms)
                    input_feature_size = max(input_feature_size, input_fs_temp)
                    output_feature_size = max(output_feature_size, output_fs_temp)
                    residual_feature_size = max(residual_feature_size, residual_fs_temp)
                elif task_type == 'attention':
                    # Attention blocks return (c_code, ms, input_fs, q_fs, o_fs, output_fs, residual_fs)
                    c_code, ms, input_fs_temp, q_fs_temp, o_fs_temp, output_fs_temp, residual_fs_temp = result
                    layer_data_code.append(c_code)
                    message_sizes.update(ms)
                    input_feature_size = max(input_feature_size, input_fs_temp)
                    q_feature_size = max(q_feature_size, q_fs_temp)
                    o_feature_size = max(o_feature_size, o_fs_temp)
                    output_feature_size = max(output_feature_size, output_fs_temp)
                    residual_feature_size = max(residual_feature_size, residual_fs_temp)
    else:
        # attention_only mode - export single attention and residual block in parallel
        # print("Exporting attention-only mode in parallel...")
        
        tasks = [
            ('attention', (0, weight_dict, pruning_dict, num_heads, o_split_input, scalings_weight, scalings_activation, num_devices, num_bits)),
            ('residual', (0, weight_dict, pruning_dict, scalings_weight, scalings_activation, num_devices, num_bits))
        ]
        
        with get_context('spawn').Pool(processes=2) as pool:
            results = []
            for task_type, task_args in tasks:
                if task_type == 'attention':
                    results.append(pool.apply_async(_export_attention_block_wrapper, (task_args,)))
                elif task_type == 'residual':
                    results.append(pool.apply_async(_export_residual_block_wrapper, (task_args,)))
            
            # Collect results
            for idx, (task_type, _) in enumerate(tasks):
                result = results[idx].get()
                
                if task_type == 'attention':
                    c_code, ms, input_fs_temp, q_fs_temp, o_fs_temp, output_fs_temp, residual_fs_temp = result
                    layer_data_code.append(c_code)
                    message_sizes.update(ms)
                    input_feature_size = max(input_feature_size, input_fs_temp)
                    q_feature_size = max(q_feature_size, q_fs_temp)
                    o_feature_size = max(o_feature_size, o_fs_temp)
                    output_feature_size = max(output_feature_size, output_fs_temp)
                    residual_feature_size = max(residual_feature_size, residual_fs_temp)
                elif task_type == 'residual':
                    c_code, ms, input_fs_temp, output_fs_temp, residual_fs_temp = result
                    layer_data_code.append(c_code)
                    message_sizes.update(ms)
                    input_feature_size = max(input_feature_size, input_fs_temp)
                    output_feature_size = max(output_feature_size, output_fs_temp)
                    residual_feature_size = max(residual_feature_size, residual_fs_temp)

        num_residual_layers = 2

    # return combine_code(c_codes), message_sizes, input_feature_size, q_feature_size, o_feature_size, output_feature_size, residual_feature_size
    layer_data_code = combine_code(layer_data_code)

    # export input
    if not attention_only:
        input_slices = get_neuron_slices(example_input.shape[1], num_devices)
        example_input_split = [example_input[:, input_slices[i]:input_slices[i + 1]] for i in range(num_devices)]
        input_data_list = [export_input(example_input_split[i], scalings_activation["in"]["in"]) for i in range(num_devices)]
    else:
        input_slices = get_neuron_slices(weight_dict["att_0"]["q"]["weights"].shape[1], num_devices)
        example_input = jax.random.normal(jax.random.PRNGKey(0), (64, weight_dict["att_0"]["q"]["weights"].shape[1]))
        example_input_split = [example_input[:, input_slices[i]:input_slices[i + 1]] for i in range(num_devices)]
        input_data_list = [export_input(example_input_split[i], scalings_activation["in"]["in"]) for i in range(num_devices)]

    # now generate the code files
    generate_dnni_mixer_config(num_devices=num_devices, message_sizes=message_sizes, num_tokens=len(example_input),
                               path_output=path_output)
    generate_dnni_config(layer_data_list=layer_data_code,  
                         input_data_list=input_data_list, 
                         length_timeseries=len(example_input), 
                         input_feature_size=input_feature_size, 
                         q_feature_size=q_feature_size,
                         o_feature_size=o_feature_size,
                         output_feature_size=output_feature_size,
                         residual_feature_size=residual_feature_size,
                         path_output=path_output)
    generate_model_c(num_residual_layers, message_sizes, attention_only, path_output=path_output)
        

def get_input_ranges(num_devices, weight_shape):
    input_range, _ = get_weight_slices(num_devices, weight_shape)
    return input_range


def export_input(inputs, scaling, num_bits=8):
    x_q = quantization.quantize_forward(inputs, scaling, num_bits)
    c_code = f"static const int8_t input[] = {{ {generate_matrix_code(x_q, True)} }};\n"
    return c_code


def export_exp(x, pos_q=8, num_bits=10):
    maximum = 2**pos_q-1
    values = []
    maximum_value = (2**(num_bits-1) - 1) / (2**pos_q)   # q(num_bits-pos_q).pos_q format
    minimum_value = -(2**(num_bits-1)) / (2**pos_q)
    
    # make the array such that the q format number is equal to the index
    for i in range(0, 2**(num_bits-1)):
        value = math.exp((i / 2**(num_bits-1)) * (maximum_value))
        values.append(value)
    for i in range(0, 2**(num_bits-1)):#
        value = math.exp(((2**(num_bits-1)-i) / 2**(num_bits-1)) * (minimum_value))
        values.append(value)
    
    c_code = f"static const float exp_table[] = {{ {generate_matrix_code(np.array(values), False)} }};\n"
    c_code += f"static const uint8_t pos_q_exp = {pos_q};\n"
    c_code += f"static const int16_t max_val_exp = {(2**(num_bits-1) - 1)};\n"
    c_code += f"static const int16_t min_val_exp = {-(2**(num_bits-1))};\n"
    
    return c_code
        


if __name__ == "__main__":
    pass

