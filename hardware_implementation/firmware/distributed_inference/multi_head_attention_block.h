#ifndef ATTENTION_BLOCK_H
#define ATTENTION_BLOCK_H

#include <stdint.h>
#include "fc_layer.h"
#include "layernorm.h"

typedef struct multi_head_attention_block_config_t_tag
{
    layernorm_config_t *layernorm_config;
    int8_t num_heads;
    int8_t **q_weight_list;
    int32_t **q_bias_list;
    float *q_quant_params_list;
    int8_t **k_weight_list;
    int32_t **k_bias_list;
    float *k_quant_params_list;
    int8_t **v_weight_list;
    int32_t **v_bias_list;
    float *v_quant_params_list;
    float *v_scaling_list;
    float *softmax_times_v_scaling_list;
    cmsis_nn_dims *input_weight_dims;  // dims of q, k and v in one head
    cmsis_nn_dims *input_bias_dims;
    float att_quant_params;
    float inv_sqrt;
    cmsis_nn_dims *input_dims;
    cmsis_nn_context *ctx;
    cmsis_nn_fc_params *fc_params;
    uint32_t *input_range;
    message_assignment_t *input_message_assignment;
    uint8_t *input_pruning;
    int8_t *input_buffer;  // *residual_buffer;
    int8_t *q_buffer;
    int8_t *k_buffer;
    int8_t *v_buffer;
    int8_t *o_buffer;
    float *qk_buffer;

    fc_layer_config_t *o_layer_config;
    float input_scaling;
    float o_layer_output_scaling;
    float sum_scaling;

} multi_head_attention_block_config_t;


void multi_head_attention_block(const multi_head_attention_block_config_t *config, const int8_t *input, int8_t *output);


#endif // ATTENTION_BLOCK_H