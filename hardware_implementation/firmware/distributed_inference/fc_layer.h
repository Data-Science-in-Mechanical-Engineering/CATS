#ifndef FC_LAYER_H
#define FC_LAYER_H


#include <stdint.h>

#include "internal_messages.h"
#include "message_assignment.h"

#include "arm_nn_types.h"
#include "arm_nn_math_types.h"


typedef struct fc_layer_config_t_tag
{
    int8_t *weight;
    int32_t *bias;
    cmsis_nn_dims *weight_dims;
    cmsis_nn_dims *bias_dims;
    cmsis_nn_dims *input_dims;
    cmsis_nn_context *ctx;
    float quant_params;
    cmsis_nn_fc_params *fc_params;
    uint32_t *input_range;
    int8_t *input_buffer;
    message_assignment_t *message_assignment;
    uint8_t *pruning;
} fc_layer_config_t;


/**
 * @brief fully connected layer
 *
 * This function implements a fully connected layer. The operations are:
 *  1) communication round: communicate input to all devices and receive their inputs
 *  2) concat the received inputs
 *  3) Run the device's part of the fully connected layer, leading to output.
 *
 * @param input input associated with the device. has to be large enough to hold the entire input (so own input and the ones received from other devices)
 * @param output output associated with the device
 * @param is_first_layer indicates whether this is the first layer (used for synchronization)
 * @param computing_time pointer to store the computing time in microseconds
 * ...
 *
 * @return [Description of the return value, including its type and meaning]
 */
void fc_layer(const fc_layer_config_t *config, const int8_t *input, int8_t *output, uint8_t is_first_layer, uint32_t *computing_time);


#endif /* FC_LAYER_H */
