#include <stdint.h>
#include <stdio.h>

#include "fc_layer.h"
#include "cp_os.h"
#include "arm_nnfunctions.h"


void fc_layer(const fc_layer_config_t *config, const int8_t *input, int8_t *output, uint8_t is_first_layer, uint32_t *computing_time)
{
    /****************
     * Communication
     ****************/
    all_gather(input, config->input_range, config->pruning, config->message_assignment, config->input_dims, config->input_buffer, is_first_layer, computing_time);
    
    // /****************
    //  * Fully Connected Layer
    //  ****************/
    STOPWATCH_START();
    cmsis_nn_dims output_dims = {.n=config->input_dims->n, .h=1, .w=1, .c=config->weight_dims->c};
    // printf("Input Dims: n=%d, h=%d, w=%d, c=%d\n", config->input_dims->n, config->input_dims->h, config->input_dims->w, config->input_dims->c);
    // printf("Weight Dims: n=%d, h=%d, w=%d, c=%d\n", config->weight_dims->n, config->weight_dims->h, config->weight_dims->w, config->weight_dims->c);
    // printf("Bias Dims: n=%d, h=%d, w=%d, c=%d\n", config->bias_dims->n, config->bias_dims->h, config->bias_dims->w, config->bias_dims->c);
    // printf("Output Dims: n=%d, h=%d, w=%d, c=%d\n", output_dims.n, output_dims.h, output_dims.w, output_dims.c);
    // run the fully connected layer using CMSIS NN
    arm_cmsis_nn_status s = arm_fully_connected_s8_f32_s8(config->ctx,
                                           config->fc_params,
                                           config->quant_params,
                                           config->input_dims,
                                           config->input_buffer,
                                           config->weight_dims,
                                           config->weight,
                                           config->bias_dims,
                                           config->bias,
                                           &output_dims,
                                           output); 
    STOPWATCH_END(*computing_time);
}