#include <math.h>

#include "residual_block.h"
#include "fc_layer.h"
#include "layernorm.h"
#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "cp_os.h"

extern uint16_t __attribute__((section(".data")))	TOS_NODE_ID;


static void print_matrix(int8_t *mat, cmsis_nn_dims *dims) {
    for (uint32_t i = 0; i < dims->n; i++) {
        for (uint32_t j = 0; j < dims->c; j++) {
            printf("%4d ", mat[i*dims->c+j]);
        }
        printf("\n");
    }
}

static void print_matrix_f32(float *mat, cmsis_nn_dims *dims) {
    for (uint32_t i = 0; i < dims->n; i++) {
        for (uint32_t j = 0; j < dims->c; j++) {
            printf("%4ld ", (int) (1000* mat[i*dims->c+j]));
        }
        printf("\n");
    }
}

void residual_block(const residual_block_config_t *config, const int8_t *input, int8_t *output, uint8_t is_first_layer)
{
    const int8_t *in = input;

    // use offset such that at the end of the loop, the result is in config->buffer, the result of the residual is then saved in output
    uint8_t offset = config->num_layers % 2;
    int8_t *out;
    cmsis_nn_dims output_dims = {.n=config->layers[config->num_layers-1]->input_dims->n, .h=1, .w=1, .c=config->layers[config->num_layers-1]->weight_dims->c};
    uint32_t device_idx = TOS_NODE_ID - 1;

    uint32_t computing_time = 0;

    for (uint8_t i = 0; i < config->num_layers; i++)
    {
        printf("layer %d\n", i);
        // if ((i+offset)%2 == 0) {
        //     out = config->buffer;
        // } else {
        //     out = (int8_t *) output;
        // }
        out = output;
        
        if (i != 0) {
            fc_layer(config->layers[i], in, out, i == 0 && is_first_layer, &computing_time);
            printf("i: %u\n", i);
            for (uint32_t j = 0; j < 10; j++) {
            printf("%4d ", out[j]);
             }
            printf("\n");
        } else {
            all_gather(in, 
                    config->layers[0]->input_range, 
                    config->layers[0]->pruning, 
                    config->layers[0]->message_assignment, 
                    config->layers[0]->input_dims, 
                    config->layers[0]->input_buffer, 
                    is_first_layer, &computing_time);

            // print_matrix(config->layers[0]->input_buffer, config->layers[0]->input_dims);
            STOPWATCH_START();        
            // for the first layer, we need to also calculate the residual. We know that the input is saved in config->buffer
            if (i == 0)  {
                if (config->residual_layer != NULL) {
                    cmsis_nn_dims output_dims = {.n=config->residual_layer->input_dims->n, .h=1, .w=1, .c=config->residual_layer->weight_dims->c};
                    arm_cmsis_nn_status s = arm_fully_connected_s8_f32_s8(config->residual_layer->ctx,
                                                                        config->residual_layer->fc_params,
                                                                        config->residual_layer->quant_params,
                                                                        config->residual_layer->input_dims,
                                                                        config->layers[i]->input_buffer,
                                                                        config->residual_layer->weight_dims,
                                                                        config->residual_layer->weight,
                                                                        config->residual_layer->bias_dims,
                                                                        config->residual_layer->bias,
                                                                        &output_dims,
                                                                        config->residual_buffer);
                } else {
                    // copy the input to the output (note that the values we care about lie in the range of the input_range)
                    int8_t *residual_buffer_ptr = config->residual_buffer;
                    int8_t *input_buffer_ptr = config->layers[i]->input_buffer + config->layers[0]->input_range[device_idx];
                    for (uint32_t j = 0; j < output_dims.n; j++) {
                        memcpy(residual_buffer_ptr, input_buffer_ptr, sizeof(int8_t) * output_dims.c);
                        residual_buffer_ptr += output_dims.c;
                        input_buffer_ptr += config->layers[0]->input_dims->c;
                    }
                }
            }

            if (config->layernorm_config != NULL) {
                layernorm(config->layernorm_config, config->layers[0]->input_buffer, config->layers[0]->input_buffer, config->layers[0]->input_dims->n, config->layers[0]->input_dims->c);
            }
            for (uint32_t j = 0; j < 10; j++) {
                printf("%4d ", config->layers[0]->input_buffer[j]);
            }
            printf("\n");
            cmsis_nn_dims output_dims = {.n=config->layers[0]->input_dims->n, .h=1, .w=1, .c=config->layers[0]->weight_dims->c};
            arm_cmsis_nn_status s = arm_fully_connected_s8_f32_s8(config->layers[0]->ctx,
                                                                config->layers[0]->fc_params,
                                                                config->layers[0]->quant_params,
                                                                config->layers[0]->input_dims,
                                                                config->layers[0]->input_buffer,
                                                                config->layers[0]->weight_dims,
                                                                config->layers[0]->weight,
                                                                config->layers[0]->bias_dims,
                                                                config->layers[0]->bias,
                                                                &output_dims,
                                                                out);
            //printf("0:\n"); 
            // for (uint32_t j = 0; j < 10; j++) {
                // printf("%4d ", out[j]);
            // }
            // printf("\n");
            STOPWATCH_END(computing_time);
        }
        
        in = out;
    }
    // for (uint32_t j = 0; j < 10; j++) {
    //     printf("%4d ", config->residual_buffer[j]);
    // }
    // printf("\n");
    // add the residual to the output. We do not use arm_elementwise_add_s8, because it does not perform a speedup
    // using SIMD.  config->layers[config->num_layers-1]->input_dims->c
    STOPWATCH_START();
    for (uint32_t i = 0; i < config->layers[config->num_layers-1]->input_dims->n * config->layers[config->num_layers-1]->weight_dims->c; i++) {
        int32_t temp = (int32_t) (roundf((out[i] * config->input_scaling + config->residual_buffer[i] * config->residual_scaling) * config->sum_scaling));
        output[i] = (int8_t)(temp > INT8_MAX ? INT8_MAX : (temp < INT8_MIN ? INT8_MIN : temp));
    }
    STOPWATCH_END(computing_time);
    printf("Residual block computing duration: %ld us\n", computing_time);

    for (uint32_t j = 0; j < 10; j++) {
        printf("%4d ", output[j]);
    }
    printf("\n");
}