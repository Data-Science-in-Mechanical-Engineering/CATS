#include <stdint.h>

#include "multi_head_attention_block.h"
#include "fc_layer.h"
#include "exp.h"
#include "math.h"
#include "layernorm.h"
#include "cp_os.h"

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nn_types.h"

extern uint16_t __attribute__((section(".data")))	TOS_NODE_ID;

#define CAUSAL_ATTENTION 0


static void print_matrix(int8_t *mat, cmsis_nn_dims *dims) {
    for (uint32_t i = 0; i < dims->n; i++) {
        for (uint32_t j = 0; j < dims->c; j++) {
            printf("%4d ", mat[i*dims->c+j]);
        }
        printf("\n");
    }
}


static void print_matrix_32(int32_t *mat, cmsis_nn_dims *dims) {
    for (uint32_t i = 0; i < dims->n; i++) {
        for (uint32_t j = 0; j < dims->c; j++) {
            printf("%4ld ", mat[i*dims->c+j]);
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

void multi_head_attention_block(const multi_head_attention_block_config_t *config, const int8_t *input, int8_t *output)
{
    /****************
     * Communication
     ****************/
    printf("Starting Attention----------\n");
    uint32_t computing_time = 0;
    all_gather(input, config->input_range, config->input_pruning, config->input_message_assignment, config->input_dims, config->input_buffer, 0, &computing_time);
    
    for (int i = 0; i < 64; i++) {
        printf("%d ", config->input_buffer[i]);
    }
    printf("\n");
    
    /****************
     * Layernorm (in-place)
     ****************/
    STOPWATCH_START();
    layernorm(config->layernorm_config, config->input_buffer, config->input_buffer, config->input_dims->n, config->input_dims->c);
    //for (int i = 0; i < 16; i++) {
        //printf("%d ", config->residual_buffer[i]);
    //}
    //printf("\n");

    /****************
     * Attention heads
     ****************/
    cmsis_nn_dims head_result_dims = {.n=config->input_dims->n, .h=1, .w=1, .c=config->input_weight_dims->c};
    cmsis_nn_dims k_t_dims = {.n=config->input_weight_dims->c, .h=1, .w=1, .c=config->input_dims->n};

    cmsis_nn_dims head_concat_dims = {.n=config->input_dims->n, .h=1, .w=1, .c=config->input_weight_dims->c*config->num_heads};

    for (uint32_t i = 0; i < config->num_heads; i++) {
        // calculate q 
        arm_cmsis_nn_status s = arm_fully_connected_s8_f32_s8(config->ctx,
                                                            config->fc_params,
                                                            config->q_quant_params_list[i],
                                                            config->input_dims,
                                                            config->input_buffer,
                                                            config->input_weight_dims,
                                                            config->q_weight_list[i],
                                                            config->input_bias_dims,
                                                            config->q_bias_list[i],
                                                            &head_result_dims,
                                                            config->q_buffer);

        // printf("q %ld\n", (int) (1000 * config->q_quant_params_list[i]));
        // if (i == 0) {
            // printf("q:\n");
            // print_matrix(config->q_buffer, &head_result_dims);
        // }
        
        // calulate k
        s = arm_fully_connected_s8_f32_s8(config->ctx,
                                    config->fc_params,
                                    config->k_quant_params_list[i],
                                    config->input_dims,
                                    config->input_buffer,
                                    config->input_weight_dims,
                                    config->k_weight_list[i],
                                    config->input_bias_dims,
                                    config->k_bias_list[i],
                                    &head_result_dims,
                                    config->k_buffer);
        
        
        // calculate v
        s = arm_fully_connected_s8_f32_s8(config->ctx,
                                    config->fc_params,
                                    config->v_quant_params_list[i],
                                    config->input_dims,
                                    config->input_buffer,
                                    config->input_weight_dims,
                                    config->v_weight_list[i],
                                    config->input_bias_dims,
                                    config->v_bias_list[i],
                                    &head_result_dims,
                                    config->v_buffer);

        // calculate q*kT
        int8_t *q_row_pointer = config->q_buffer;
        const int32_t *kernel_sum = (const int32_t *) config->ctx->buf;

        for (uint32_t j = 0; j < head_result_dims.n; j++) {
            // go through the rows of q*kT, calculate softmax and quantize
            // note that  for arm_nn_vec_mat_mult_t_s8_in_s32_out the rhs has to be transposed in memory. As we need to use kT, we can use the same buffer as for k.
            s = arm_nn_vec_mat_mult_t_s8_in_f32_out(q_row_pointer,  // lhs
                                                    config->k_buffer,  // rhs
                                                    kernel_sum, //kernel_sum
                                                    NULL,                   // no bias
                                                    config->qk_buffer, // dst
                                                     0,  // lhs_offset
                                                    config->att_quant_params,  // dst_multiplier,
                                                    head_result_dims.c,  // rhs_cols,  (note that k.T)
                                                    #if CAUSAL_ATTENTION
                                                    j+1,  // head_result_dims.n for non-causal attention,  // rhs_rows,
                                                    #else
                                                    head_result_dims.n,  // rhs_rows,
                                                    #endif
                                                    1L,  //address_offset,
                                                    0  // rhs_offset
                                                    );

            
            // softmax
            float sum_exp = 0;
            #if CAUSAL_ATTENTION
            uint32_t end = j+1;
            #else
            uint32_t end = head_result_dims.n;
            #endif
            for (uint32_t k = 0; k < end; k++) {
                // printf("%ld, ", (int) (1000* config->qk_buffer[k]));
                config->qk_buffer[k] = expf(config->qk_buffer[k] * config->inv_sqrt);
                sum_exp += config->qk_buffer[k];
            }
            // printf("\n");
            for (uint32_t k = 0; k < end; k++) {
                config->qk_buffer[k] /= sum_exp;
            }

            q_row_pointer += head_result_dims.c;
            int8_t *o_pointer = config->o_buffer + i*head_result_dims.c + j*head_result_dims.c*config->num_heads;
            // calculate row of softmax(qk)v
            for (uint32_t k = 0; k < head_result_dims.c; k++) {
                float temp = 0;
                int8_t *v_pointer = config->v_buffer+k;
                // (only till j+1, because of causal attention)
                for (uint32_t l = 0; l < end; l++) {
                    temp += ((config->qk_buffer[l]) * (config->v_scaling_list[i] *  v_pointer[0]));
                    v_pointer += head_result_dims.c;
                }
                temp = roundf(config->softmax_times_v_scaling_list[i] * temp);
                
                *o_pointer = (int8_t) (temp>127 ? 127 : (temp < -128 ? -128 : temp));
                o_pointer++;
            }
        }
    }
    STOPWATCH_END(computing_time);

    printf("0:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", config->o_buffer[i]);
    }
    printf("\n");
    

    /****************
     * Calculate y_o*W_o
     ****************/
    fc_layer(config->o_layer_config, config->o_buffer, output, 0, &computing_time);

    for (uint32_t j = 0; j < 10; j++) {
        printf("%4d ", output[j]);
    }
    printf("\n");

    /****************
     * Output residual
     ****************/

    // add the input (after layernorm) to the output. We do not use arm_elementwise_add_s8, becasue it does not perform a speedup
    // using SIMD.
    STOPWATCH_START();
    cmsis_nn_dims output_dims = {.n=config->input_dims->n, .h=1, .w=1, .c=config->o_layer_config->weight_dims->c};
    int8_t *input_ptr = input;
    int8_t *output_ptr = output;
    uint32_t device_idx = TOS_NODE_ID - 1;
    for (uint32_t i = 0; i < output_dims.n; i++) {
        // input_ptr = input + i*config->input_dims->c + config->input_range[device_idx];  // config->residual_buffer + i*config->input_dims->c + config->input_range[device_idx];
        for (uint32_t j = 0; j < output_dims.c; j++) {
            int32_t temp = (int32_t) (roundf((*output_ptr* config->o_layer_output_scaling + *input_ptr * config->input_scaling) * config->sum_scaling));
            *output_ptr = (int8_t)(temp > INT8_MAX ? INT8_MAX : (temp < INT8_MIN ? INT8_MIN : temp));
            input_ptr++;
            output_ptr++;
        }
    }
    STOPWATCH_END(computing_time);
    printf("Attention block computing duration: %ld us\n", computing_time);

    for (uint32_t j = 0; j < 10; j++) {
        printf("%4d ", output[j]);
    }
    printf("\n");
}