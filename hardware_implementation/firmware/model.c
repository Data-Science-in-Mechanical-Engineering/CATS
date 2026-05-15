#include "dnni_config.h"
#include "residual_block.h"
#include "multi_head_attention_block.h"
#include "message_layer.h"
#include "arm_nn_types.h"
#include "model.h"

int8_t output1[OUTPUT_BUFFER_SIZE];
int8_t output2[OUTPUT_BUFFER_SIZE];

static void print_matrix(int8_t *mat, cmsis_nn_dims *dims) {
    for (uint32_t i = 0; i < dims->n; i++) {
        for (uint32_t j = 0; j < dims->c; j++) {
            printf("%4d ", mat[i*dims->c+j]);
        }
        printf("\n");
    }
}


void init_model()
{
    init_message_assignment(&message_assignment_attention_input_attention_block_0);
init_message_assignment(&message_assignment_o_layer_attention_block_0);
init_message_assignment(&message_assignment_residual_block_0_0);
init_message_assignment(&message_assignment_residual_block_0_1);

}

void run_model()
{
    // cmsis_nn_dims output_dims = {.n=residual_block_config_input_residual.layers[0]->input_dims->n, .h=1, .w=1, .c=residual_block_config_input_residual.layers[residual_block_config_input_residual.num_layers-1]->weight_dims->c};
    residual_block(&residual_block_config_residual_block_0, output2, output1, 1);
multi_head_attention_block(&mha_config_attention_block_0, output1, output2);


    printf("\n\n");
}