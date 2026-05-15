#ifndef RESIDUAL_BLOCK_H
#define RESIDUAL_BLOCK_H

#include <stdint.h>

#include "fc_layer.h"
#include "layernorm.h"


typedef struct residual_block_config_t_tag
{
    fc_layer_config_t **layers;
    uint8_t num_layers;
    fc_layer_config_t *residual_layer;
    float input_scaling;
    float residual_scaling;
    float sum_scaling;
    // int8_t *buffer;
    int8_t *residual_buffer;
    layernorm_config_t *layernorm_config;

} residual_block_config_t;


/**
 */
void residual_block(const residual_block_config_t *config, const int8_t *input, int8_t *output, uint8_t is_first_layer);


#endif /* RESIDUAL_BLOCK_H */

