#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <stdint.h>

typedef struct layernorm_config_t_tag
{
    float *multiplier;  // contains the multiplier of layernorm * scaling_out
    float *bias;
    float scaling_in;
} layernorm_config_t;

void layernorm(const layernorm_config_t *config, const int8_t *input, int8_t *output, uint32_t num_rows, uint32_t num_cols);

#endif  // LAYERNORM_H