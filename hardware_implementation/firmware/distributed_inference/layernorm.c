#include <stdint.h>

#include "layernorm.h"
#include "math.h"

void layernorm(const layernorm_config_t *config, const int8_t *input, int8_t *output, uint32_t num_rows, uint32_t num_cols)
{
    for (uint32_t i = 0; i < num_rows; ++i)
    {
        float mean = 0.0f;
        for (uint32_t j = 0; j < num_cols; ++j)
        {
            float value = input[i * num_cols + j] * config->scaling_in;
            mean += value;
        }
        mean /= num_cols;

        float variance = 0;
        for (uint32_t j = 0; j < num_cols; ++j)
        {
            float value = input[i * num_cols + j] * config->scaling_in - mean;
            variance += value * value;
        }
        variance /= num_cols-1;

        float stddev = sqrtf(variance + 1e-7f); // Adding a small epsilon to avoid division by zero
        // printf("mean: %d, stddev: %d\n", (int) (mean*1000), (int) (stddev*1000));

        for (uint32_t j = 0; j < num_cols; ++j)
        {
            float value = input[i * num_cols + j] * config->scaling_in;
            value = (((value - mean) / stddev)) +  config->bias[j];
            value = roundf(value * config->multiplier[j]);
            output[i * num_cols + j] = (int8_t) (value > INT8_MAX ? INT8_MAX : (value < INT8_MIN ? INT8_MIN : value));
        }
        // printf("\n");
    }
}
