#ifndef MESSAGE_LAYER_H
#define MESSAGE_LAYER_H

#include "mixer_config.h"
#include "internal_messages.h"
#include "dnni_mixer_config.h"

#define MX_GENERATION_SIZE_USED (MX_GENERATION_SIZE - 1)   // -1 because of initiator

typedef struct message_layer_state_t_tag
{
    uint8_t current_round;  // current mixer round
    uint16_t mixer_message_idx;  // index of the current mixer message
    uint16_t current_message_idx;  // index of the current message
    uint16_t current_message_offset;  // offset of the message in mixer messages (current_message_offset*MX_GENERATION_SIZE bytes where already received)
} message_layer_state_t;

void init_message_assignment(message_assignment_t *message_assignment);

uint8_t read_message_from_mixer(uint8_t mixer_idx, uint8_t *msg_p, uint16_t size);

uint8_t get_assignment_idx(const message_assignment_t *message_assignment, uint8_t id);

void message_layer_mixer_round_finished_callback(const message_assignment_t *message_assignment, message_t *messages, uint8_t *succ, message_layer_state_t *state);

void message_layer_mixer_round_starting_callback(const message_assignment_t *message_assignment, message_t **messages, uint8_t *valid, message_layer_state_t *state);


#endif