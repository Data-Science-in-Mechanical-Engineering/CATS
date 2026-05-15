#include "message_layer.h"
#include "mixer/mixer.h"
#include "gpi/tools.h"
#include "gpi/platform.h"
#include "gpi/interrupts.h"
#include "gpi/clocks.h"
#include "gpi/olf.h"
#include <stdint.h>
#include <string.h>
#include "internal_messages.h"
#include "message_assignment.h"

#include "dnni_mixer_config.h"

#include <math.h>

#ifndef MIN
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#endif

#define GET_CURRENT_MIXER_IDX(IDX) (IDX - (IDX / MX_GENERATION_SIZE_USED) * MX_GENERATION_SIZE_USED + 1)

static volatile uint8_t read_message_partial(message_assignment_element_t *message_assignment_element, message_t *msg, message_layer_state_t *state)
{
  uint8_t succ = 1;
  while (state->mixer_message_idx < message_assignment_element->mixer_assignment_end - 1 
         && state->current_round == state->mixer_message_idx / MX_GENERATION_SIZE_USED) {
    
    uint8_t temp = read_message_from_mixer(GET_CURRENT_MIXER_IDX(state->mixer_message_idx), ((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE, MX_PAYLOAD_SIZE);
    // if (temp) {
    //   printf("GET_CURRENT_MIXER_IDX(state->mixer_message_idx): %u\n", GET_CURRENT_MIXER_IDX(state->mixer_message_idx));
    //   printf("state->current_message_offset*MX_PAYLOAD_SIZE: %u\n", state->current_message_offset*MX_PAYLOAD_SIZE);
    //   printf("c: %02x\n", *(((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE));
    // }
    succ = succ && temp;
    state->current_message_offset++;
    state->mixer_message_idx++;
  }
  if (state->current_round == state->mixer_message_idx / MX_GENERATION_SIZE_USED) {
    // the last piece is smaller than MX_PAYLOAD_SIZE and thus, we should only read this smaller piece, to not hurt any memory locations
    uint8_t temp = read_message_from_mixer(GET_CURRENT_MIXER_IDX(state->mixer_message_idx), 
                                           ((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE, 
                                           message_assignment_element->size_end);
    succ = succ && temp;
    state->current_message_offset = 0;
    state->current_message_idx++;
    state->mixer_message_idx++;
  }
  return succ;
}


static void write_message_partial(message_assignment_element_t *message_assignment_element, message_t *msg, message_layer_state_t *state)
{
  printf("%u\n", state->mixer_message_idx);
  while (state->mixer_message_idx < message_assignment_element->mixer_assignment_end - 1 
         && state->current_round == state->mixer_message_idx / MX_GENERATION_SIZE_USED) {
    // printf("GET_CURRENT_MIXER_IDX(state->mixer_message_idx): %u\n", GET_CURRENT_MIXER_IDX(state->mixer_message_idx));
    // printf("c: %02x\n", *(((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE));
    mixer_write(GET_CURRENT_MIXER_IDX(state->mixer_message_idx), ((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE, MX_PAYLOAD_SIZE);
    state->current_message_offset++;
    state->mixer_message_idx++;
  }

  if (state->current_round == state->mixer_message_idx / MX_GENERATION_SIZE_USED) {
    // printf("GET_CURRENT_MIXER_IDX(state->mixer_message_idx): %u\n", GET_CURRENT_MIXER_IDX(state->mixer_message_idx));
    // printf("state->current_message_offset: %u\n", state->current_message_offset);
    // the last piece is smaller than MX_PAYLOAD_SIZE and thus, we should only read this smaller piece, to not hurt any memory locations
    mixer_write(GET_CURRENT_MIXER_IDX(state->mixer_message_idx), 
                ((uint8_t *) msg) + state->current_message_offset*MX_PAYLOAD_SIZE, 
                message_assignment_element->size_end);
    state->current_message_offset = 0;
    state->current_message_idx++;
    state->mixer_message_idx++;
  }
  // printf("------------------\n");
}


/**
 * @brief Initializes a message assignment list.
 *
 * This function initializes the given message_assignment list. It calculates the mixer assignment start and end indices
 * for each message_assignment element based on the size of the message and the maximum payload size of the mixer.
 * If the last mixer assignment end index exceeds the maximum generation size, an error is indicated by blinking an LED.
 *
 * @param message_assignment List of message_assignment_t to be initialized.
 */
void init_message_assignment(message_assignment_t *message_assignment)
{
  uint32_t idx = 0;
  for (uint32_t i=0; i<message_assignment->length; i++) {
    message_assignment->assignments[i].mixer_assignment_start = idx;
    uint32_t num_mixer_messages = ceil((float) (message_assignment->assignments[i].size) / MX_PAYLOAD_SIZE - 1e-7);

    message_assignment->assignments[i].size_end = message_assignment->assignments[i].size - (num_mixer_messages-1)*MX_PAYLOAD_SIZE;
    idx += num_mixer_messages;  // the number of mixer messages is the rounded up number
    message_assignment->assignments[i].mixer_assignment_end = idx;
  }

  message_assignment->num_mixer_rounds = idx / MX_GENERATION_SIZE_USED;
  if (idx % MX_GENERATION_SIZE_USED != 0) {
    message_assignment->num_mixer_rounds++;
  }

  if (message_assignment->assignments[message_assignment->length-1].mixer_assignment_end > MX_GENERATION_SIZE_USED * MX_NUMBER_ROUNDS) {
    // something is wrong, blink slowly
    while(1) {
      NRF_P0->OUTCLR = BV(25);
      gpi_milli_sleep(2000);
      NRF_P0->OUTSET = BV(25);
      gpi_milli_sleep(2000);
    }
  }
}


uint8_t get_assignment_idx(const message_assignment_t *message_assignment, uint8_t id)
{
  uint8_t idx = 0;
  printf("message_assignment->length: %u\n", message_assignment->length);
  while (idx < message_assignment->length) {
    if (id == message_assignment->assignments[idx].id) {
      return idx;
      break;
    }
    idx += 1;
  }
  return 255;
}



uint8_t read_message_from_mixer(uint8_t mixer_idx, uint8_t *msg_p, uint16_t size)
{
  void *p = mixer_read(mixer_idx);

  // check if message was received. Return 0 if not.
  if (NULL == p) {
    return 0;
  } else if ((void*)-1 == p) {
    return 0;
  } 
  memcpy((void *) msg_p, p, size);
  return 1;
}

void message_layer_reset_state(message_layer_state_t *state)
{
  state->current_round = 0;
  state->mixer_message_idx = 0;
  state->current_message_idx = 0;
  state->current_message_offset = 0;
}

/**
 * Retrieves a message from the message layer.
 *
 * @param message_assignment The list of message assignments.
 * @param messages The buffer to store the retrieved message.
 * @param succ The buffer to store the success of the message retrieval (succ[i] == 1, when messages[i] was successfully received).
 * @param state The state of the message layer.
 */
void message_layer_mixer_round_finished_callback(const message_assignment_t *message_assignment, message_t *messages, uint8_t *succ, message_layer_state_t *state)
{
  while (state->current_round == state->mixer_message_idx / MX_GENERATION_SIZE_USED && state->current_message_idx < message_assignment->length) {
    uint16_t current_message_idx = state->current_message_idx;  // might change during the call of read_message_partial
    uint8_t succ_red = read_message_partial(&message_assignment->assignments[state->current_message_idx], &messages[state->current_message_idx], state);
    succ[current_message_idx] = succ[current_message_idx] && succ_red;
  }
  state->current_round++;
}

/**
 * Retrieves a message from the message layer.
 *
 * @param message_assignment The list of message assignments.
 * @param messages The buffer to store the retrieved message.
 * @param succ The buffer to store the success of the message retrieval (succ[i] == 1, when messages[i] was successfully received).
 * @param state The state of the message layer.
 */
void message_layer_mixer_round_starting_callback(const message_assignment_t *message_assignment, message_t **messages, uint8_t *valid, message_layer_state_t *state)
{
  message_layer_state_t tx_state;
  memcpy(&tx_state, state, sizeof(message_layer_state_t));
  int num_weak_slots = MX_GENERATION_SIZE - 1 + tx_state.mixer_message_idx;
  // printf("tx_state.mixer_message_idx: %u\n", tx_state.mixer_message_idx);
  while (tx_state.current_round == tx_state.mixer_message_idx / MX_GENERATION_SIZE_USED && tx_state.current_message_idx < message_assignment->length) {
    // printf("idx: %u\n", tx_state.current_message_idx);
    // printf("valid: %u\n", valid[tx_state.current_message_idx]);
    if (valid[tx_state.current_message_idx]) {
      // write messages to mixer, write_messages_partial will update the state
      write_message_partial(&message_assignment->assignments[tx_state.current_message_idx], messages[tx_state.current_message_idx], &tx_state);
    } else {
      // printf("l");
        // just forward state
        tx_state.mixer_message_idx += message_assignment->assignments[tx_state.current_message_idx].mixer_assignment_end - message_assignment->assignments[tx_state.current_message_idx].mixer_assignment_start;
        // printf("%u\n", tx_state.mixer_message_idx);
        // printf("dddd%u/%u\n", tx_state.current_message_idx, message_assignment->length);
        tx_state.current_message_offset = 0;
        tx_state.current_message_idx++;
    }
  }
  num_weak_slots -= (int) tx_state.mixer_message_idx;
  // write weak zeros to non-used mixer messagages
  for (uint16_t i = 1; i <= num_weak_slots; i++) {
    mixer_write(MX_GENERATION_SIZE - i, NULL, 0);
  }
  
}



/**
//  * Retrieves a message from the message layer.
//  *
//  * @param message_assignment The list of message assignments.
//  * @param id The index of the message to retrieve.
//  * @param msg The buffer to store the retrieved message.
//  * @return 0 if error occured (message loss).
//  */
// uint8_t message_layer_get_message(const message_assignment_t *message_assignment, uint8_t idx, message_t *msg)
// {
//   if (idx == 255) {
//     return 0;
//   }
//   for (uint32_t i = 0; 
//       i < message_assignment->assignments[idx].mixer_assignment_end - message_assignment->assignments[idx].mixer_assignment_start - 1; 
//       i++) {
//       uint8_t succ = read_message_from_mixer(i + message_assignment->assignments[idx].mixer_assignment_start, ((uint8_t *) msg) + i*MX_PAYLOAD_SIZE, MX_PAYLOAD_SIZE);
//     if (!succ) {
//       return 0;
//     }
//   }
//   // the last piece is smaller than MX_PAYLOAD_SIZE and thus, we should only read this smaller piece, to not hurt any memory locations
//   uint8_t succ = read_message_from_mixer(message_assignment->assignments[idx].mixer_assignment_end - 1, 
//                           ((uint8_t *) msg) + (message_assignment->assignments[idx].size-message_assignment->assignments[idx].size_end), 
//                           message_assignment->assignments[idx].size_end);
//   if (!succ) {
//       return 0;
//   }
//   return 1;
// }
