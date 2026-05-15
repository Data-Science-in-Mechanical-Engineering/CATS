#ifndef INC_DNNI_MIXER_CONFIG_H
#define INC_DNNI_MIXER_CONFIG_H

#include <stdint.h>
#include "message_assignment.h"
#include "gpi/tools.h"

static const uint8_t nodes[] = {  1,  2,  3,  4, };

#define NUM_PLANTS (NUM_ELEMENTS(nodes))
#define MAXIMUM_NUMBER_MESSAGES (4)
#define SIZE_ACTIVATIONS 832

static message_assignment_element_t message_assignment_elements_attention_input_attention_block_0[] = {{.id=1, .size=836}, {.id=2, .size=836}, {.id=3, .size=836}, {.id=4, .size=836}, };
static message_assignment_element_t message_assignment_elements_o_layer_attention_block_0[] = {{.id=1, .size=836}, {.id=2, .size=836}, {.id=3, .size=836}, {.id=4, .size=836}, };
static message_assignment_element_t message_assignment_elements_residual_block_0_0[] = {{.id=1, .size=836}, {.id=2, .size=836}, {.id=3, .size=836}, {.id=4, .size=836}, };
static message_assignment_element_t message_assignment_elements_residual_block_0_1[] = {{.id=1, .size=836}, {.id=2, .size=836}, {.id=3, .size=836}, {.id=4, .size=836}, };


static message_assignment_t message_assignment_attention_input_attention_block_0 = {.id=1, .num_mixer_rounds=1, .length=4, .assignments=message_assignment_elements_attention_input_attention_block_0};
static message_assignment_t message_assignment_o_layer_attention_block_0 = {.id=2, .num_mixer_rounds=1, .length=4, .assignments=message_assignment_elements_o_layer_attention_block_0};
static message_assignment_t message_assignment_residual_block_0_0 = {.id=3, .num_mixer_rounds=1, .length=4, .assignments=message_assignment_elements_residual_block_0_0};
static message_assignment_t message_assignment_residual_block_0_1 = {.id=4, .num_mixer_rounds=1, .length=4, .assignments=message_assignment_elements_residual_block_0_1};



#define MX_NUMBER_ROUNDS 1
#define MX_PAYLOAD_SIZE 150
#define MX_ROUND_LENGTH 275
#define MX_SLOT_LENGTH GPI_TICK_US_TO_HYBRID2(947)
#define ROUND_LENGTH_MS            ((MX_ROUND_LENGTH*MX_SLOT_LENGTH / (GPI_HYBRID_CLOCK_RATE / 1000000)) / 1000 + 1000)
#define MX_GENERATION_SIZE 25

// ------------- Aggregatefield-------------

#define CONTROL_MSGS_M_C					0
#define PRIO_WIDTH							8
#define NODE_ID_WIDTH						5
#define AGGREGATE_CONTAINS_ALL_PRIORITIES	0

ASSERT_CT_STATIC(NUM_PLANTS <= ((1 << NODE_ID_WIDTH) - 1), NODE_ID_WIDTH_cannot_support_all_nodes);
ASSERT_CT_STATIC(NODE_ID_WIDTH <= 8, NODE_ID_WIDTH_greater_8_is_not_implemented);

#define AGGREGATE_SIZE_M_C_PRIORITIES	((NUM_PLANTS + CONTROL_MSGS_M_C * PRIO_WIDTH + CONTROL_MSGS_M_C * NODE_ID_WIDTH + 7) / 8)
#define AGGREGATE_SIZE_ALL_PRIORITIES	((NUM_PLANTS * PRIO_WIDTH + 7) / 8)


#if AGGREGATE_CONTAINS_ALL_PRIORITIES == 1
	#error "NOT IMPLEMENTED"
	#define AGGREGATE_SIZE	AGGREGATE_SIZE_ALL_PRIORITIES
#else
	#define AGGREGATE_SIZE	AGGREGATE_SIZE_M_C_PRIORITIES
#endif

typedef uint8_t priority_t;
ASSERT_CT_STATIC(PRIO_WIDTH <= (sizeof(priority_t) * 8), prob_info_t_needs_to_implement_support_for_priority_widths_greater_8_bit);


#endif /* INC_DNNI_MIXER_CONFIG_H */