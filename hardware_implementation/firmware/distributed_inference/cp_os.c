#include "cp_os.h"
#include "spi.h" 
#include "mixer/mixer.h"
#include "mixer/mixer_internal.h"
#include "mixer_aggregate.h"
#include "gpi/trace.h"
#include "gpi/tools.h"
#include "gpi/platform.h"
#include "gpi/interrupts.h"
#include "gpi/clocks.h"
#include "gpi/olf.h"
#include "message_layer.h"
#include "arm_nn_types.h"
#include <string.h>
#include "mixer_config.h"


#define PRINT_HEADER()	printf("# ID:%u ", TOS_NODE_ID)

static uint8_t node_id;
static uint8_t fl_node_idx;
static uint32_t round_nr;
static uint32_t real_round_nr;
static uint8_t agg_input[AGGREGATE_SIZE];
static uint8_t agg_output[AGGREGATE_SIZE];
static Gpi_Hybrid_Tick t_ref;
extern uint16_t __attribute__((section(".data"))) TOS_NODE_ID;

static message_t dummy_message;

static init_message_t init_pkt = {.round = 0};
static message_t *tx_messages[MAXIMUM_NUMBER_MESSAGES];
static uint8_t valid[MAXIMUM_NUMBER_MESSAGES] = {0};

static uint16_t initiator_message_not_received_counter = 0;

static uint8_t synchronized = 0;

static message_t rx_messages[MAXIMUM_NUMBER_MESSAGES];
static uint8_t succ_buffer[MAXIMUM_NUMBER_MESSAGES];

static uint32_t start_tick;

static float message_loss_gamma = 0.2f;
static float message_loss = 0.0f;

void init_cp_os()
{	
	round_nr = 1;
  	t_ref = gpi_tick_hybrid();

	for (node_id = 0; node_id < NUM_ELEMENTS(nodes); ++node_id) {
		if (nodes[node_id] == TOS_NODE_ID)
				break;
  	}
}

void run_communication_round(uint32_t num_tx_messages, message_t **tx_data, message_assignment_t *current_message_assignment, message_t *rx_data, uint8_t *succ, uint8_t is_first_layer, uint32_t *computing_time)
{
	if (!synchronized) {
		if (!is_first_layer) {
			while(1) {}
		}
	}
	// wait till the first layer starts.
	do {
		Gpi_Hybrid_Tick ticks_start;

		uint8_t received_initiator_message = 0;
		Gpi_Hybrid_Tick calculation_time, communication_time, print_time, last_evaluation_time;   // we exclued the evaluation from energy calculation
		calculation_time = gpi_tick_hybrid();
		uint32_t last_calculation_time = 0;
		uint32_t last_communication_time = 0;

		calculation_time = gpi_tick_hybrid() - calculation_time;
		last_calculation_time = gpi_tick_hybrid_to_us(calculation_time);
			
		// ******************************** */
		// Communication phase
		//******************************** */
		uint8_t is_initiator = TOS_NODE_ID == 1; //((round_nr-1) % NUM_ELEMENTS(fl_nodes) + 1) == TOS_NODE_ID;
		if (is_initiator) {
			synchronized = 1;
		}
		last_communication_time = communication_time;
		communication_time = 0;
		// reset aggregate and write prios in it.
		//memset(agg_input, 0, AGGREGATE_SIZE);
		// set_flag_in_agg(agg_input, fl_node_idx);
		// set_node_in_agg(agg_input, 0, TOS_NODE_ID);
		// set_prio_in_agg(agg_input, 0, device_prio);

		STOPWATCH_START();
		memset(valid, 0, MAXIMUM_NUMBER_MESSAGES);

		// write transmitted messages into the communication system.
		// write into mixer
		if (synchronized || is_initiator) {
			for (uint16_t tx_message_idx = 0; tx_message_idx < num_tx_messages; tx_message_idx++) {
				// when the agent does not want to send anything, it sends a TYPE_DUMMY
				if (tx_data[tx_message_idx]->header.type != TYPE_DUMMY) {
					uint8_t message_assignent_idx = get_assignment_idx(current_message_assignment, tx_data[tx_message_idx]->header.id);
					tx_data[tx_message_idx]->header.message_loss = (uint16_t)(message_loss * 10000);
					tx_messages[message_assignent_idx] = tx_data[tx_message_idx];
					valid[message_assignent_idx] = 1;
				}
			}
		}
		STOPWATCH_END(*computing_time);

		message_layer_state_t message_layer_state;
		message_layer_reset_state(&message_layer_state);
		memset(succ, 1, MAXIMUM_NUMBER_MESSAGES);
		uint8_t num_mixer_rounds = current_message_assignment->num_mixer_rounds;
		while (message_layer_state.current_round < num_mixer_rounds) {
			// init mixer
			mixer_init(node_id);
			mixer_set_weak_release_slot(WEAK_RELEASE_SLOT);
			mixer_set_weak_return_msg((void*)-1);

			
			// Initiator sends initiator packet. 
			if (is_initiator)
			{
				init_pkt.round = round_nr;
				init_pkt.round2 = round_nr;
				init_pkt.is_first_layer = is_first_layer;
				init_pkt.current_message_assignment_id = current_message_assignment->id;
				init_pkt.num_mixer_rounds = num_mixer_rounds;
				init_pkt.current_mixer_round = message_layer_state.current_round;
					
				// NOTE: we specified that the control packet uses index 0 and data packets use
				// indexes > 0. 

				mixer_write(0, &init_pkt, sizeof(init_message_t));
				// mixer_write(1, &init_pkt, sizeof(init_message_t));
			}
			
			STOPWATCH_START();
			message_layer_mixer_round_starting_callback(current_message_assignment, tx_messages, valid, &message_layer_state);
			STOPWATCH_END(*computing_time);

			// now write data into the communication system.
			//mixer_write_agg(agg_input);


			// arm mixer
			// start first round with infinite scan
			// -> nodes join next available round, does not require simultaneous boot-up
			//mixer_write_agg(agg_input);
			CLR_COM_LED();
			mixer_arm(((is_initiator) ? MX_ARM_INITIATOR : 0) | ((1 == round_nr && message_layer_state.current_round == 0) ? MX_ARM_INFINITE_SCAN : 0));

			// ******************************** */
			// single Communication round
			//******************************** */

			// poll such that mixer round starts at the correct time.
			// delay initiator a bit
			// -> increase probability that all nodes are ready when initiator starts the round
			// -> avoid problems in view of limited t_ref accuracy
			if (message_layer_state.current_round == 0) {
				// wait till  all have finished their computation (they have ROUND_PERIOD - MIXER_DURATION time for this)
				if (MX_INITIATOR_ID == TOS_NODE_ID)
				{
					while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, ROUND_PERIOD) + MIXER_INITIATOR_DELAY) < 0);
				}
				else
				{
					while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, ROUND_PERIOD)) < 0);
				}
			} else {
				while(1)  {}
				if (MX_INITIATOR_ID == TOS_NODE_ID)
				{
					while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, GPI_TICK_MS_TO_HYBRID2(10)) + MIXER_INITIATOR_DELAY) < 0);
				}
				else
				{
					while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, GPI_TICK_MS_TO_HYBRID2(10))) < 0);
				}
			}
			SET_COM_LED();   
			// ATTENTION: don't delay after the polling loop (-> print before)
			ticks_start = gpi_tick_hybrid();
			uint32_t mixer_start_tick = gpi_tick_hybrid();
			t_ref = mixer_start(); 

			// sometimes communication ends a bit earlier, when agent has received everything and its neightbours too.
			while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), SYNC_OFFSET(t_ref)) < 0);
			communication_time += gpi_tick_hybrid_to_us(gpi_tick_hybrid() - mixer_start_tick);
			CLR_COM_LED(); 

			// synchronize to the initiator node
			init_message_t init_message;
			STOPWATCH_START();
			received_initiator_message = read_message_from_mixer(0, (uint8_t *) &init_message, sizeof(init_message_t));
			STOPWATCH_END(*computing_time);
			if (received_initiator_message) {
				printf("""received initiator message\n");
				// sometimes we receive a dummy message.
				if (init_message.round != init_message.round2) {
					received_initiator_message = 0;
				}
				if (received_initiator_message) {
					initiator_message_not_received_counter = 0;
					if (init_message.is_first_layer) {
						synchronized = 1;
					}
					if (1 == round_nr) {
						round_nr = init_message.round;
						message_layer_state.current_round = init_message.current_mixer_round;
						num_mixer_rounds = init_message.num_mixer_rounds;
					// resynchronize when round number does not match
					} else if (init_message.round != round_nr) {
						printf("round_nr: %lu\n", round_nr);
						printf("init_message.round: %lu\n", init_message.round);
						round_nr = 0;	// increments to 1 with next round loop iteration
						while (1) {
							SET_COM_LED();  
							gpi_milli_sleep(100);
							CLR_COM_LED();  
							gpi_milli_sleep(100);
						}
					}
				}
			}
			if (!received_initiator_message) {
				initiator_message_not_received_counter++;
				if (initiator_message_not_received_counter > 10) {
					round_nr = 0;
					return;
				}
			}

			// read received data
			printf("succ[0]: %u\n", succ[0]);
			printf("succ[1]: %u\n", succ[1]);
			STOPWATCH_START();
			message_layer_mixer_round_finished_callback(current_message_assignment, rx_data, succ, &message_layer_state);
			message_layer_state.current_round++;
			STOPWATCH_END(*computing_time);
			
			// uint16_t succ_rx = 0;
			// for (uint16_t i = 0; i < current_message_assignment->length; i++) {
			// 	if (succ[i]) {
			// 		printf("id: %u\n", rx_data[i].header.id);
			// 		printf("type: %u\n", rx_data[i].header.type);
			// 		succ_rx++;
			// 	}
			// }
			// printf("rx: %u/%u\n", succ_rx, current_message_assignment->length);
			mixer_print_statistics();
			uint8_t rank = 0;
			for (unsigned i = 0; i < MX_GENERATION_SIZE; i++)
			{
					if (mixer_stat_slot(i) >= 0) ++rank;
			}
			printf("rank: %u\n", rank);
		}

		printf("Communication duration: %lu us\n", communication_time);

		// print how many received
		uint16_t messages_received_idx = 0;
		uint32_t total_message_loss = 0.0f;
		for (uint16_t i = 0; i < current_message_assignment->length; i++) {
			// printf("succ[%u]: %u\n", i, succ[i]);
			if (succ[i]) {
				// printf("%u\n", rx_data[i].header.id);
				// printf("%u\n", rx_data[i].header.type);
				total_message_loss += rx_data[i].header.message_loss;
				messages_received_idx++;
			}
		}
		if (messages_received_idx != 0) {
			total_message_loss /= messages_received_idx;
		}

		float current_message_loss = 1.0f - ((float)messages_received_idx / (float)(current_message_assignment->length));
		message_loss = message_loss * (1.0f - message_loss_gamma) + current_message_loss * message_loss_gamma;
		calculation_time = gpi_tick_hybrid();

		printf("Message loss this round: %u\n", total_message_loss);

		round_nr++;
	} while (!synchronized);
}


void all_gather(const int8_t *input, const uint32_t *input_range, const uint8_t *input_pruning, const message_assignment_t *input_message_assignment, const cmsis_nn_dims *dst_shape, int8_t *dst, uint8_t is_first_layer, uint32_t *computing_time)
{
	message_t message;
    message.activations_message.header.type = TYPE_ACTIVATIONS;
    message.activations_message.header.id = TOS_NODE_ID;
	
	uint8_t device_idx = TOS_NODE_ID - 1;
	uint32_t width_input = input_range[device_idx + 1] - input_range[device_idx];
	uint32_t activations_idx = 0;
	STOPWATCH_START();
	for (uint16_t k = 0; k < dst_shape->n; k++) {
		for (uint16_t j = input_range[device_idx]; j < input_range[device_idx + 1]; j++) {
			if (input_pruning[j - input_range[device_idx]] == 1) {
				message.activations_message.activations[activations_idx] = input[k*width_input + j-input_range[device_idx]];
				activations_idx++;
			}
		}
	}
	STOPWATCH_END(*computing_time);
	message_t *tx_data[1];
	tx_data[0] = &message;
	run_communication_round(1, tx_data, input_message_assignment, rx_messages, succ_buffer, is_first_layer, computing_time);
	start_tick = gpi_tick_hybrid();

	memset(dst, 0, sizeof(int8_t) * input_range[NUM_ELEMENTS(nodes)-1] * dst_shape->n);

	// accumulate received activations
	for (uint16_t i = 0; i < MAXIMUM_NUMBER_MESSAGES; i++) {
		if (succ_buffer[i] && rx_messages[i].header.type == TYPE_ACTIVATIONS) {
			uint16_t other_idx = rx_messages[i].header.id - 1;

			// we write the input data of our device later.
			if (rx_messages[i].header.id == TOS_NODE_ID) {
				continue;
			}

			// now copy the data from the message to the corresponding place in the input
			uint32_t activations_idx = 0;
			for (uint16_t k = 0; k < dst_shape->n; k++) {
				for (uint16_t j = input_range[other_idx]; j < input_range[other_idx + 1]; j++) {
					dst[k*dst_shape->c+j] = rx_messages[i].activations_message.activations[activations_idx];
					activations_idx++;
				}
			}
		}
	}

	device_idx = TOS_NODE_ID - 1;
    // now write input data of own device. It is the output of the last layer.
    activations_idx = 0;
    for (uint16_t k = 0; k < dst_shape->n; k++) {
        for (uint16_t j = input_range[device_idx]; j < input_range[device_idx + 1]; j++) {
            dst[k*dst_shape->c+j] = input[activations_idx];
            activations_idx++;
        }
    }
}

