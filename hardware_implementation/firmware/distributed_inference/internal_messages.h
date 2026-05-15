/*
 * internal_messages.h
 *
 * DEPRECATED DESCRIPTION: messages, which are send via spi and mixer. Please define your messages in messages.h. via typedef mixer_data_t your_message_t
 * in order to get the length of the messages, define a makro MESSAGES_SIZES(type), which returns the size of the message for its type.
 * Define the type of the message starting with 0 counting up.
 *  Created on: Mar 18, 2022
 *      Author: mf724021
 */

#ifndef INC_INTERNAL_MESSAGES_H_
#define INC_INTERNAL_MESSAGES_H_

#include <stdint.h>
#include "dnni_mixer_config.h"
#include "message_assignment.h"

#define TYPE_ERROR 0  // when something went wrong and the following data should not been used (e.g. CP did not receive any new data from AP)
                      // can be used to check if AP is online and ready
#define TYPE_METADATA 1  // message contains metadata about itself (sent by CP, when CP is booting up)
#define TYPE_AP_ACK 2 // AP has acknowledged that CP is online and AP is ready.
#define TYPE_CP_ACK 3 // CP acknowledged that AP is ready. AP should now wait for CP to find connection to other CPs
#define TYPE_ALL_AGENTS_READY 4  // all agents are ready
#define TYPE_AP_DATA_REQ 5      // request data from AP this message is send right before the CP wants to receive data from the AP.

// Deprecated comments
#define TRANSFORM_TYPE_BACK(type) (type - TYPE_AP_DATA_REQ - 1)  // the type, which is received by mixer or spi
                                                                  // transformed to the type defined in messages.h
#define TRANSFORM_TYPE(type) (type + TYPE_AP_DATA_REQ + 1)       // the type defined in messages.h,
                                                                  // transformed to the type, which should be sent via spi and mixer
#define TYPE_DUMMY 255                                            // the AP sends this message (metadata), when the CP should ignore this message (we need this, because otherwise the UART would not work)

// every (custom) message type must contain message_t as header!
typedef struct __attribute__((packed)) header_message_t_tag
{
        uint8_t type;
        uint8_t id; //ID of Agent, which sends the message (or the slot of the mesage)
        uint16_t message_loss;
        // uint16_t pad;
} header_message_t;

/**
 * struct sent in order to share state of whole system with AP
 */
typedef struct __attribute__((packed)) metadata_message_t_tag
{
        header_message_t header;
        uint16_t round_nmbr;
} metadata_message_t;

typedef struct __attribute__((packed)) init_message_t_tag
{       
        uint32_t round;
        uint32_t round2;
        uint8_t is_first_layer;
        uint8_t num_mixer_rounds;
        uint8_t current_mixer_round;
        uint8_t current_message_assignment_id;
} init_message_t;

/***********user defined messages***************/
// ATTENTION do not forget to overwrite uint16_t message_sizes(uint8_t type) in internal_messages.c

#define TYPE_ACTIVATIONS TRANSFORM_TYPE(0)

#define MESSAGES_SIZES(type) message_sizes(type)

// TODO: Ensure that MAXIMUM_SIZE_LAYER is defined during code generation!!

// Add custom message types here:
// Do not forget to add them to mixer_message_t_tag

// typedef struct __attribute__((packed)) custom_message_t_tag
// {
//      var_t var;
// } custom_message_t;


#ifndef LENGTH_TIME_SERIES
#define LENGTH_TIME_SERIES (300)
typedef int8_t time_series_type_t;
#endif

typedef struct __attribute__((packed)) activations_message_t_tag
{
        header_message_t header;
        int8_t activations[SIZE_ACTIVATIONS];
} activations_message_t;

// write all possible messages here. This allows us to quickly transform the data from bytes to useful structs.
typedef union message_t_tag
{
	header_message_t header;
	metadata_message_t metadata_message;
        activations_message_t activations_message;
} message_t;


typedef struct ap_com_handle_tag
{
	uint8_t *raw_data_buffer;
	uint8_t *dummy;

        void (*ap_send)(uint8_t *, uint16_t); 
        void (*ap_receive)(uint8_t *, uint16_t);
	void (*rx_wait)();
        void (*tx_wait)();
} ap_com_handle;

void init_ap_com(ap_com_handle *hap_com, void (*ap_send_p)(uint8_t *, uint16_t), void (*ap_receive_p)(uint8_t *, uint16_t), void (*rx_wait_p)(), void (*tx_wait_p)());
uint16_t get_size(const message_t *message);

uint16_t raw_data_to_messages(const uint8_t *raw_data, message_t **messages, uint16_t size);

uint16_t messages_to_raw_data(uint8_t *raw_data, const message_t *messages, uint16_t size);

void send_data_to(ap_com_handle *hap_com, const message_t *messages, uint16_t size);

uint16_t receive_data_from(ap_com_handle *hap_com, message_t **messages);

#endif /* INC_MESSAGES_H_ */
