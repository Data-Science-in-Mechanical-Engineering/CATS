#ifndef MESSAGE_ASSIGNMENT_H
#define MESSAGE_ASSIGNMENT_H

#include <stdint.h>

typedef struct message_assignment_element_t_tag 
{ 
	uint8_t id;   // id of message assignment
	uint16_t size;  // slot size in byte 
	uint16_t mixer_assignment_start;  // the index in mixer, the message starts 
	uint16_t mixer_assignment_end;   // the index in mixer the message ends (not including this index)
	uint16_t size_end; // the size of the piece of the message in the mixer message at index mixer_assignment_end-1 
} message_assignment_element_t;

typedef struct message_assignment_t_tag 
{ 
	uint8_t id;   // id of message assignment 
	uint16_t num_mixer_rounds;
	uint16_t length;
  	message_assignment_element_t *assignments; 
} message_assignment_t;


#endif