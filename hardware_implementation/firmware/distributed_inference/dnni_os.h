#ifndef DNNI_OS_H
#define DNNI_OS_H

#include "internal_messages.h"

uint16_t process_data(uint32_t round_idx, message_t *rx_data, uint8_t *succ, message_t **tx_data, message_assignment_t *tx_message_assignment);

#endif  // DNNI_OS_H