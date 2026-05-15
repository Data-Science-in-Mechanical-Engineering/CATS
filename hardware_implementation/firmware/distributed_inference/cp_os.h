/***************************************************************************************************
 ***************************************************************************************************
 *
 *	Copyright (c) 2022-, Institute for Data Science in Mechanical Engineering, RWTH Aachen
 *	All rights reserved.
 *
 *	Redistribution and use in source and binary forms, with or without
 *	modification, are permitted provided that the following conditions are met:
 *		* Redistributions of source code must retain the above copyright
 *		  notice, this list of conditions and the following disclaimer.
 *		* Redistributions in binary form must reproduce the above copyright
 *		  notice, this list of conditions and the following disclaimer in the
 *		  documentation and/or other materials provided with the distribution.
 *		* Neither the name of the DSME or RWTH nor the
 *		  names of its contributors may be used to endorse or promote products
 *		  derived from this software without specific prior written permission.
 *
 *	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
 *	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***********************************************************************************************//**
 *
 *	@file					cp_os.h
 *
 *	@brief					defines the functions and makros needed to run the communication processor
 *
 *	@version				$Id$
 *	@date					TODO
 *
 *	@author					Alexander Graefe
 *
 ***************************************************************************************************

 	@details

	

 **************************************************************************************************/

#ifndef CP_OS_H
#define CP_OS_H

#include "spi.h"
#include "internal_messages.h"
#include "mixer_config.h"
#include "wireless_control.h"
#include "arm_nn_types.h"
#include <stdint.h>

#ifndef SET_DEBUG_PIN
  #define STOPWATCH_START() {Gpi_Hybrid_Tick ticks_start_stopwatch = gpi_tick_hybrid();

#define STOPWATCH_END(X) X += gpi_tick_hybrid_to_us(gpi_tick_hybrid() - ticks_start_stopwatch);}
#else
#define STOPWATCH_START() {Gpi_Hybrid_Tick ticks_start_stopwatch = gpi_tick_hybrid(); SET_DEBUG_PIN

#define STOPWATCH_END(X) X += gpi_tick_hybrid_to_us(gpi_tick_hybrid() - ticks_start_stopwatch); RESET_DEBUG_PIN}
#endif

#define PRINT_COMPUTING_DURATION(X) printf("Computing duration: %ld us\n", X);


void init_cp_os();

/**
 * runs the communication processor. Call this method to run the communication including search for AP and other CPs
 */
void run_communication_round(uint32_t num_tx_messages, message_t **tx_data, message_assignment_t *current_message_assignment, message_t *rx_data, uint8_t *succ, uint8_t is_first_layer, uint32_t *computing_time);

/**
 * All gather operation writes input_data into Mixer (with pruning). Then receives all data from all devices and writes them into dst.
 * 
 * @param input input data
 * @param input_range range of the input data
 * @param input_pruning pruning of the input data (inter-device pruning)
 * @param dst_shape shape of the destination data
 * @param dst destination data
 */
void all_gather(const int8_t *input, const uint32_t *input_range, const uint8_t *input_pruning, const message_assignment_t *input_message_assignment, const cmsis_nn_dims *dst_shape, int8_t *dst, uint8_t is_first_layer, uint32_t *computing_time);

#endif
