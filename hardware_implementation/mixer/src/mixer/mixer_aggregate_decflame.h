/***************************************************************************************************
 ***************************************************************************************************
 *
 *	Copyright (c) 2025, Networked Embedded Systems Lab, TU Dresden
 *	All rights reserved.
 *
 *	Redistribution and use in source and binary forms, with or without
 *	modification, are permitted provided that the following conditions are met:
 *		* Redistributions of source code must retain the above copyright
 *		  notice, this list of conditions and the following disclaimer.
 *		* Redistributions in binary form must reproduce the above copyright
 *		  notice, this list of conditions and the following disclaimer in the
 *		  documentation and/or other materials provided with the distribution.
 *		* Neither the name of the NES Lab or TU Dresden nor the
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
 *	@file					mixer_aggregate_decflame.h
 *
 *	@brief					Mixer aggregate management
 *
 *	@version				$Id: cb1e3af624ab38899aa44402bec1ffa5b97c768c $
 *	@date					TODO
 *
 *	@author					Carsten Herrmann
 *
 ***************************************************************************************************

 	@details

	TODO

 **************************************************************************************************/

#ifndef __MIXER_AGGREGATE_DECFLAME_H__
#define __MIXER_AGGREGATE_DECFLAME_H__

//**************************************************************************************************
//***** Includes ***********************************************************************************

#include "gpi/tools.h"

#include <stdint.h>

//**************************************************************************************************
//***** Global (Public) Defines and Consts *********************************************************

#define MX_AGGREGATE_CONFIG_DECFLAME(n, l, w, sp, tc)	(	\
	((n), (l), (w), (sp), ((tc) ? (tc) : -1)), \
	sizeof(Aggregate_Decflame),	\
	mx_aggregate_init_decflame,	\
	mx_aggregate_update_decflame,	\
	mx_aggregate_merge_decflame,	\
	mx_aggregate_read_decflame,	\
	mx_aggregate_get_status_decflame,	\
	mx_aggregate_print_statistics_decflame,	\
	100	)

//**************************************************************************************************
//***** Local (Private) Defines and Consts *********************************************************

#define MX_AGG_NUM_FLAGS			MX_AGGREGATE_FIELD_(0, MX_AGGREGATE_INTERNAL)
#define MX_AGG_MAX_LIST_LEN			MX_AGGREGATE_FIELD_(1, MX_AGGREGATE_INTERNAL)
#define MX_AGG_NUM_BITS_PER_PRIO	MX_AGGREGATE_FIELD_(2, MX_AGGREGATE_INTERNAL)
#define MX_AGG_MAX_NUM_SLOTS_0		MX_AGGREGATE_FIELD_(3, MX_AGGREGATE_INTERNAL)
#define MX_AGG_TURN_OFF_COUNTDOWN	MX_AGGREGATE_FIELD_(4, MX_AGGREGATE_INTERNAL)

#define	MX_AGG_NUM_BITS_PER_NODEID	(MSB(MX_NUM_NODES) + ((1 << MSB(MX_NUM_NODES)) != MX_NUM_NODES ? 1 : 0))
#define MX_AGG_NODEID_LIST_SIZE		(((MX_AGG_NUM_BITS_PER_NODEID * MX_AGG_MAX_LIST_LEN) + 7) / 8)

//**************************************************************************************************
//***** Forward Class and Struct Declarations ******************************************************



//**************************************************************************************************
//***** Global Typedefs and Class Declarations *****************************************************

// NOTE: highest value is reserved to mark invalid list (absent value).
// We distinguish this case from empty list because the latter may be a valid value.
ASSERT_CT_STATIC(MX_AGG_MAX_LIST_LEN < 63, MX_AGGREGATE_CONFIG_DECFLAME_max_list_len_too_high);

typedef struct __attribute__((packed)) Aggregate_Decflame
{
	uint8_t			progress_flags[(MX_AGG_NUM_FLAGS + 7) / 8];

	struct
	{
		uint8_t		list_len	: 6;	// consistent with ASSERT_CT_STATIC(MX_AGG_MAX_LIST_LEN < ...) above
		uint8_t		phase		: 2;
	};

	// phases:
	// 0: priority exchange
	// 1: Paxos prepare
	// 2: Paxos accept

	// set of nodes, stored as list of node IDs or bitfield with one bit per node (whatever is smaller).
	// phase 0: nodes corresponding to priorities
	// phase 1+2 (Paxos): proposed / accepted value
	uint8_t			nodes[MIN(MX_AGG_NODEID_LIST_SIZE, (MX_NUM_NODES + 7) / 8)];

	union
	{
		// phase 0 (priority exchange)
		uint8_t		priorities[(MX_AGG_MAX_LIST_LEN * MX_AGG_NUM_BITS_PER_PRIO + 7) / 8];

		// phase 1 (Paxos prepare)
		struct
		{
			uint8_t		proposal;
			uint8_t		max_accepted_proposal;
		};

		// phase 2 (Paxos accept)
		struct
		{
			uint8_t		_proposal_;				// dummy to avoid duplicate name
			uint8_t		max_observed_proposal;
			uint8_t		paxos_end[0];			// Paxos data end marker, no content
		};
	};

} Aggregate_Decflame;

//**************************************************************************************************

typedef struct Aggregate_Decflame_Result
{
	uint8_t		list_len;
	uint8_t		nodes[MX_AGG_MAX_LIST_LEN];

} Aggregate_Decflame_Result;

//**************************************************************************************************
//***** Global Variables ***************************************************************************



//**************************************************************************************************
//***** Prototypes of Global Functions *************************************************************

#ifdef __cplusplus
	extern "C" {
#endif

void*			mx_aggregate_init_decflame(uint_fast8_t mode);
void*			mx_aggregate_update_decflame(void *current, unsigned int i, const void *value);
void*			mx_aggregate_merge_decflame(void *current, const void *update, const void *packet);
const void*		mx_aggregate_read_decflame(void *current);
Mx_Aggregate_Status		mx_aggregate_get_status_decflame(uint_fast8_t called_from_tx);
void			mx_aggregate_print_statistics_decflame();

#ifdef __cplusplus
	}
#endif

//**************************************************************************************************
//***** Implementations of Inline Functions ********************************************************



//**************************************************************************************************
//**************************************************************************************************

#endif // __MIXER_AGGREGATE_DECFLAME_H__
