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
 *	@file					mixer_aggregate.c
 *
 *	@brief					Mixer aggregate management
 *
 *	@version				$Id: be0e025acb50b5e3d7ac98f722fc152df6637dd0 $
 *	@date					TODO
 *
 *	@author					Carsten Herrmann
 *
 ***************************************************************************************************

 	@details

	TODO

 **************************************************************************************************/
//***** Trace Settings *****************************************************************************

#include "gpi/trace.h"

// message groups for TRACE messages (used in GPI_TRACE_MSG() calls)
#define TRACE_INFO				GPI_TRACE_MSG_TYPE_INFO
#define TRACE_WARNING			GPI_TRACE_MSG_TYPE_WARNING
#define TRACE_ERROR				GPI_TRACE_MSG_TYPE_ERROR
#define TRACE_VERBOSE			GPI_TRACE_MSG_TYPE_VERBOSE

// select active message groups, i.e., the messages to be printed (others will be dropped)
#ifndef GPI_TRACE_BASE_SELECTION
	#define GPI_TRACE_BASE_SELECTION	GPI_TRACE_LOG_STANDARD | GPI_TRACE_LOG_PROGRAM_FLOW
#endif
GPI_TRACE_CONFIG(mixer_aggregate, GPI_TRACE_BASE_SELECTION | GPI_TRACE_LOG_USER);

//**************************************************************************************************
//**** Includes ************************************************************************************

#include "mixer_internal.h"

#include "gpi/tools.h"
#include "gpi/olf.h"

#include <stddef.h>
#include <string.h>

#if MX_AGGREGATE

//**************************************************************************************************
//***** Local Defines and Consts *******************************************************************

#if MX_AGGREGATE_IS_SIMPLE_(MX_AGGREGATE_CONFIG)
	// TODO: resolve manual field numbers
	//#define AGG_NUM_FLAGS	MX_AGGREGATE_FIELD_(0, MX_AGGREGATE_INTERNAL)
	//#define AGG_DATATYPE	MX_AGGREGATE_FIELD_(1, MX_AGGREGATE_INTERNAL)
	//#define AGG_FUNCTION	MX_AGGREGATE_FIELD_(2, MX_AGGREGATE_INTERNAL)
#elif MX_AGGREGATE_IS_MULTIPLEX_(MX_AGGREGATE_CONFIG)
	#define AGG_NUM_AGGS		MX_AGGREGATE_NUM_ELEMENTS_(MX_AGGREGATE_INTERNAL)
	#define AGG_NUM_FLAGS(i)	MX_AGGREGATE_NUM_FLAGS_(MX_AGGREGATE_FIELD_(i, MX_AGGREGATE_INTERNAL))
	#define AGG_VALUE_TYPE(i)	MX_AGGREGATE_VALUE_TYPE_(MX_AGGREGATE_FIELD_(i, MX_AGGREGATE_INTERNAL))
	#define AGG_FN_VALUE(i)		MX_AGGREGATE_FN_VALUE_(MX_AGGREGATE_FIELD_(i, MX_AGGREGATE_INTERNAL))
//#else
//	#error invalid aggregate type
#endif

//TEST_EXPANSION(AGG_NUM_AGGS)
//TEST_EXPANSION(AGG_NUM_FLAGS(0))
//TEST_EXPANSION(AGG_FUNCTION(1))
//TEST_EXPANSION(VA_NARG MX_AGGREGATE_INTERNAL)
//TEST_EXPANSION(MX_AGGREGATE_FIELD_(0, MX_AGGREGATE_FIELD_(1, MX_AGGREGATE_INTERNAL)))

//**************************************************************************************************
//***** Local Typedefs and Class Declarations ******************************************************

#if MX_AGGREGATE_IS_SIMPLE_(MX_AGGREGATE_CONFIG)

typedef struct __attribute__((packed)) Aggregate
{
	uint8_t			progress_flags[(MX_AGGREGATE_NUM_FLAGS_(MX_AGGREGATE_CONFIG) + 7) / 8];

	MX_AGGREGATE_VALUE_TYPE_(MX_AGGREGATE_CONFIG)	value;

} Aggregate;

#endif

//**************************************************************************************************

#if MX_AGGREGATE_IS_MULTIPLEX_(MX_AGGREGATE_CONFIG)

typedef union __attribute__((packed)) Aggregate
{
	#if (AGG_NUM_AGGS >= 1)
		struct __attribute__((packed))
		{
			uint8_t				progress_flags[(AGG_NUM_FLAGS(0) + 7) / 8];
			AGG_VALUE_TYPE(0)	value;

		} _0_;
	#endif

	#if (AGG_NUM_AGGS >= 2)
		struct __attribute__((packed))
		{
			uint8_t				progress_flags[(AGG_NUM_FLAGS(1) + 7) / 8];
			AGG_VALUE_TYPE(1)	value;

		} _1_;
	#endif

	#if (AGG_NUM_AGGS >= 3)
		struct __attribute__((packed))
		{
			uint8_t				progress_flags[(AGG_NUM_FLAGS(2) + 7) / 8];
			AGG_VALUE_TYPE(2)	value;

		} _2_;
	#endif

	#if (AGG_NUM_AGGS >= 4)
		#error TODO: extend union
	#endif

} Aggregate;

#endif

//**************************************************************************************************
#if 0
typedef struct __attribute__((packed)) Aggregate_Multiphase
{
	uint8_t		phase;

	union
	{
		// AggType	phase_1;
		// AggType	phase_2;
		// ...
	};

} Aggregate_Multiphase;
#endif

//**************************************************************************************************
//***** Forward Declarations ***********************************************************************

static void max_i8 (void *dst, const void *src, const void *upd);
static void max_u8 (void *dst, const void *src, const void *upd);
static void max_i16(void *dst, const void *src, const void *upd);
static void max_u16(void *dst, const void *src, const void *upd);

//**************************************************************************************************
//***** Local (Static) Variables *******************************************************************

static Aggregate	aggregate[2];

#if MX_AGGREGATE_IS_MULTIPLEX_(MX_AGGREGATE_CONFIG)

static const struct
{
	uint16_t	num_flags;
	uint8_t		sizeof_flags;
	uint8_t		sizeof_value;
	void		(*fn_agg_value)(void *dst, const void *src, const void *upd);

} MULTIPLEX[] =
{
	#define X(i)	{	\
		AGG_NUM_FLAGS(i),	\
		sizeof(aggregate[0]._ ## i ## _.progress_flags),	\
		sizeof(aggregate[0]._ ## i ## _.value),	\
		AGG_FN_VALUE(i)	},

	#if (AGG_NUM_AGGS > 0)
		X(0)
	#endif
	#if (AGG_NUM_AGGS > 1)
		X(1)
	#endif
	#if (AGG_NUM_AGGS > 2)
		X(2)
	#endif
	#if (AGG_NUM_AGGS > 3)
		#error TODO: extend data structure
	#endif

	#undef X
};

static typeof(MULTIPLEX[0])	*desc;

#endif

//**************************************************************************************************
//***** Global Variables ***************************************************************************



//**************************************************************************************************
//***** Local Functions ****************************************************************************

static void max_i8(void *dst, const void *src, const void *upd)
{
	*(int8_t*)dst = max(*(const int8_t*)src, *(const int8_t*)upd);
}

static void max_u8(void *dst, const void *src, const void *upd)
{
	*(uint8_t*)dst = max(*(const uint8_t*)src, *(const uint8_t*)upd);
}

static void max_i16(void *dst, const void *src, const void *upd)
{
	*(int16_t*)dst = max(*(const int16_t*)src, *(const int16_t*)upd);
}

static void max_u16(void *dst, const void *src, const void *upd)
{
	*(uint16_t*)dst = max(*(const uint16_t*)src, *(const uint16_t*)upd);
}

//**************************************************************************************************
//***** Global Functions ***************************************************************************

#if MX_AGGREGATE_IS_SIMPLE_(MX_AGGREGATE_CONFIG)

//**************************************************************************************************

void* mx_aggregate_init_simple(uint_fast8_t mode)
{
	GPI_TRACE_FUNCTION();

	if (0 != mode)
		GPI_TRACE_RETURN((void*)NULL);

	memset(&aggregate[0], 0, sizeof(aggregate[0]));

	GPI_TRACE_RETURN(&aggregate[0]);
}

//**************************************************************************************************

void* mx_aggregate_update_simple(void *current, unsigned int i, const void *value)
{
	GPI_TRACE_FUNCTION();

	assert(NULL != current);

	typeof(aggregate[0])	*p = current;

	if (i < MX_AGGREGATE_NUM_FLAGS_(MX_AGGREGATE_CONFIG))
		p->progress_flags[i / 8] |= gpi_slu(1, i % 8);

	MX_AGGREGATE_FN_VALUE_(MX_AGGREGATE_CONFIG)(&(p->value), &(p->value), value);

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags", p->progress_flags, sizeof(p->progress_flags));
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value", &p->value, sizeof(p->value));

	GPI_TRACE_RETURN(p);
}

//**************************************************************************************************

void* mx_aggregate_merge_simple(void *current_, const void *update_)
{
	GPI_TRACE_FUNCTION();

	//assert(NULL != current_);

	const typeof(aggregate[0])	*update = update_;
	typeof(aggregate[0])		*current = current_;
	typeof(aggregate[0])		*next = (current == &aggregate[0]) ? (current + 1) : (current - 1);
	const uint8_t				*src, *upd;
	uint8_t						*dst;

	MX_AGGREGATE_FN_VALUE_(MX_AGGREGATE_CONFIG)(&(next->value), &(current->value), &(update->value));

	src = &(current->progress_flags[0]);
	dst = &(next->progress_flags[0]);
	upd = &(update->progress_flags[0]);

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags current", current->progress_flags, sizeof(next->progress_flags));
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value current", &(current->value), sizeof(next->value));

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags receiced", update->progress_flags, sizeof(next->progress_flags));
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value receiced", &(update->value), sizeof(next->value));

	while (src != &(current->progress_flags[NUM_ELEMENTS(current->progress_flags)]))
		*dst++ = *src++ | *upd++;

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags new", next->progress_flags, sizeof(next->progress_flags));
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value new", &(next->value), sizeof(next->value));

	GPI_TRACE_RETURN(next);
}

//**************************************************************************************************

const void* mx_aggregate_read_simple(void *current)
{
	GPI_TRACE_FUNCTION();

	assert(NULL != current);

	GPI_TRACE_RETURN(current);
}

//**************************************************************************************************
//**************************************************************************************************

#elif MX_AGGREGATE_IS_MULTIPLEX_(MX_AGGREGATE_CONFIG)

//**************************************************************************************************

void* mx_aggregate_init_multiplex(uint_fast8_t mode)
{
	GPI_TRACE_FUNCTION();

	if (mode >= AGG_NUM_AGGS)
		GPI_TRACE_RETURN((void*)NULL);

	memset(&aggregate[0], 0, sizeof(aggregate[0]));

	desc = &MULTIPLEX[mode];

	GPI_TRACE_RETURN(&aggregate[0]);
}

//**************************************************************************************************

void* mx_aggregate_update_multiplex(void *current, unsigned int i, const void *value)
{
	GPI_TRACE_FUNCTION();

	assert(NULL != current);

	uint8_t	*pf = current;
	void	*pv = pf + desc->sizeof_flags;

	if (i < desc->num_flags)
		pf[i / 8] |= gpi_slu(1, i % 8);

	(desc->fn_agg_value)(pv, pv, value);

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags", pf, desc->sizeof_flags);
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value", pv, desc->sizeof_value);

	GPI_TRACE_RETURN(current);
}

//**************************************************************************************************

void* mx_aggregate_merge_multiplex(void *current_, const void *update)
{
	GPI_TRACE_FUNCTION();

	//assert(NULL != current_);

	typeof(aggregate[0])	*current = current_;
	void					*next = (current == &aggregate[0]) ? (current + 1) : (current - 1);
	const uint8_t			*src_flags = current_;
	const uint8_t			*upd_flags = update;
	uint8_t					*dst_flags = next;
	const void				*src_value = src_flags + desc->sizeof_flags;
	const void				*upd_value = upd_flags + desc->sizeof_flags;
	void					*dst_value = dst_flags + desc->sizeof_flags;

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags current", src_flags, desc->sizeof_flags);
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value current", src_value, desc->sizeof_value);

	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags received", upd_flags, desc->sizeof_flags);
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value received", upd_value, desc->sizeof_value);

	(desc->fn_agg_value)(dst_value, src_value, upd_value);

	for (uint_fast8_t i = desc->sizeof_flags; i-- > 0;)
		*dst_flags++ = *src_flags++ | *upd_flags++;

	dst_flags -= desc->sizeof_flags;
	TRACE_DUMP(TRACE_VERBOSE, "aggregate flags current", dst_flags, desc->sizeof_flags);
	TRACE_DUMP(TRACE_VERBOSE, "aggregate value current", dst_value, desc->sizeof_value);

	GPI_TRACE_RETURN(next);
}

//**************************************************************************************************
//**************************************************************************************************

#endif	// MX_AGGREGATE
