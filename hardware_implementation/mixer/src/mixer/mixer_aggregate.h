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
 *	@file					mixer_aggregate.h
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

#ifndef __MIXER_AGGREGATE_H__
#define __MIXER_AGGREGATE_H__

//**************************************************************************************************
//***** Includes ***********************************************************************************

#include "gpi/tools.h"

#include <stdint.h>

//**************************************************************************************************
//***** Global (Public) Defines and Consts *********************************************************
/*
#define MX_AGGREGATE_CONFIG_MAX(n, d)	(	\
	(n, d, _Generic(*(d*)0, uint8_t: max_u8, int8_t: max_i8, uint16_t: max_u16, int16_t: max_i16)), \
	((n + 7) % 8 + sizeof(d)),	\
	mx_aggregate_init_simple,	\
	mx_aggregate_update_simple,	\
	mx_aggregate_merge_simple,	\
	mx_aggregate_read_simple,	\
	mx_aggregate_get_status_simple,	\
	mx_aggregate_print_statistics_simple,	\
	0	)
//	sizeof(Aggregate_Simple)

#define MX_AGGREGATE_MULTIPLEX2(a, b)	(	\
	(a, b), \
	MAX(MX_AGGREGATE_SIZE_(a), MX_AGGREGATE_SIZE_(b)),	\
	mx_aggregate_init_multiplex,	\
	mx_aggregate_update_multiplex,	\
	mx_aggregate_merge_multiplex,	\
	mx_aggregate_read_simple,	\
	mx_aggregate_get_status_multiplex,	\
	mx_aggregate_print_statistics_multiplex,	\
	1	)
//	sizeof(Aggregate_...)
*/
// expand macros, as used in typical two-step approaches
// NOTE: MX_AGGREGATE_EXPAND_ x (without parenthesis) can be used to enter tuple x = (a, b, c, ...)
// (i.e. to remove the outer parenthesis)
#define MX_AGGREGATE_EXPAND_(...)		__VA_ARGS__

// extract i-th field from tupel x
#define MX_AGGREGATE_FIELD_(i, x)		MX_AGGREGATE_EXTRACT_(i, MX_AGGREGATE_EXPAND_ x)
#define MX_AGGREGATE_EXTRACT_(i, ...)	VA_ARG_i(i, __VA_ARGS__)

// get number of elements in tuple x
#define MX_AGGREGATE_NUM_ELEMENTS_(x)	MX_AGGREGATE_EXPAND_(VA_NARG x)

#define MX_AGGREGATE_INTERNAL_(x)		MX_AGGREGATE_FIELD_(0, x)
#define MX_AGGREGATE_SIZE_(x)			MX_AGGREGATE_FIELD_(1, x)
#define MX_AGGREGATE_FN_INIT_(x)		MX_AGGREGATE_FIELD_(2, x)
#define MX_AGGREGATE_FN_UPDATE_(x)		MX_AGGREGATE_FIELD_(3, x)
#define MX_AGGREGATE_FN_MERGE_(x)		MX_AGGREGATE_FIELD_(4, x)
#define MX_AGGREGATE_FN_READ_(x)		MX_AGGREGATE_FIELD_(5, x)
#define MX_AGGREGATE_FN_STATUS_(x)		MX_AGGREGATE_FIELD_(6, x)
#define MX_AGGREGATE_FN_PRINT_STAT_(x)	MX_AGGREGATE_FIELD_(7, x)
#define MX_AGGREGATE_TYPE_(x)			MX_AGGREGATE_FIELD_(8, x)

//#define MX_AGGREGATE_IS_SIMPLE_(x)		(0 == MX_AGGREGATE_TYPE_(x))
//#define MX_AGGREGATE_IS_MULTIPLEX_(x)	(1 == MX_AGGREGATE_TYPE_(x))

//#define MX_AGGREGATE_NUM_FLAGS_(x)		MX_AGGREGATE_FIELD_(0, MX_AGGREGATE_INTERNAL_(x))
//#define MX_AGGREGATE_VALUE_TYPE_(x)		MX_AGGREGATE_FIELD_(1, MX_AGGREGATE_INTERNAL_(x))
//#define MX_AGGREGATE_FN_VALUE_(x)		MX_AGGREGATE_FIELD_(2, MX_AGGREGATE_INTERNAL_(x))

#define MX_AGGREGATE_INTERNAL			MX_AGGREGATE_INTERNAL_ (MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_SIZE				MX_AGGREGATE_SIZE_     (MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_INIT			MX_AGGREGATE_FN_INIT_  (MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_UPDATE			MX_AGGREGATE_FN_UPDATE_(MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_MERGE			MX_AGGREGATE_FN_MERGE_ (MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_READ			MX_AGGREGATE_FN_READ_  (MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_STATUS			MX_AGGREGATE_FN_STATUS_(MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_FN_PRINT_STAT		MX_AGGREGATE_FN_PRINT_STAT_(MX_AGGREGATE_CONFIG)
#define MX_AGGREGATE_TYPE				MX_AGGREGATE_TYPE_     (MX_AGGREGATE_CONFIG)

//**************************************************************************************************
//***** Local (Private) Defines and Consts *********************************************************



//**************************************************************************************************
//***** Forward Class and Struct Declarations ******************************************************



//**************************************************************************************************
//***** Global Typedefs and Class Declarations *****************************************************

typedef struct Mx_Aggregate_Status
{
	uint8_t		is_ready;
	int8_t		tx_hint;
					//  0 = normal
					//  1 = prefer (e.g. agg. is innovative)
					// -1 = don't send (e.g. inactive or empty or complete)

} Mx_Aggregate_Status;

//**************************************************************************************************
/*
typedef struct Mixer_Aggregate_Configuration
{
	size_t		aggregate_size;

	void*		(*fn_init)(int_fast8_t mode);
	void*		(*fn_update)(void *current, unsigned int i, const void *value);
	void*		(*fn_merge)(void *current, const void *update);

} Mixer_Aggregate_Configuration;
*/
//**************************************************************************************************
//***** Global Variables ***************************************************************************



//**************************************************************************************************
//***** Prototypes of Global Functions *************************************************************

#ifdef __cplusplus
	extern "C" {
#endif

void*			mx_aggregate_init_simple(uint_fast8_t mode);
void*			mx_aggregate_update_simple(void *current, unsigned int i, const void *value);
void*			mx_aggregate_merge_simple(void *current, const void *update);
const void*		mx_aggregate_read_simple(void *current);

void*			mx_aggregate_init_multiplex(uint_fast8_t mode);
void*			mx_aggregate_update_multiplex(void *current, unsigned int i, const void *value);
void*			mx_aggregate_merge_multiplex(void *current, const void *update);

#ifdef __cplusplus
	}
#endif

//**************************************************************************************************
//***** Implementations of Inline Functions ********************************************************



//**************************************************************************************************
//***** Extra Includes *****************************************************************************

// extra header files are included *after* definitions and declarations from above

#ifdef MX_AGGREGATE_INCLUDE
	#include STRINGIFY(MX_AGGREGATE_INCLUDE)
#endif

//**************************************************************************************************
//**************************************************************************************************

#endif // __MIXER_AGGREGATE_H__
