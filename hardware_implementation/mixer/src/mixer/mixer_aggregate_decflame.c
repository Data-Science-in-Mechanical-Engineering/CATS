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
 *	@file					mixer_aggregate_decflame.c
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
GPI_TRACE_CONFIG(mixer_aggregate_decflame, GPI_TRACE_BASE_SELECTION | GPI_TRACE_LOG_USER);

//**************************************************************************************************
//**** Includes ************************************************************************************

#include "mixer_internal.h"

#include "gpi/tools.h"
#include "gpi/olf.h"

#include <stddef.h>		// NULL
#include <string.h>		// memcpy()
#include <stdio.h>		// snprintf()

#if MX_AGGREGATE

//**************************************************************************************************
//***** Local Defines and Consts *******************************************************************

#if (100 != MX_AGGREGATE_TYPE)
	#error "MX_AGGREGATE_CONFIG implementation mismatch"
#endif

#define MX_AGG_NODES_USE_BITFIELD	(MX_AGG_NODEID_LIST_SIZE >= (MX_NUM_NODES + 7) / 8)

#if (GPI_TRACE_MODE_IS_TRACE)
	#define TRACE_AGG(group, header, agg)	if (gpi_trace_module_desc.msg_config & (group)) trace_aggregate((header), (agg))
#else
	#define TRACE_AGG(group, header, agg)	if (0)
#endif

//**************************************************************************************************
//***** Local Typedefs and Class Declarations ******************************************************

ASSERT_CT_STATIC(MX_AGG_NUM_BITS_PER_NODEID <= 8, node_id_width_exceeds_limit);
ASSERT_CT_STATIC(MX_AGG_NUM_BITS_PER_PRIO <= 24, priority_width_exceeds_limit);

// node-with-priority entry in all variants
union Node_Prio_Compound
{
	// NOTE: some configurations could be packed more dense using bitfields,
	// but we omit that in favor of performance
	// ATTENTION: all layouts must guarantee that raw member does not include any padding bits
	// (e.g. in bitfields or alignment padding) to ensure that comparisons of entries based on
	// raw member produce same results as comparisons using node and prio members

	#if (MX_AGG_NUM_BITS_PER_PRIO <= 8)

		union
		{
			uint16_t		raw;
			struct
			{
				uint8_t		node;
				uint8_t		prio;
			};
		} np;

		union
		{
			uint16_t		raw;
			struct
			{
				uint8_t		prio;
				uint8_t		node;
			};
		} pn;

	#elif (MX_AGG_NUM_BITS_PER_PRIO <= 16)

		union
		{
			uint32_t		raw;
			struct
			{
				uint16_t	node;	// uint16_t to avoid undefined value of alignment padding
				uint16_t	prio;
			};
		} np;

		union
		{
			uint32_t		raw;
			struct
			{
				uint16_t	prio;
				uint16_t	node;	// uint16_t to avoid undefined value of alignment padding
			};
		} pn;

	#elif (MX_AGG_NUM_BITS_PER_PRIO <= 24)

		union
		{
			uint32_t		raw;
			struct
			{
				uint32_t	node	: 8;
				uint32_t	prio	: 24;
			};
		} np;

		union
		{
			uint32_t		raw;
			struct
			{
				uint32_t	prio	: 24;
				uint32_t	node	: 8;
			};
		} pn;

	#else
		#error "priority width must be <= 24"
	#endif
};

// node-with-priority entries accounting for the byte order.
// this ensures the priorities of node and prio members when comparing raw values
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	typedef typeof_member(union Node_Prio_Compound, pn)		Node_Prio_Entry;
	typedef typeof_member(union Node_Prio_Compound, np)		Prio_Node_Entry;
#elif (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
	typedef typeof_member(union Node_Prio_Compound, np)		Node_Prio_Entry;
	typedef typeof_member(union Node_Prio_Compound, pn)		Prio_Node_Entry;
#else
	#error "unknown byte order"
#endif

// nodes-with-priorities list.
// internally it consists of two lists, one sorted by node and then by prio,
// the other sorted by prio and then by node. This makes it easy to search
// both, nodes and priorities, efficiently.
typedef struct Node_Prio_List
{
	uint8_t				list_len;

	Node_Prio_Entry		node_prio_list[MX_AGG_MAX_LIST_LEN];
	Prio_Node_Entry		prio_node_list[MX_AGG_MAX_LIST_LEN];

} Node_Prio_List;

//**************************************************************************************************

typedef struct Paxos_Value
{
	uint8_t				list_len;
	uint8_t				nodes[sizeof_member(Aggregate_Decflame, nodes)];

} Paxos_Value;

//**************************************************************************************************
//***** Forward Declarations ***********************************************************************

static uint_fast8_t agg_unpack(Node_Prio_List *dst, const Aggregate_Decflame *src);

//**************************************************************************************************
//***** Local (Static) Variables *******************************************************************

static Aggregate_Decflame	aggregate[2];
static Mx_Aggregate_Status	agg_status;

static struct
{
	int8_t					phase;

	union
	{
		// during priority exchange
		Node_Prio_List		node_prio_list;

		// during Paxos phases
		struct
		{
			// proposer data
			uint8_t			proposal;
			Paxos_Value		proposed_value;
			int8_t			proposer_state;
								//  0 = proposal active
								//  1 = proposal accepted by majority of acceptors
								// -1 = proposal dropped (higher proposal issued by other proposer)

			// acceptor data
			uint8_t			min_proposal;
			int16_t			accepted_proposal;
			Paxos_Value		accepted_value;

			// received aggregate values
			uint8_t			max_observed_proposal;
			int16_t			max_accepted_proposal;
			Paxos_Value		max_accepted_value;

//			Paxos_Value		learned_value;
			uint8_t			turn_off_countdown;
		};

		// after post-processing of accepted proposal
		Aggregate_Decflame_Result	result;
	};

} local;

#if MX_VERBOSE_STATISTICS
static struct
{
	uint16_t	slot_paxos_start;
	uint16_t	slot_proposer_lost;
	uint16_t	slot_proposer_accept;
	uint16_t	slot_proposer_done;
	uint16_t	slot_value_learned;
	uint16_t	slot_complete;
	uint16_t	slot_off;

} statistics;
#endif

//**************************************************************************************************
//***** Global Variables ***************************************************************************



//**************************************************************************************************
//***** Local Functions ****************************************************************************

#if (GPI_TRACE_MODE_IS_TRACE)

static void trace_aggregate(const char *header, const Aggregate_Decflame *agg)
{
	const int8_t	list_len = (agg->list_len > MX_AGG_MAX_LIST_LEN) ? -1 : agg->list_len;
	char			h[40];
	char* const		s = h + snprintf(h, sizeof(h), "%s ", header);
	const size_t	sizeof_s = &h[sizeof(h)] - s;

	snprintf(s, sizeof_s, ":");
	TRACE_DUMP(TRACE_VERBOSE, h, agg, sizeof(*agg));

	snprintf(s, sizeof_s, "flags =");
	TRACE_DUMP(TRACE_VERBOSE, h, agg->progress_flags, sizeof(agg->progress_flags));

	GPI_TRACE_MSG(TRACE_VERBOSE, "%s phase = %" PRIu8 ", len = %" PRIi8,
		header, agg->phase, list_len);

	switch (agg->phase)
	{
		case 0:
			snprintf(s, sizeof_s, "prios (packed) =");
			TRACE_DUMP(TRACE_VERBOSE, h, agg->priorities, sizeof(agg->priorities));
			break;

		case 1:
			GPI_TRACE_MSG(TRACE_VERBOSE,
				"%s proposal = %" PRIu8 ", max_accepted_proposal = %" PRIi16,
				header, agg->proposal, (list_len < 0) ? INT16_C(-1) : (int16_t)(agg->max_accepted_proposal));
			break;

		case 2:
			GPI_TRACE_MSG(TRACE_VERBOSE,
				"%s proposal = %" PRIu8 ", max_observed_proposal = %" PRIu8,
				header, agg->proposal, agg->max_observed_proposal);
			break;
	}

	if (list_len >= 0)
	{
		Node_Prio_List	npl;
		uint_fast8_t	i;

		snprintf(s, sizeof_s, "nodes (packed) =");
		TRACE_DUMP(TRACE_VERBOSE, h, agg->nodes, sizeof(agg->nodes));

		agg_unpack(&npl, agg);

		for (i = 0; i < list_len; ++i)
			GPI_TRACE_MSG(TRACE_VERBOSE, "%s node %" PRIu16 " prio %" PRIi32,
				header, npl.node_prio_list[i].node, (agg->phase > 0) ? INT32_C(-1) : (int32_t)(npl.node_prio_list[i].prio));
	}
}

#endif
//**************************************************************************************************

// read own node ID
static inline uint_fast8_t my_node_id()
{
	return mx.tx_packet.sender_id;
}

//**************************************************************************************************

// set i-th bit in bitfield, e.g. i-th progress flag
static inline void set_bit(uint8_t *flags, uint_fast8_t i)
{
	flags[i / 8] |= gpi_slu(UINT8_C(1), i % 8);
}

//**************************************************************************************************

// return index of element with node == value, -1 if not in list
static int_fast16_t list_search(const Node_Prio_Entry *list, uint_fast8_t len, uint_fast16_t value)
{
	uint_fast8_t	left = 0, right = len, middle;
	uint_fast16_t	x;

	while (left < right)
	{
		middle = (left + right) / 2;
		x = list[middle].node;

		if (x == value)
			return middle;

		else if (x > value)
			right = middle;

		else left = middle + 1;
	}

	return -1;
}

//**************************************************************************************************

// re-sort list after element i has been added or updated.
// can be used for node_prio_list as well as prio_node_list
static void list_update(void* list_, uint_fast8_t len, uint_fast8_t i)
{
	typeof_field(Node_Prio_Entry, raw)*	list = list_;
	typeof(list[0])						x;

	if (len < 2)
		return;

	// the following algorithm is similar to a single step of insertion sort algorithm

	x = list[i];

	if (i > 0 && list[i - 1] > x)
	{
		do
		{
			list[i] = list[i - 1];
			i--;
		}
		while (i > 0 && list[i - 1] > x);

		list[i] = x;
	}

	else if (i < len - 1 && list[i + 1] < x)
	{
		do
		{
			list[i] = list[i + 1];
			i++;
		}
		while (i < len - 1 && list[i + 1] < x);

		list[i] = x;
	}
}

//**************************************************************************************************

// pack nodes+priorities list from internal format (optimized for performance) to packet format (optimized for space)
static void agg_pack(Aggregate_Decflame *dst, const Node_Prio_List *src)
{
	ASSERT_CT(MX_AGG_NUM_BITS_PER_NODEID <= 8);
	ASSERT_CT(MX_AGG_NUM_BITS_PER_PRIO <= 16);

	const uint_fast8_t	list_len = src->list_len;
	uint8_t				*n_dst = &(dst->nodes[0]);
	uint8_t				*p_dst = &(dst->priorities[0]);
	uint_fast32_t		x;			// raw data buffer
	uint_fast8_t		x_used;		// number of valid bits in x
	uint_fast8_t		i;

	// check if all node IDs are valid
	// NOTE: list is sorted, so it is enough to check last entry
	if (list_len > 0 && src->node_prio_list[list_len - 1].node >= MX_NUM_NODES)
	{
		GPI_TRACE_MSG(TRACE_ERROR, "node IDs > %u detected, dropping aggregate", MX_NUM_NODES - 1);
		assert(0);
		dst->list_len = 0;
		return;
	}

	// if nodes stored as list of node IDs
	if (!MX_AGG_NODES_USE_BITFIELD)
	{
		x = 0;
		x_used = 0;

		for (i = 0; i < list_len; ++i)
		{
			// ATTENTION: result gets wrong if value does not fit into MX_AGG_NUM_BITS_PER_NODEID
			// (then upper bits get ORed with lower bits from following list entry).
			// We could mask it for safety, but resulting value would still be wrong,
			// so we save the effort.
			// assert(src->node_prio_list[i].u16_h < (1 << MX_AGG_NUM_BITS_PER_NODEID));

			x |= src->node_prio_list[i].node << x_used;
			x_used += MX_AGG_NUM_BITS_PER_NODEID;

			if (x_used >= 8)
			{
				*n_dst++ = x;
				x >>= 8;
				x_used -= 8;
			}
		}

		if (x_used > 0)
			*n_dst++ = x;
	}

	// if nodes stored as bitfield
	else
	{
		memset(dst->nodes, 0, sizeof(dst->nodes));

		for (i = 0; i < list_len; ++i)
		{
			uint_fast8_t x = src->node_prio_list[i].node;
			set_bit(dst->nodes, x);
		}
	}

	// priorities
	{
		x = 0;
		x_used = 0;

		for (i = 0; i < list_len; ++i)
		{
			// ATTENTION: result gets wrong if value does not fit into MX_AGG_NUM_BITS_PER_PRIO
			// (then upper bits get ORed with lower bits from following list entry).
			// We could mask it for safety, but resulting value would still be wrong,
			// so we save the effort.
			// assert(src->node_prio_list[i].prio < (1 << MX_AGG_NUM_BITS_PER_PRIO));

			x |= src->node_prio_list[i].prio << x_used;
			x_used += MX_AGG_NUM_BITS_PER_PRIO;

			while (x_used >= 8)
			{
				*p_dst++ = x;
				x >>= 8;
				x_used -= 8;
			}
		}

		if (x_used > 0)
			*p_dst++ = x;
	}

	dst->list_len = list_len;
}

//**************************************************************************************************

// unpack nodes+priorities list from packet format (optimized for space) to internal format (optimized for performance)
static uint_fast8_t agg_unpack(Node_Prio_List *dst, const Aggregate_Decflame *src)
{
	GPI_TRACE_FUNCTION();

	ASSERT_CT(MX_AGG_NUM_BITS_PER_NODEID <= 8);
	ASSERT_CT(MX_AGG_NUM_BITS_PER_PRIO <= 16);

	const uint_fast8_t	NODE_MASK = (1 << MX_AGG_NUM_BITS_PER_NODEID) - 1;
	const uint_fast8_t	PRIO_MASK = (1 << MX_AGG_NUM_BITS_PER_PRIO) - 1;

	uint_fast8_t		list_len = src->list_len;
	const uint8_t		*n_src = &(src->nodes[0]);
	const uint8_t		*p_src = &(src->priorities[0]);
	uint_fast32_t		p = 0;			// priorities raw data buffer
	uint_fast8_t		p_used = 0;		// number of valid bits in p
	Node_Prio_Entry		node_prio;

	// return empty list if src is malformed
	dst->list_len = 0;

	if (list_len > MX_AGG_MAX_LIST_LEN)
	{
		GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: list_len == %" PRIuFAST8 " > %u",
			list_len, MX_AGG_MAX_LIST_LEN);
		GPI_TRACE_RETURN(0);
	}

	// if nodes stored as list of node IDs
	if (!MX_AGG_NODES_USE_BITFIELD)
	{
		uint_fast16_t	n = 0;			// nodes raw data buffer
		uint_fast8_t	n_used = 0;		// number of valid bits in n

		for (uint_fast8_t i = 0; i < list_len; ++i)
		{
			if (n_used < MX_AGG_NUM_BITS_PER_NODEID)
			{
				n |= *n_src++ << n_used;
				n_used += 8;
			}

			// NOTE: read byte-wise to avoid unaligned word accesses (to be platform-independent)
			while (p_used < MX_AGG_NUM_BITS_PER_PRIO)
			{
				p |= *p_src++ << p_used;
				p_used += 8;
			}

			node_prio.node = n & NODE_MASK;
			node_prio.prio = p & PRIO_MASK;

			n >>= MX_AGG_NUM_BITS_PER_NODEID;
			n_used -= MX_AGG_NUM_BITS_PER_NODEID;

			p >>= MX_AGG_NUM_BITS_PER_PRIO;
			p_used -= MX_AGG_NUM_BITS_PER_PRIO;

			if (node_prio.node >= MX_NUM_NODES)
			{
				GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: node ID %u > %u detected",
					(int)node_prio.node, MX_NUM_NODES - 1);

				// NOTE: Instead of ignoring the whole aggregate we could drop just the
				// current entry, but we prefer to ignore malformed aggregates (for safety).
				#if 0
					list_len--;
					continue;
				#else
					GPI_TRACE_RETURN(0);
				#endif
			}

			dst->node_prio_list[i] = node_prio;
			dst->prio_node_list[i].prio = node_prio.prio;
			dst->prio_node_list[i].node = node_prio.node;

			list_update(dst->prio_node_list, i + 1, i);

			// NOTE: if src is well-formed then node IDs are stored in ascending order,
			// so sorting can be skipped. We could do it anyway to catch malformed aggregates,
			// but we prefer to ignore malformed aggregates (for safety). Note that sorting
			// would be fast for well-formed aggregates with ordered node IDs.
			if (i > 0 && dst->node_prio_list[i - 1].raw >= node_prio.raw)
			{
				GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: invalid node list order");
				#if 0
					list_update(dst->node_prio_list, i + 1, i);
				#else
					GPI_TRACE_RETURN(0);
				#endif
			}
		}
	}

	// if nodes stored as bitfield
	else
	{
		uint_fast8_t	i = 0;
		uint_fast8_t	n_offset = 0;
		uint_fast8_t	n;

		for (; n_src < &(src->nodes[sizeof(src->nodes)]); n_offset += 8)
		{
			n = *n_src++;

			while (n)
			{
				if (i >= list_len)
				{
					GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: inconsistent list length");
					GPI_TRACE_RETURN(0);
				}

				// NOTE: read byte-wise to avoid unaligned word accesses (to be platform-independent)
				while (p_used < MX_AGG_NUM_BITS_PER_PRIO)
				{
					p |= *p_src++ << p_used;
					p_used += 8;
				}

				node_prio.node = n_offset + gpi_get_lsb(n);
				node_prio.prio = p & PRIO_MASK;

				// clear LSB
				n &= ~(-n);

				p >>= MX_AGG_NUM_BITS_PER_PRIO;
				p_used -= MX_AGG_NUM_BITS_PER_PRIO;

				if (node_prio.node >= MX_NUM_NODES)
				{
					GPI_TRACE_MSG(TRACE_WARNING, "node ID %u > %u detected, dropping entry",
						(int)node_prio.node, MX_NUM_NODES - 1);

					// NOTE: Instead of ignoring the whole aggregate we could drop just the
					// current entry, but we prefer to ignore malformed aggregates (for safety).
					#if 0
						list_len--;
						continue;
					#else
						GPI_TRACE_RETURN(0);
					#endif
				}

				dst->node_prio_list[i] = node_prio;
				dst->prio_node_list[i].prio = node_prio.prio;
				dst->prio_node_list[i].node = node_prio.node;

				list_update(dst->prio_node_list, i + 1, i);
				// NOTE: nodes are sorted implicitly, so there is no need to sort node_prio_list

				i++;
			}
		}

		if (i != list_len)
		{
			GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: inconsistent list length");
			GPI_TRACE_RETURN(0);
		}
	}

	dst->list_len = list_len;

	GPI_TRACE_RETURN(1);
}

//**************************************************************************************************
//***** Global Functions ***************************************************************************

void* mx_aggregate_init_decflame(uint_fast8_t mode)
{
	GPI_TRACE_FUNCTION();

	if (0 != mode)
		GPI_TRACE_RETURN((void*)NULL);

	// check node ID (for safety)
	// NOTE: We assume that the return value of my_node_id() does not change
	// during a round (we address user mistakes, rather than bugs). Alternatively
	// we could perform the test inside my_node_id() to be absolutely safe,
	// but we avoid this in favor of performance.
	if (my_node_id() >= MX_NUM_NODES && my_node_id() >= MX_AGG_NUM_FLAGS)
	{
		GPI_TRACE_MSG(TRACE_ERROR, "will not activate aggregate because node ID exceeds limits!");
		assert(0);
		GPI_TRACE_RETURN((void*)NULL);
	}

	memset(&aggregate[0], 0, sizeof(aggregate[0]));
	local.phase = 0;
	local.node_prio_list.list_len = 0;
	agg_status.is_ready = 0;
	agg_status.tx_hint = -1;

	#if MX_VERBOSE_STATISTICS
		memset(&statistics, 0, sizeof(statistics));
	#endif

	GPI_TRACE_RETURN(&aggregate[0]);
}

//**************************************************************************************************

Mx_Aggregate_Status mx_aggregate_get_status_decflame(uint_fast8_t called_from_tx)
{
	GPI_TRACE_FUNCTION();

	if (called_from_tx)
	{
		// reset tx_hint == PREFER
		if (agg_status.tx_hint > 0)
			agg_status.tx_hint = 0;

		// if turn-off countdown active
		if (MX_AGG_TURN_OFF_COUNTDOWN > 0 && agg_status.is_ready > 1 && agg_status.tx_hint >= 0)
		{
			// turn-off after specified number of aggregate transmissions
			// NOTE: we count aggregate transmissions, not slots or transmissions in general
			if (!--local.turn_off_countdown)
			{
				GPI_TRACE_MSG(TRACE_VERBOSE, "aggregate turn off");

				agg_status.tx_hint = -1;

				#if MX_VERBOSE_STATISTICS
					statistics.slot_off = mx.slot_number;
				#endif
			}
		}
	}

	// else called after merge

	GPI_TRACE_RETURN_MSG(agg_status, "ready=%" PRIu8 ", tx_hint=%" PRId8, agg_status.is_ready, agg_status.tx_hint);
}

//**************************************************************************************************

void* mx_aggregate_update_decflame(void *current, unsigned int i, const void *value)
{
	GPI_TRACE_FUNCTION();

	assert(NULL != current);

	// We consider this function as not time-critical (called before a round, and likely only once).
	// So we save the effort of explicitly updating internal data and instead construct a packed
	// aggregate that we handover to mx_aggregate_merge_decflame().

	Aggregate_Decflame	*cur = current;
	Aggregate_Decflame	*upd = (cur == &aggregate[0]) ? (cur + 1) : (cur - 1);

	memset(upd->progress_flags, 0, sizeof(upd->progress_flags));
	if (i < MX_AGG_NUM_FLAGS)
		set_bit(upd->progress_flags, i);

	upd->phase = 0;
	upd->list_len = 1;

	assert(i < MX_NUM_NODES);

	// if nodes stored as list of node IDs
	if (!MX_AGG_NODES_USE_BITFIELD)
		upd->nodes[0] = i;

	// if nodes stored as bitfield
	else
	{
		memset(upd->nodes, 0, sizeof(upd->nodes));
		set_bit(upd->nodes, i);
	}

	memcpy(upd->priorities, value, (MX_AGG_NUM_BITS_PER_PRIO + 7) / 8);

	cur = mx_aggregate_merge_decflame(cur, upd, NULL);

	agg_status.tx_hint = 1;

	GPI_TRACE_RETURN(cur);
}

//**************************************************************************************************

void* mx_aggregate_merge_decflame(void *current_, const void *update_, const void *packet_)
{
	GPI_TRACE_FUNCTION();

	//assert(NULL != current);

	const uint_fast8_t			SIZEOF_FLAGS = sizeof_member(Aggregate_Decflame, progress_flags);

	Aggregate_Decflame*			cur = current_;
	const Aggregate_Decflame*	upd = update_;
	const Packet* const			packet = packet_;
	Aggregate_Decflame*			next = (cur == &aggregate[0]) ? (cur + 1) : (cur - 1);
	const uint_fast16_t			current_slot = (NULL != packet) ? packet->slot_number : mx.slot_number;
	uint_fast8_t				update_phase = upd->phase;
	int_fast16_t				progress_flag = -1;
	uint_fast8_t				progress = 0;

	// if turn-off countdown active or finished
	if (agg_status.is_ready > 1)
	{
		// determine if received update is complete (Accept phase and all progress flags set)
		if (2 == upd->phase)
		{
			// MASK selects unused bits in last byte
			const uint8_t	NBITS = MX_AGG_NUM_FLAGS % 8;
			const uint8_t	MASK = (NBITS > 0) ? (0xff << NBITS) & 0xff : 0x00;

			const uint8_t*	flags = upd->progress_flags;
			uint_fast16_t	sum = 0;
			uint_fast8_t	i;

			// instead of directly counting set bits, progress == 100% can also be
			// determined by summing up all flag bytes and compare the result to the
			// all-flags-set value (sum >= value only if all flags set)
			for (i = SIZEOF_FLAGS - 1; i-- > 0;)
				sum += *flags++;

			// last byte with explicit handling of unused bits
			sum += *flags | MASK;

			// progress = all flags set?
			progress = (sum >= SIZEOF_FLAGS * 0xff);
		}

		// if not complete
		if (!progress)
		{
			// restart turn-off countdown if countdown != forever
			if (MX_AGG_TURN_OFF_COUNTDOWN > 0)
			{
				GPI_TRACE_MSG(TRACE_VERBOSE, "incomplete aggregate received, restarting turn-off countdown");

				local.turn_off_countdown = MX_AGG_TURN_OFF_COUNTDOWN;

				#if MX_VERBOSE_STATISTICS
					statistics.slot_off = 0;
				#endif
			}

			// push response
			agg_status.tx_hint = 1;
		}

		GPI_TRACE_RETURN(cur);
	}

	// reset tx_hint == PREFER
	if (agg_status.tx_hint > 0)
		agg_status.tx_hint = 0;

//	TRACE_AGG(TRACE_VERBOSE, "aggregate current", cur);
	TRACE_AGG(TRACE_VERBOSE, "aggregate update", upd);

	// catch malicious packets
	if (upd->phase > 2)
	{
		GPI_TRACE_MSG(TRACE_WARNING, "malformed aggregate: invalid phase %" PRIu8, upd->phase);
		GPI_TRACE_RETURN(cur);
	}


	// process update stage 1 (before updating progress flags)
	switch (local.phase)
	{
		// priority exchange
		case 0:
		{
			// if update for current phase
			if (0 == update_phase)
			{
				Node_Prio_List* const	unpacked = &local.node_prio_list;
				Node_Prio_List			update;

				GPI_TRACE_MSG(TRACE_VERBOSE, "update priorities");

				// unpack update into local data structure
				agg_unpack(&update, upd);

				// DEBUG: allow node 0 to inject priorities of absent nodes
				// (e.g. to emulate larger settings with only a handful of nodes)
				#if 0
				const uint8_t NODES_ONLINE = 3;
				if (MX_AGG_NUM_FLAGS > NODES_ONLINE && 0 == my_node_id())
				{
					typeof(progress_flag)	i;

					// scan flags for progress
					for (i = 0; i < SIZEOF_FLAGS; ++i)
					{
						if (upd->progress_flags[i] & ~(cur->progress_flags[i]))
							break;
					}

					// if no progress
					if (i >= SIZEOF_FLAGS)
					{
						// select random absent node
						i =	NODES_ONLINE +
							gpi_mulu_16x16(mixer_rand(), MX_AGG_NUM_FLAGS - NODES_ONLINE) >> 16;

						// inject priority if not seen (i.e. likely not injected) before
						if (!(cur->progress_flags[i / 8] & gpi_slu(UINT8_C(1), i % 8)))
						{
							Node_Prio_Entry	np;

							// ATTENTION: do not touch *upd, use *next instead
							memset(next->progress_flags, 0, SIZEOF_FLAGS);
							set_bit(next->progress_flags, i);
							upd = next;

							// ATTENTION: injection can happen multiple times per node,
							// so using random prio can lead to differing entries.
							// Use fixed value if this is an issue.
							np.node = i;
							np.prio = mixer_rand();

							GPI_TRACE_MSG(TRACE_INFO, "DEBUG: injecting node %u prio %u",
								(int)np.node, (int)np.prio);

							update.list_len = 1;
							update.node_prio_list[0] = np;
							update.prio_node_list[0].node = np.node;
							update.prio_node_list[0].prio = np.prio;
						}
					}
				}
				#endif

				// merge unpacked data with current aggregate
				// NOTE: we can touch unpacked local data (we do not touch *cur)
				for (uint_fast8_t i = 0; i < update.list_len; ++i)
				{
					Prio_Node_Entry	prio_node = update.prio_node_list[i];
					uint_fast8_t	kn, kp;

					// NOTE: instead of using list_search(node_prio_list, list_len, node)
					// we test the corresponding progress flag (which is faster), as the latter provides
					// equivalent information: If it is not set, then entry is not in list for sure.
					// If it is set, then entry may be in list or not (depending on prio > min. prio),
					// but there is no need to consider it again.
					// Note that we do not update the progress flags immediately, so if update contains
					// duplicate entries (i.e. it is malformed) then merged result can also do.
					kn = prio_node.node;
					uint_fast8_t is_in_list = cur->progress_flags[kn / 8] & gpi_slu(UINT8_C(1), kn % 8);

					// if (list is not full or prio > min. prio) and node not already in list
					if ((unpacked->list_len < NUM_ELEMENTS(unpacked->prio_node_list) ||
						prio_node.raw > unpacked->prio_node_list[0].raw) &&
						!is_in_list)
					{
						// if list is full: replace min. prio entry
						if (unpacked->list_len >= NUM_ELEMENTS(unpacked->prio_node_list))
						{
							kp = 0;
							kn = list_search(unpacked->node_prio_list, unpacked->list_len, unpacked->prio_node_list[0].node);
						}

						// else add entry
						else kp = kn = unpacked->list_len++;

						// update entry and resort list(s)
						unpacked->prio_node_list[kp] = prio_node;
						unpacked->node_prio_list[kn].node = prio_node.node;
						unpacked->node_prio_list[kn].prio = prio_node.prio;
						list_update(unpacked->prio_node_list, unpacked->list_len, kp);
						list_update(unpacked->node_prio_list, unpacked->list_len, kn);
					}
				}

				// pack updated local data
				next->phase = 0;
				agg_pack(next, unpacked);

				// set own progress flag (processed below)
				progress_flag = my_node_id();

				break;
			}

			// adopt phase update
			else
			{
				GPI_TRACE_MSG(TRACE_VERBOSE, "switch phase prio -> Paxos");

				#if MX_VERBOSE_STATISTICS
					statistics.slot_paxos_start = current_slot;
				#endif

				// NOTE: we know that our node-prio-list is not from a full aggregate
				// (called optimal in the following), otherwise we would not be here

				// if received proposal has access to optimal value
				if (upd->proposal >= MX_NUM_NODES)
				{
					// switch to next phase (Paxos)
					local.phase = 1;

					// init proposer logic, do not propose an own value
					// as the received proposal is a better option
					local.proposed_value.list_len = 0;
					local.proposal = my_node_id();
					local.proposer_state = -1;
					#if MX_VERBOSE_STATISTICS
						statistics.slot_proposer_lost = current_slot;
					#endif

					// init acceptor logic, prepare for received proposal
					local.accepted_proposal = -1;
					local.min_proposal = upd->proposal;

					// init aggregate values
					local.max_accepted_proposal = -1;
					local.max_observed_proposal = local.min_proposal;
//					local.learned_value.list_len = 0;

					// init corresponding current packet for processing below
					next->phase = local.phase;
					next->list_len = -1;
					next->proposal = local.proposal;
					cur = next;

					// skip initialization of flags as they will be reinitialized below for sure
					// (as cur->proposal < MX_NUM_NODES <= upd->proposal)
				}

				else
				{
					// NOTE: Being here (with received non-optimal proposal) means that current slot
					// is beyond MX_AGG_MAX_NUM_SLOTS_0. Hence, we could just ignore the packet
					// because we will switch to Paxos phase anyway below. However, we process
					// it because we do not want to waste it.

					// switch to next phase (Paxos)
					local.phase = 1;

					// store last packed data as our proposed value
					memcpy(&local.proposed_value.nodes, cur->nodes, sizeof(cur->nodes));
					local.proposed_value.list_len = cur->list_len;

					// init proposer logic
					local.proposal = my_node_id();
					local.proposer_state = 0;

					// init acceptor logic, prepare for own proposal
					local.accepted_proposal = -1;
					local.min_proposal = local.proposal;

					// init aggregate values
					local.max_accepted_proposal = -1;
					local.max_observed_proposal = local.min_proposal;
//					local.learned_value.list_len = 0;

					// init corresponding current packet for processing below
					next->phase = local.phase;
					next->list_len = -1;
					next->proposal = local.proposal;
					cur = next;

					// skip initialization of flags if they will be reinitialized below
					if (!(upd->proposal > local.proposal))
					{
						ASSERT_CT(MX_AGG_NUM_FLAGS >= MX_NUM_NODES);
						memset(next->progress_flags, 0, sizeof(next->progress_flags));
						set_bit(next->progress_flags, my_node_id());
						progress = 1;	// for MX_VERBOSE_AGGREGATE, see below
					}
				}

				// don't break, continue processing
 			}
		}

		// Paxos
		case 1:
		case 2:
		{
			uint_fast8_t is_update = (update_phase >= 1) && (
				(upd->proposal > cur->proposal) ||
				(upd->proposal == cur->proposal && upd->phase >= cur->phase));

			// ignore outdated aggregates and old proposals
			if (!is_update)
			{
				// NOTE: we could simply set next = cur instead of memcpy()
				// because *next will not be touched below (which must be
				// guaranteed for *cur) in current phase and without providing
				// updates. However we do not use this optimization to avoid potential
				// mistakes in case something changes.
				memcpy(next, cur, sizeof(*next));
				cur = NULL;		// avoids updating progress flags below
				agg_status.tx_hint = 1;
				break;
			}

			GPI_TRACE_MSG(TRACE_VERBOSE, "update Paxos");

			// new phase = new proposal received or current proposal transitioned from prepare to accept
			uint_fast8_t is_new_phase =
				!(upd->proposal == cur->proposal && upd->phase == cur->phase);

			// inherit packet content (except for progress flags)
			ASSERT_CT(0 == offsetof(Aggregate_Decflame, progress_flags));
			memcpy(
				(uint8_t*)next + SIZEOF_FLAGS,
				(const uint8_t*)upd + SIZEOF_FLAGS,
				offsetof(Aggregate_Decflame, paxos_end) - SIZEOF_FLAGS);

			// if new phase: reinit progress flags and local state
			if (is_new_phase)
			{
				memset(next->progress_flags, 0, SIZEOF_FLAGS);
				cur = next;

				local.max_accepted_proposal = -1;
				local.max_observed_proposal = 0;
			}

			// acceptor logic
			{
				// prepare
				if (1 == update_phase)
				{
					int_fast16_t map =
						(upd->list_len > MX_AGG_MAX_LIST_LEN) ? -1 : upd->max_accepted_proposal;

					// Paxos: update minProposal
					if (upd->proposal > local.min_proposal)
					{
						GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos acceptor min_proposal %" PRIu8 " -> %" PRIu8,
							local.min_proposal, upd->proposal);

						local.min_proposal = upd->proposal;
					}

					// Paxos: report last acceptedProposal of current acceptor (if any)
					// Wireless Paxos: report max. acceptedProposal of all (heard) acceptors
					if (map < local.max_accepted_proposal)
					{
						next->max_accepted_proposal = local.max_accepted_proposal;
						memcpy(next->nodes, local.max_accepted_value.nodes, sizeof(next->nodes));
						next->list_len = local.max_accepted_value.list_len;

						agg_status.tx_hint = 1;
					}
					else if (map > local.max_accepted_proposal)
					{
						GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos max_accepted_proposal %" PRId16 " -> %" PRIdFAST16,
							local.max_accepted_proposal, map);

						local.max_accepted_proposal = map;
						memcpy(local.max_accepted_value.nodes, upd->nodes, sizeof(upd->nodes));
						local.max_accepted_value.list_len = upd->list_len;
					}
				}

				// accept
				else
				{
					// Paxos: accept if proposal >= minProposal
					if (upd->proposal >= local.min_proposal)
					{
						local.min_proposal = upd->proposal;

						// optimization: accept proposal only once to avoid unnecessary copy operations.
						// this is possible because proposer is not allowed to change proposed value
						// during accept phase
						if (upd->proposal > local.accepted_proposal)
						{
							GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos accept %" PRIu8, upd->proposal);

							local.accepted_proposal = upd->proposal;
							memcpy(local.accepted_value.nodes, upd->nodes, sizeof(upd->nodes));
							local.accepted_value.list_len = upd->list_len;
						}
					}

					// update max. acceptedProposal
					if (local.max_accepted_proposal < local.accepted_proposal)
					{
						local.max_accepted_proposal = local.accepted_proposal;
						memcpy(&local.max_accepted_value, &local.accepted_value, sizeof(local.accepted_value));
					}

					// Wireless Paxos: report max. minProposal heard so far
					if (local.max_observed_proposal < local.min_proposal)
						local.max_observed_proposal = local.min_proposal;
					if (local.max_observed_proposal < upd->max_observed_proposal)
						local.max_observed_proposal = upd->max_observed_proposal;
					next->max_observed_proposal = local.max_observed_proposal;

					if (next->max_observed_proposal > upd->max_observed_proposal)
						agg_status.tx_hint = 1;
				}

				// set own progress flag (processed below)
				progress_flag = my_node_id();

				// DEBUG: allow node 0 to set progress flags of absent nodes
				// (e.g. to emulate larger settings with only a handful of nodes)
				#if 0
				const uint8_t NODES_ONLINE = 3;
				if (MX_AGG_NUM_FLAGS > NODES_ONLINE)
				{
					// if node 0 and own flag already set
					if (0 == progress_flag && next->progress_flags[0] & 0x01)
					{
						// set flag of a randomly selected absent node
						progress_flag =
							NODES_ONLINE +
							gpi_mulu_16x16(mixer_rand(), MX_AGG_NUM_FLAGS - NODES_ONLINE) >> 16;
					}
				}
				#endif
			}

			// proposer logic
			if (0 == local.proposer_state)
			{
				// if update refers to own proposal
				if (upd->proposal == local.proposal)
				{
					// safety check: nobody else must update phase for our proposal
					// NOTE: we can receive (outdated) packets from prior phase
					assert(update_phase <= local.phase);

					// if packet from current phase
					if (update_phase == local.phase)
					{
						// prepare phase
						if (1 == local.phase)
						{
							// safety check.
							// opposite cannot happen (theoretically) because acceptor must not
							// answer prepare requests for proposals < minProposal, i.e.
							// receiving an answer implies proposal >= minProposal >= acceptedProposal
							assert(local.proposal >= local.max_accepted_proposal);
						}

						// accept phase
						else
						{
							// Paxos: give up if acceptors report higher minProposal
							// (i.e. a higher proposal has been issued by another proposer)
							if (local.max_observed_proposal > local.proposal)
							{
								GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos proposer lost (%" PRIu8 " < %" PRIu8 ")",
									local.proposal, local.max_observed_proposal);

								local.proposer_state = -1;
							}
						}
					}
				}

				// if higher proposal existing: give up
				else if (upd->proposal > local.proposal)
				{
					GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos proposer lost (%" PRIu8 " < %" PRIu8 ")",
						local.proposal, upd->proposal);

					local.proposer_state = -1;
				}

				// if lower proposal: inform about own proposal
				else agg_status.tx_hint = 1;

				#if MX_VERBOSE_STATISTICS
					if (local.proposer_state < 0)
						statistics.slot_proposer_lost = current_slot;
				#endif
			}

			break;
		}

		default:
			assert(0);
	}


	// update progress flags
	if (NULL != cur)
	{
		const uint8_t	*src_flags = cur->progress_flags;
		const uint8_t	*upd_flags = upd->progress_flags;
		uint8_t			*dst_flags = next->progress_flags;
		uint_fast8_t	tx_hint, i;

		// count number of set flags
		progress = 0;
		tx_hint = 0;

		// merge local and update flags
		if (NULL != upd)
		{
			for (i = SIZEOF_FLAGS - 1; i-- > 0;)
			{
				tx_hint |= *src_flags & ~*upd_flags;
				*dst_flags = *src_flags++ | *upd_flags++;
				progress += gpi_popcnt(*dst_flags++);
			}

			// mask unused bits in last byte
			{
				const uint8_t MASK = 0xff >> ((8 - (MX_AGG_NUM_FLAGS % 8)) % 8);
				tx_hint |= (*src_flags & ~*upd_flags) & MASK;
				*dst_flags = (*src_flags | *upd_flags) & MASK;
				progress += gpi_popcnt(*dst_flags);
			}
		}

		// copy local flags
		else
		{
			for (i = SIZEOF_FLAGS; i-- > 0;)
			{
				*dst_flags = *src_flags++;
				progress += gpi_popcnt(*dst_flags++);
			}
		}

		// set specified progress flag if requested
		if (progress_flag >= 0)
		{
			if (progress_flag >= MX_AGG_NUM_FLAGS)
			{
				GPI_TRACE_MSG(TRACE_ERROR, "!!! progress_flag exceeds MX_AGG_NUM_FLAGS -> check program, must not happen !!!");
				assert(0);
			}
			else
			{
				dst_flags = &(next->progress_flags[progress_flag / 8]);
				i = gpi_slu(UINT8_C(1), progress_flag % 8);

				if (!(*dst_flags & i))
				{
					*dst_flags |= i;

					progress += 1;

					if (NULL != upd)
						tx_hint = 1;
				}
			}
		}

		if (tx_hint)
			agg_status.tx_hint = 1;

		// NOTE: progress == 100% could also be determined by summing up all flag bytes
		// and compare the result to all-flags-set value
	}


	// process stage 2 (after updating progress flags),
	// switch to next phase when possible
	switch (local.phase)
	{
		// priority exchange
		case 0:
		{
			// if list is complete or timeout reached: switch to next phase
			if (progress >= MX_AGG_NUM_FLAGS || current_slot >= MX_AGG_MAX_NUM_SLOTS_0)
			{
				GPI_TRACE_MSG(TRACE_VERBOSE, "switch phase prio -> Paxos");

				#if MX_VERBOSE_STATISTICS
					statistics.slot_paxos_start = current_slot;
				#endif

				// store packed data as proposed value
				memcpy(&local.proposed_value.nodes, next->nodes, sizeof(next->nodes));
				local.proposed_value.list_len = next->list_len;

				// switch phase
				local.phase = 1;

				// init proposer logic
				// NOTE: If progress == MX_AGG_NUM_FLAGS, then our value is optimal.
				// In this case we select a higher proposal number (namely node_id + MX_NUM_NODES)
				// to increase the chance that the optimal value wins the race
				// (however note that the latter is not guaranteed, as the proposer
				// may have to adopt another previously accepted value).
				// As a side effect, a receiver in phase 0 can deduce that there is
				// no need to stay in phase 0 any longer.
				// TODO: for now we limit MX_NUM_NODES to 128 so that node_id + MX_NUM_NODES
				// fits into 8 bit proposal numbers. It is easy to extend this if necessary
				// (in case, use bitfields in Aggregate_Decflame to not waste more than needed).
				ASSERT_CT(MX_NUM_NODES <= 128);
				local.proposal = my_node_id() + ((progress >= MX_AGG_NUM_FLAGS) ? MX_NUM_NODES : 0);
				local.proposer_state = 0;

				// init acceptor logic, prepare for own proposal
				local.accepted_proposal = -1;
				local.min_proposal = local.proposal;

				// init aggregate values
				local.max_accepted_proposal = -1;
				local.max_observed_proposal = local.proposal;
//				local.learned_value.list_len = 0;

				// update packet
				next->phase = local.phase;
				next->list_len = -1;
				next->proposal = local.proposal;

				// reinit flags
				ASSERT_CT(MX_AGG_NUM_FLAGS >= MX_NUM_NODES);
				memset(next->progress_flags, 0, sizeof(next->progress_flags));
				set_bit(next->progress_flags, my_node_id());
				progress = 1;	// for MX_VERBOSE_AGGREGATE, see below

				agg_status.tx_hint = 1;
			}

			break;
		}

		// Paxos
		case 1:
		case 2:
		{
			// if accept packet
			if (2 == next->phase)
			{
				// learn value when acceptance of a proposal reached majority
				// NOTE: We can learn the value from any proposal that reached majority,
				// it does not need to be the highest proposal (which will eventually win).
				// The reason is that Paxos ensures that any higher proposal must adopt
				// the value of a lower proposal if the latter reached majority at some
				// point, so eventually the values of all proposals (temporarily) reaching
				// majority are the same.
				if (progress > MX_AGG_NUM_FLAGS / 2 && !agg_status.is_ready)
				{
					GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos learn value");

					// NOTE: we can reuse local.proposed_value instead of an explicit variable
					// local.learned_value because
					// * if local.proposal < next->proposal then local proposer
					//   gave up already (handled above)
					// * if local.proposal == next->proposal then
					//   proposed value == next->value anyway
					// * if local.proposal > next->proposal then local proposer
					//   has or will adopt same value as ensured by Paxos
					assert(local.proposal >= next->proposal || 0 != local.proposer_state);
					memcpy(local.proposed_value.nodes, next->nodes, sizeof(next->nodes));
					local.proposed_value.list_len = next->list_len;

					agg_status.is_ready = 1;

					#if MX_VERBOSE_STATISTICS
						statistics.slot_value_learned = current_slot;
					#endif
				}

				// if aggregate complete (all flags set)
				// NOTE: all flags set also implies proposal == max_observed_proposal,
				// because otherwise the node proposing max_observed_proposal would not
				// have accepted proposal (as it accepted max_observed_proposal internally
				// right at the beginning)
				if (MX_AGG_NUM_FLAGS == progress)
				{
					GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos completed");

					// use countdown to retransmit packet a few times before turning aggregation off
					// (MX_AGG_TURN_OFF_COUNTDOWN <= 0 means countdown = forever)
					local.turn_off_countdown = MX_AGG_TURN_OFF_COUNTDOWN;
					agg_status.is_ready = 2;
					agg_status.tx_hint = 1;

					#if MX_VERBOSE_STATISTICS
						statistics.slot_complete = current_slot;
					#endif
				}
			}

			// if proposer not active anymore: done
			if (0 != local.proposer_state)
				break;

			// if packet refers to own proposal and current phase
			if (next->proposal == local.proposal && next->phase == local.phase)
			{
				// if majority reached: switch to next phase
				if (progress > MX_AGG_NUM_FLAGS / 2)
				{
					GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos majority reached");

					// prepare -> accept
					if (1 == local.phase)
					{
						GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos proposer prepare -> accept");

						#if MX_VERBOSE_STATISTICS
							statistics.slot_proposer_accept = current_slot;
						#endif

						// Paxos: adopt max. acceptedProposal's value
						// NOTE: local.max_accepted_proposal has been updated
						// by the acceptor logic beforehand (see above)
						if (local.max_accepted_proposal >= 0)
						{
							GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos proposer adopt value");
							memcpy(&local.proposed_value, &local.max_accepted_value, sizeof(local.proposed_value));
						}

						local.phase = 2;
						local.max_accepted_proposal = -1;
						local.max_observed_proposal = 0;

						// update packet

						next->phase = local.phase;

						memcpy(next->nodes, local.proposed_value.nodes, sizeof(next->nodes));
						next->list_len = local.proposed_value.list_len;

						memset(next->progress_flags, 0, SIZEOF_FLAGS);
						set_bit(next->progress_flags, my_node_id());
						progress = 1;	// for MX_VERBOSE_AGGREGATE, see below

						// let own acceptor accept immediately

						// Paxos: accept if proposal >= minProposal
						if (next->proposal >= local.min_proposal)
						{
							local.min_proposal = next->proposal;
							local.accepted_proposal = next->proposal;
							memcpy(local.accepted_value.nodes, next->nodes, sizeof(next->nodes));
							local.accepted_value.list_len = next->list_len;
						}

						// update max. acceptedProposal
						if (local.max_accepted_proposal < local.accepted_proposal)
						{
							local.max_accepted_proposal = local.accepted_proposal;
							memcpy(&local.max_accepted_value, &local.accepted_value, sizeof(local.accepted_value));
						}

						// Wireless Paxos: report max. minProposal heard so far
						if (local.max_observed_proposal < local.min_proposal)
							local.max_observed_proposal = local.min_proposal;
						next->max_observed_proposal = local.max_observed_proposal;

						agg_status.tx_hint = 1;
					}

					// accept -> proposer done (got majority of acceptors)
					else
					{
						GPI_TRACE_MSG(TRACE_VERBOSE, "Paxos proposer accept -> done");

						local.proposer_state = 1;

						#if MX_VERBOSE_STATISTICS
							statistics.slot_proposer_done = current_slot;
						#endif
					}
				}
			}

			break;
		}

		default:
			assert(0);
	}


	TRACE_AGG(TRACE_VERBOSE, "aggregate new", next);
	GPI_TRACE_MSG(TRACE_VERBOSE, "aggregate status %" PRIu8 " / %" PRId8,
		agg_status.is_ready, agg_status.tx_hint);

	// print aggregate in compact log-parser-friendly format
	#if MX_VERBOSE_AGGREGATE
	{
		static typeof(progress)	last_progress;

		// if progress has not been determined for current packet
		// (i.e. when *next = *cur; cur = NULL): take previous value
		if (0 == progress)
			progress = last_progress;
		else last_progress = progress;

		// line format:
		// # MXA<fmt (%x)>: <slot (%04x)> <phase+agg (%03x)> <progress (%02x)>:<progress flags (%x)> /
		// <tx_phase_data> <local_phase_data>
		//
		// <fmt> = markers regarding the log format, 1 = this one (DecFLAME)
		//
		// <phase+agg>:
		//	0x000...0x3FF: phase = 0, 0b0000_00xx_xxxx = list length
		//  0x400...0x7FF: phase = 1, 0b00xx_xxxx_xxxx = proposal
		//	0x800...0xBFF: phase = 2, 0b00xx_xxxx_xxxx = proposal
		//
		// <progress> = progress of current phase (number of set progress flags)
		//
		// <tx_phase_data> in phase 0:
		// <node (%02x)>:<prio (%02x)> for each list entry
		//
		// <tx_phase_data> in phase 1:
		// <max_accepted_proposal (%03x)> <value (%x)>
		//
		// <tx_phase_data> in phase 2:
		// <max_observed_proposal (%03x)> <value (%x)>
		//
		// <local_phase_data> in phase 1+2:
		// < min_proposal (%03x)> <accepted_proposal (%03x)> <max_observed_proposal> <max_accepted_proposal>
		//
		// proposal numbers: -1 (== all bits 1) == none

		char	msg[
					7 + 5 + 4 + 4
					+ 2*sizeof(next->progress_flags)
					+ 2
					+ MAX(
						MAX(6*MX_AGG_MAX_LIST_LEN, 2 + 2*sizeof(next->nodes) + 2*sizeof(next->priorities)),
						5 + 2*sizeof(next->nodes))
					+ 4*4
					+ 1];
		char*	ps = msg;
		int		i;

		// <slot> <tx phase, list len | proposal> <tx progress>
		// NOTE: <slot> is the rx slot of the update, not the tx slot of the aggregate printed here.
		// The latter can be determined, e.g., via MX_VERBOSE_PACKETS.
		ps += sprintf(ps, "# MXA1: %04x %03x %02x:",
			(int)current_slot,
			((int)(next->phase) << 10) | ((0 == next->phase) ? next->list_len : next->proposal),
			(int)progress);

		// <tx progress flags>
		for (i = 0; i < sizeof(next->progress_flags); ++i)
			ps += sprintf(ps, "%02x", (int)(next->progress_flags[i]));

		ps += sprintf(ps, " /");

		if (0 == next->phase)
		{
			#if 1
				Node_Prio_List	npl;
				agg_unpack(&npl, next);

				// <tx node:prio>+
				for (i = 0; i < npl.list_len; ++i)
					ps += sprintf(ps, " %02x:%02x",
						(int)npl.node_prio_list[i].node,
						(int)npl.node_prio_list[i].prio);
			#else
				// <tx nodes>
				ps += sprintf(ps, " ");
				for (i = 0; i < sizeof(next->nodes); ++i)
					ps += sprintf(ps, "%02x", (int)(next->nodes[i]));

				// <tx priorities>
				ps += sprintf(ps, " ");
				for (i = 0; i < sizeof(next->priorities); ++i)
					ps += sprintf(ps, "%02x", (int)(next->priorities[i]));
			#endif
		}

		else
		{
			if (1 == next->phase)
				i = (next->list_len > MX_AGG_MAX_LIST_LEN) ? -1 : next->max_accepted_proposal;
			else i = next->max_observed_proposal;

			// <tx max. accepted | observed proposal>
			ps += sprintf(ps, " %03x ", i & 0xfff);

			// <tx value>
			for (i = 0; i < sizeof(next->nodes); ++i)
				ps += sprintf(ps, "%02x", (int)(next->nodes[i]));
		}

		// < min. proposal> <accepted proposal> <max. observed proposal> <max. accepted proposal>
		if (local.phase > 0)
			ps += sprintf(ps, " %03x %03x %03x %03x",
				(int)local.min_proposal,
				(int)local.accepted_proposal & 0xfff,
				(int)local.max_accepted_proposal & 0xfff,
				(int)local.max_observed_proposal);

		assert(ps < &msg[sizeof(msg)]);

		puts(msg);
	}
	#endif

	GPI_TRACE_RETURN(next);
}

//**************************************************************************************************

const void* mx_aggregate_read_decflame(void *current)
{
	GPI_TRACE_FUNCTION();

	// post-process data if not already done
	// (the latter is marked by local.phase < 0)
	if (local.phase >= 0)
	{
		Aggregate_Decflame *cur = current;

		local.phase = -1;

		if (!agg_status.is_ready)
			local.proposed_value.list_len = 0;

		// unpack learned value
		// NOTE: using agg_unpack() with dummy priorities (instead of just unpacking the node list)
		// is a bit inefficient. However, since performance is not that critical here we prefer to
		// reuse agg_unpack() and avoid case distinctions inside it (performance inside agg_unpack()
		// is more important than here).
		cur->phase = 0;
		cur->list_len = local.proposed_value.list_len;
		memcpy(cur->nodes, local.proposed_value.nodes, sizeof(cur->nodes));
		// cur->priorities don't care here
		agg_unpack(&local.node_prio_list, cur);

		// convert node_prio_list into node list

		Node_Prio_Entry	*src = &local.node_prio_list.node_prio_list[0];
		uint8_t			*dst = &local.result.nodes[0];
		uint_fast8_t	i;

		// ATTENTION: src and dst can overlap in local, so make sure that following loop
		// does not damage anything. Compare ...[1] instead of ...[0] to allow to have
		// (small) members before result.nodes (e.g. list_len).
		ASSERT_CT(&local.result.nodes[1] <= (uint8_t*)&local.node_prio_list.node_prio_list[1]);

		for (i = 0; i < local.node_prio_list.list_len; ++i)
			*dst++ = (*src++).node;

		// ATTENTION: local.node_prio_list.list_len may be invalid now, so use i instead
		local.result.list_len = i;
	}

	GPI_TRACE_RETURN(&local.result);
}

//**************************************************************************************************

void mx_aggregate_print_statistics_decflame()
{
	#if MX_VERBOSE_STATISTICS

		#ifdef PRINT
			#error change macro name
		#endif

		// _Generic() inside string concatenation causes problems,
		// so instead we concatenate inside Generic()
		#define PRINT2(n, a, b)	\
			_Generic(statistics.n, uint8_t: a PRIu8 b, uint16_t: a PRIu16 b, uint32_t: a PRIu32 b, default: a "%" b)
		#define PRINT(n)	\
			printf(PRINT2(n, "aggregate." #n ": %", "\n"), statistics.n)

		PRINT(slot_paxos_start);
		PRINT(slot_proposer_lost);
		PRINT(slot_proposer_accept);
		PRINT(slot_proposer_done);
		PRINT(slot_value_learned);
		PRINT(slot_complete);
		PRINT(slot_off);

		#undef PRINT
		#undef PRINT2

	#endif
}

//**************************************************************************************************
//**************************************************************************************************

#endif	// MX_AGGREGATE
