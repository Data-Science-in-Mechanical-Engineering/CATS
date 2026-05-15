#ifndef __MIXER_CONFIG_H__
#define __MIXER_CONFIG_H__

// mixer configuration file
// Adapt the settings to the needs of your application.

#include "gpi/platform_spec.h"		// GPI_ARCH_IS_...
#include "gpi/tools.h"				// NUM_ELEMENTS()
#include "gpi/trace.h"				// GPI_TRACE_MODE_...

// tiny test on developer's desk
static const uint8_t nodes[] = { 1, 2, 3};
static const uint8_t payload_distribution[] = {  1,  2,  1,  2,  1,  2,  1,  2};

/*****************************************************************************/
/* basic settings **********************************************************/

#define MX_ROUND_LENGTH			50 // in slots
#define MX_NUM_NODES			NUM_ELEMENTS(nodes)
#define MX_GENERATION_SIZE		NUM_ELEMENTS(payload_distribution)
#define MX_PAYLOAD_SIZE			16
#define INITIATOR_ID			payload_distribution[0]		// used in main.c (not inside Mixer)

// choose a slot length according to your settings
// NOTE: Measurement unit is hybrid ticks.
// For the tutorial project we use quiet long slots to be able to trace all messages
// on the UART-based console.
#if GPI_TRACE_MODE_IS_TRACE
	#define MX_SLOT_LENGTH		GPI_TICK_US_TO_HYBRID2(500000)
#else
	#define MX_SLOT_LENGTH		GPI_TICK_US_TO_HYBRID2(2000)
#endif

// Possible values (Gpi_Radio_Mode):
//		IEEE_802_15_4	= 1
//		BLE_1M			= 2
//		BLE_2M			= 3
//		BLE_125k		= 4
//		BLE_500k		= 5
#define MX_PHY_MODE				2

// Values mentioned in the manual (nRF52840_PS_v1.1):
// +8dBm,  +7dBm,  +6dBm,  +5dBm,  +4dBm,  +3dBm, + 2dBm,
//  0dBm,  -4dBm,  -8dBm, -12dBm, -16dBm, -20dBm, -40dBm
#define MX_TX_PWR_DBM			8

/*****************************************************************************/
/* special settings **********************************************************/

#define MX_WEAK_ZEROS			0
#define WEAK_RELEASE_SLOT		1	// used in main.c (not inside Mixer)
#define MX_WARMSTART			0
#define MX_WARMSTART_ROUNDS_TO_SURVIVE		1
#define MX_REQUEST				1
#define MX_REQUEST_HEURISTIC	2

#define MX_SMART_SHUTDOWN		1
// 0	no smart shutdown
// 1	no unfinished neighbor, without full-rank map(s)
// 2	no unfinished neighbor
// 3	all nodes full rank
// 4	all nodes full rank, all neighbors ACKed knowledge of this fact
// 5	all nodes full rank, all nodes ACKed knowledge of this fact
#define MX_SMART_SHUTDOWN_MODE	2

#define MX_AGGREGATE			1
#define MX_AGGREGATE_INCLUDE	mixer_aggregate_decflame.h
#define MX_AGGREGATE_CONFIG		MX_AGGREGATE_CONFIG_DECFLAME(MX_NUM_NODES, 3, 5, 10, 3)
//#define MX_AGGREGATE_CONFIG		MX_AGGREGATE_MULTIPLEX2(MX_AGGREGATE_CONFIG_MAX(3, uint8_t), MX_AGGREGATE_CONFIG_MAX(5, int16_t))
//#define MX_AGGREGATE_CONFIG		MX_AGGREGATE_MULTIPLEX2(MX_AGGREGATE_CONFIG_MAX(MX_NUM_NODES, uint8_t), MX_AGGREGATE_CONFIG_MAX(MX_NUM_NODES, int16_t))
//#define MX_AGGREGATE_CONFIG		MX_AGGREGATE_CONFIG_MAX(MX_NUM_NODES, uint8_t)

// turn verbose log messages on or off
#define MX_VERBOSE_STATISTICS	1
#define MX_VERBOSE_PACKETS		0
#define MX_VERBOSE_AGGREGATE	0
#define MX_VERBOSE_PROFILE		0

#endif // __MIXER_CONFIG_H__
