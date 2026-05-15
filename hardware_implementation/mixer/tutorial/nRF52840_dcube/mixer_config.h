#ifndef __MIXER_CONFIG_H__
#define __MIXER_CONFIG_H__

// mixer configuration file
// Adapt the settings to the needs of your application.

#include "gpi/platform_spec.h"		// GPI_ARCH_IS_...
#include "gpi/tools.h"				// NUM_ELEMENTS()
#include "testbed_graz.h"

ASSERT_CT_STATIC(GPI_ARCH_IS_BOARD(nRF_PCA10056));
#define PRINT_HEADER()
#define NODE_ID			cfg.node_id
#define NODE_ID_INIT	volatile config_t __attribute__((section(".testbedConfigSection"))) cfg = { .node_id = 1 }

// all nodes GRAZ
static const uint8_t nodes[] =
	{	100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
		116, 117, 118, 119, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
		212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227 };
static const uint8_t payload_distribution[] =
	{	100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
		116, 117, 118, 119, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
		212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227 };

/*****************************************************************************/
/* basic settings ************************************************************/

#define MX_ROUND_LENGTH 		2000 // in #slots
#define MX_PAYLOAD_SIZE			18 // +2 because of senderID
#define MX_SLOT_LENGTH			GPI_TICK_US_TO_HYBRID2(800)
#define MX_NUM_NODES			NUM_ELEMENTS(nodes)
#define MX_INITIATOR_ID			payload_distribution[0]
#define MX_GENERATION_SIZE		NUM_ELEMENTS(payload_distribution)

// Possible values (Gpi_Radio_Mode):
//		IEEE_802_15_4	= 1
//		BLE_1M			= 2
//		BLE_2M			= 3
//		BLE_125k		= 4
//		BLE_500k		= 5
#define MX_PHY_MODE				3
// Values mentioned in the manual (nRF52840_PS_v1.1):
// +8dBm,  +7dBm,  +6dBm,  +5dBm,  +4dBm,  +3dBm, + 2dBm,
//  0dBm,  -4dBm,  -8dBm, -12dBm, -16dBm, -20dBm, -40dBm
#define MX_TX_PWR_DBM			8


/*****************************************************************************/
/* special settings **********************************************************/

#define MX_WEAK_ZEROS			0
#define WEAK_RELEASE_SLOT		1
#define MX_WARMSTART			0
#define WARMSTART_HISTORY		0
#define MX_REQUEST				1
#define MX_REQUEST_HEURISTIC	2

// turn verbose log messages on or off
// NOTE: These additional prints might take too long when using short round intervals.
#define MX_VERBOSE_STATISTICS	1
#define MX_VERBOSE_PACKETS		0
#define MX_VERBOSE_PROFILE		0

#define MX_SMART_SHUTDOWN		1
// 0	no smart shutdown
// 1	no unfinished neighbor, without full-rank map(s)
// 2	no unfinished neighbor
// 3	all nodes full rank
// 4	all nodes full rank, all neighbors ACKed knowledge of this fact
// 5	all nodes full rank, all nodes ACKed knowledge of this fact
#define MX_SMART_SHUTDOWN_MODE	2

#endif // __MIXER_CONFIG_H__
