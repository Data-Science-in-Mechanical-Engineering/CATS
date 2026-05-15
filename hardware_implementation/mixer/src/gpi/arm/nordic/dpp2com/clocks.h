
// functionality is not board specific -> provide simple wrapper

// activate periodic LFRC calibration (by default)
#ifndef GPI_ARM_NRF_LFRC_CALIBRATION_CYCLE
	#define GPI_ARM_NRF_LFRC_CALIBRATION_CYCLE	4	// in 0.25s steps, 0 = disabled
#endif

#include "../nrf528xx/clocks.h"
