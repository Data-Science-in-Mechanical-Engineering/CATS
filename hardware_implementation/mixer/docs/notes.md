# General

- TRACE:
    - 32bit mask defines which groups should be traced (upper bits are already predefined, lower bits (GPI_TRACE_LOG_USER) can be used for user groups)
    - GPI_TRACE_RETURN(...) replaces return statement and can be used to print function exit and (elementary) return values
    - GPI_TRACE_*() immediately flushes output
    - GPI_TRACE_*_FAST() uses and internal print buffer and allows to flush later (timestamp is created vi a delayed service routine (DSR)) -> should be used to print in ISRs
    - use printf() for the protocols log output since trace msgs can be deactivated
- nodes array contains physical node ids and logical node id (node_id) is mapped from left to right starting at 0
- output:
    // trace_packet output:
    slot   send.  fl.  IV   CV   payload
    0096 - 800e - 80 - a0 - 01 - 0000010300000000...

    // mx_trac_dump "Rx packet" output:
    slot   send.  fl.  CV   payload              IV
    9600   0e     80   01   0000010300000000 ... a0

# GPI internals

- gpi_tick_X_extended():
    - sw extension that simply counts higher than the actual timer width (e.g. 32 bit instead of 16 bit for MSP430) -> has to be called periodically in order to avoid missing overruns
- includes are processed from the most specific part (platform) to the most general part (cpu family)
- gpi_tick_hybrid_reference() returns a reference time of the near past where the relation between slow and fast clock becomes clear (do not use this to schedule events)
    - hybrid clock combines slow and fast clock to provide an efficient clock (better resolution than slow clock but can be used more energy efficient than fast clock)
    - fast ticks between slow ticks are counted and both are combined into a hybrid clock
- gpi_tick_hybrid() computes the hybrid time for the current time of the fast timer

# MSP430

- GPI_SLOW_CLOCK_RATE 32KHz Timer A (ACLK)
- GPI_FAST_CLOCK_RATE 4MHz Timer B (SMCLK)

# History

- activated when MX_COORDINATED_TX set
- initially all nodes chained together in absent list
- sentinel nodes mark the beginning of different lists and provide information about heard (present), unknown (absent) and finished nodes (additionally acked nodes if MX_SMART_SHUTDOWN >= 4)
