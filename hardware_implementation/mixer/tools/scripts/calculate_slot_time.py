import math

import click


@click.command()
@click.argument("message_size", type=int)  # , help='Size of Mixer messages.')
@click.argument(
    "num_messages", type=int
)  # , help='Number of Mixer messages.')
def calculate_slot_time(message_size, num_messages):
    MX_SLOT_LENGTH = 80000  # initial value for iterative approach, in ticks
    RX_TO_GRID_OFFSET = 40 * 16  # ticks
    ISR_LATENCY_BUFFER = 20 * 16  # ticks
    MX_GENERATION_SIZE = num_messages
    MX_PAYLOAD_SIZE = message_size  # B
    PHY_PAYLOAD_SIZE = (
        2 + 1 + 1 + 2 * math.ceil(MX_GENERATION_SIZE / 8) + MX_PAYLOAD_SIZE
    )  # B
    PACKET_AIR_TIME = ((2 + 4 + 2 + PHY_PAYLOAD_SIZE + 3) * 4) * 16  # ticks
    JITTER_TOLERANCE = 4 * 16  # ticks

    while True:
        DRIFT_TOLERANCE = min(
            2500, max(math.ceil(MX_SLOT_LENGTH / 1000), 1)
        )  # ticks
        RX_WINDOW_MIN = 2 * (
            (3 * DRIFT_TOLERANCE) + (2 * JITTER_TOLERANCE) + 5 * 16
        )  # ticks
        RX_WINDOW_INCREMENT = (3 * DRIFT_TOLERANCE) / 2  # ticks
        RX_WINDOW_MAX = min(
            RX_WINDOW_MIN + (20 * RX_WINDOW_INCREMENT),
            (
                MX_SLOT_LENGTH
                - PACKET_AIR_TIME
                - RX_TO_GRID_OFFSET
                - ISR_LATENCY_BUFFER
            )
            / 2,
        )

        min_len_slot = (
            PACKET_AIR_TIME
            + RX_TO_GRID_OFFSET
            + 2 * RX_WINDOW_MAX
            + ISR_LATENCY_BUFFER
            + 25 * 16
        ) * 1.0003

        if min_len_slot == MX_SLOT_LENGTH:
            break
        else:
            MX_SLOT_LENGTH = min_len_slot

    print(
        f"Slot time for {num_messages} msgs of {message_size} B (BLE 2M): {math.ceil(MX_SLOT_LENGTH / 16)} us (MX_SLOT_LENGTH = {math.ceil(MX_SLOT_LENGTH)})"
    )


if __name__ == "__main__":
    calculate_slot_time()
