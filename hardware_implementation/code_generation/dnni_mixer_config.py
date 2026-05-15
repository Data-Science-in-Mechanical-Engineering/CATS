from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader
import math

import numpy as np

def get_max_message_size(sizes: dict):
    m = 0
    for key in sizes:
        if type(sizes[key]) == dict:
            m = max(m, get_max_message_size(sizes[key]))

        else:
            m = max(m, max(sizes[key]))
    return m

def calculate_phy_payload_size(message_size_total, num_messages):
    return 2 + 1 + 1 + 2 * math.ceil(num_messages / 8) + message_size_total

def calculate_slot_time(message_size_total, num_messages):
    MX_SLOT_LENGTH = 80000  # initial value for iterative approach, in ticks
    RX_TO_GRID_OFFSET = 40 * 16  # ticks
    ISR_LATENCY_BUFFER = 20 * 16  # ticks
    MX_GENERATION_SIZE = num_messages
    MX_PAYLOAD_SIZE = message_size_total  # B
    PHY_PAYLOAD_SIZE = 2 + 1 + 1 + 2 * math.ceil(MX_GENERATION_SIZE / 8) + MX_PAYLOAD_SIZE  # B
    PACKET_AIR_TIME = ((2 + 4 + 2 + PHY_PAYLOAD_SIZE + 3) * 4) * 16  # ticks
    JITTER_TOLERANCE = 4 * 16  # ticks

    while True:
        DRIFT_TOLERANCE = min(2500, max(math.ceil(MX_SLOT_LENGTH / 1000), 1))  # ticks
        RX_WINDOW_MIN = 2 * ((3 * DRIFT_TOLERANCE) + (2 * JITTER_TOLERANCE) + 5 * 16)  # ticks
        RX_WINDOW_INCREMENT = (3 * DRIFT_TOLERANCE) / 2  # ticks
        RX_WINDOW_MAX = min(RX_WINDOW_MIN + (20 * RX_WINDOW_INCREMENT),
                        (MX_SLOT_LENGTH - PACKET_AIR_TIME - RX_TO_GRID_OFFSET - ISR_LATENCY_BUFFER) / 2)

        min_len_slot = (PACKET_AIR_TIME + RX_TO_GRID_OFFSET + 2 * RX_WINDOW_MAX + ISR_LATENCY_BUFFER + 25 * 16) * 1.0003

        if min_len_slot == MX_SLOT_LENGTH:
            break
        else:
            MX_SLOT_LENGTH = min_len_slot

    return math.ceil(MX_SLOT_LENGTH / 16)


def calculate_air_size(message_size_total, num_messages):
    MX_GENERATION_SIZE = num_messages
    MX_PAYLOAD_SIZE = message_size_total  # B
    PHY_PAYLOAD_SIZE = 2 + 1 + 1 + 2 * math.ceil(MX_GENERATION_SIZE / 8) + MX_PAYLOAD_SIZE  # B
    return (2 + 4 + 2 + PHY_PAYLOAD_SIZE + 3)


def calculate_num_slots(num_messages):
    base_num_slots = 200
    return 3 * num_messages + base_num_slots  # max(4 * num_messages, base_num_rounds)


def calculate_mixer_settings(message_size, num_messages, agg_size):
    num_slots = calculate_num_slots(num_messages)
    duration_slot = calculate_slot_time(message_size + agg_size, num_messages)
    return num_slots, duration_slot


def calculate_num_messages(message_size, message_list):
    num_messages = 0
    for m in message_list:
        num_messages += math.ceil(m / message_size - 1e-6)
    return num_messages + 1  # because of initator message


def generate_node_array(name, id_nodes):
    code = f"static const uint8_t {name}[] = {{"
    for idn in id_nodes:
        code += f"{idn}, "
    code = code[:-2]

    code += "};\n"
    return code



def generate_timing_configuration(message_size_list, agg_size):
    num_mixer_rounds = 0
    min_duration_round = 10000000
    best_message_size = 0
    best_num_messages = 0
    best_num_slots = 0
    best_duration_slot = 0
    best_num_mixer_rounds = 0

    for size in range(150, 230):
        num_messages_total = calculate_num_messages(size, message_size_list)
        num_mixer_rounds = 1
        for num_mixer_rounds in range(1,10):
            num_messages = int(round((num_messages_total - 1e-4) // num_mixer_rounds + 1))
            if calculate_phy_payload_size(size + agg_size, num_messages) > 250:
                continue
            num_slots, duration_slot = calculate_mixer_settings(size, num_messages, agg_size)
            duration_round = num_slots * duration_slot * num_mixer_rounds * 1e-6
            if duration_round < min_duration_round:
                min_duration_round = duration_round
                best_message_size = size
                best_num_messages = num_messages
                best_num_slots = num_slots
                best_duration_slot = duration_slot
                best_num_mixer_rounds = num_mixer_rounds

    return best_num_mixer_rounds, best_message_size, best_num_slots, best_duration_slot, best_num_messages


def generate_mixer_config(filepath, num_devices, num_total_nodes, size_data, size_data_double, calculation_duration, num_triggered_devices):
    print()
    print("Generating mixer configuration ...")
    id_devices = [i + 1 for i in range(num_devices)]
    id_relays = [i + num_devices + 1 for i in range(num_total_nodes - num_devices + 1)]

    header_message_size = 4  # 2 + padding
    data_message_size = header_message_size + size_data
    data_message_size_double = header_message_size + size_data_double

    message_ids = [i+1 for i in range(num_triggered_devices)]
    message_sizes = [data_message_size] * num_triggered_devices

    aggregate_flag_size = 0

    agg_size = (num_triggered_devices * 8 + num_triggered_devices * 8 + num_devices + 7) // 8

    mx_number_rounds, mx_payload_size, mx_round_length, slot_length, mx_generation_size = generate_timing_configuration(
        message_size_list=message_sizes,
        agg_size=agg_size)

    
    mixer_config = {
    
    }


################
# Generation of assignements
#################
def generate_message_assignment_list(name, sizes: list):
    code = f"static message_assignment_element_t message_assignment_elements_{name}[] = {{"
    for i, s in enumerate(sizes):
        code += f"{{.id={i + 1}, .size={int(round(s+4))}}}, "
    code += "};\n"
    return code
    

def generate_message_assignment(name, idx, num_mixer_rounds, length):
    return f"static message_assignment_t message_assignment_{name} = {{.id={idx+1}, .num_mixer_rounds={num_mixer_rounds}, .length={length}, .assignments=message_assignment_elements_{name}}};\n"


def generate_dnni_mixer_config(num_devices, message_sizes, num_tokens, path_output=None):
    message_sizes = {k: np.array(v) * num_tokens for k, v in message_sizes.items()}
    size_activations=int(round(get_max_message_size(message_sizes)))
    message_assignment_list_code = ""
    message_assignment_code = ""    
    largest_messages = None
    
    # Generate message assignment code for all message types
    for i, (name, sizes) in enumerate(message_sizes.items()):
        message_assignment_list_code += generate_message_assignment_list(name, sizes)
        message_assignment_code += generate_message_assignment(name, i, 1, len(sizes))

    id_devices = [i + 1 for i in range(num_devices)]

    # Calculate configurations for each message type
    configs = []
    for name, sizes in message_sizes.items():
        total_sizes = [int(round(s + 4)) for s in sizes]
        mx_number_rounds, mx_payload_size, mx_round_length, slot_length, mx_generation_size = generate_timing_configuration(
            message_size_list=total_sizes,
            agg_size=0)
        
        config = {
            'name': name,
            'mx_number_rounds': mx_number_rounds,
            'mx_payload_size': mx_payload_size,
            'mx_round_length': mx_round_length,
            'slot_length': slot_length,
            'mx_generation_size': mx_generation_size
        }
        configs.append(config)

    # Filter out configs that don't work for all message types
    valid_configs = []
    extra_mesages = 0
    while not valid_configs:
        for config in configs:
            config['mx_generation_size'] += extra_mesages
            is_valid = True
            for name, sizes in message_sizes.items():
                total_sizes = [int(round(s + 4)) for s in sizes]
                num_mx_messages = calculate_num_messages(config['mx_payload_size'], total_sizes)
                if num_mx_messages > config['mx_generation_size']:
                    is_valid = False
                    break
            
            if is_valid:
                # Calculate total round time
                round_time = config['mx_number_rounds'] * config['mx_round_length'] * config['slot_length']
                config['round_time'] = round_time
                valid_configs.append(config)
        extra_mesages += 1
        if not valid_configs:
            print("Could not find valid mixer configuration for all message types")

    # if not valid_configs:
    #     raise ValueError("No valid configuration found for all message types")

    # Select the configuration with the smallest round time
    best_config = min(valid_configs, key=lambda x: x['round_time'])
    
    mx_number_rounds = best_config['mx_number_rounds']
    mx_payload_size = best_config['mx_payload_size']
    mx_round_length = best_config['mx_round_length']
    slot_length = best_config['slot_length']
    mx_generation_size = best_config['mx_generation_size']
    
    # Find largest messages for backward compatibility
    max_total_size = 0
    for name, sizes in message_sizes.items():
        total_size = sum([int(round(s + 4)) for s in sizes])
        if total_size > max_total_size:
            max_total_size = total_size
            largest_messages = sizes


    config = {
        "nodes": id_devices,
        "mx_number_rounds": mx_number_rounds,
        "mx_payload_size": mx_payload_size,
        "mx_round_length": mx_round_length,
        "slot_length": slot_length + 50,
        "calculation_duration": 1000,
        "mx_generation_size": mx_generation_size,
        "aggregate_flag_size": 0,
        "aggregate_content_size":  0,
        "message_assignment_list_code": message_assignment_list_code,
        "message_assignment_code": message_assignment_code,
        "maximum_number_messages": len(largest_messages),
        "size_activations": size_activations
    }

    if path_output is None:
        path_output = f"{Path(__file__).parent.absolute()}/../firmware/distributed_inference"

    jinja_environment = Environment(loader=FileSystemLoader(f'{Path(__file__).parent.absolute()}/templates'))
    mixer_config_h = jinja_environment.get_template('dnni_mixer_config.h.jinja')
    output = mixer_config_h.render(config)
    with open(f"{path_output}/dnni_mixer_config.h", 'w') as f:
        f.write(output)
