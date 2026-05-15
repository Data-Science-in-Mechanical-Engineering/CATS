import hydra
import omegaconf

import math

import pandas as pd

def flash_calculation(F, H=None, D=None):
    """ H, D not none means, we split along the heads and among devices"""
    if H is None:
        flash = 0
        flash += F*F*3  # Wq, Wk, Wv
        flash += F*F  # Wo
        return flash
    
    flash = 0
    heads_per_device = math.ceil(H / D)
    F_local = F * heads_per_device / H
    flash += F_local * F * 3  # Wq, Wk, Wv
    flash += F_local * F  # Wo
    return flash

def calculate_resources_Hu_2024(D, N, F, H):
    ram = 0
    ram += N*F  # X
    ram += N/D*F/H  # Q
    ram += N*F/H # K
    ram += N*F/H  # V
    ram += N/D * F  # H
    ram += N/D * F # O

    flash = flash_calculation(F)

    com = 0
    com += N*F

    return ram, flash, com

def calculate_resources_Liu_2025(D, N, F, H):
    ram = 0
    nx = N/D*F + (D-1) / D * N * F * 5/8
    ram += nx # X
    ram += N/D * F/H # Q
    ram += nx / H  # K
    ram += nx / H  # V
    ram += N/D * F  # H
    ram += N/D * F  # O

    flash = flash_calculation(F)

    com = N * F * 5/8

    return ram, flash, com


def calculate_resources_Bochem_2025(D, N, F, H):
    ram = 0
    ram_attention = 0
    ram_attention += N*F  # X
    ram_attention += N*F/H  # Q
    ram_attention += N*F/H # K
    ram_attention += N*F/H  # V
    heads_per_device = math.ceil(H / D)
    H_local_size = N * F / H * heads_per_device
    ram_attention += H_local_size  # H

    ram_head = H_local_size  # H
    ram_head += N*F # O
    ram_head += N*F * (D - 1) # received Os

    ram = max(ram_attention, ram_head)

    flash = flash_calculation(F, H, D)

    com = N * F * D * 2

    return ram, flash, com


def calculate_resources_cats(D, N, F, H, pruning_ratio):
    ram_attention = 0
    ram_attention += N * F / D  # X
    ram_attention += N * F / D * (D-1) * (1-pruning_ratio)  # X_received
    ram_attention += N * F / H  # Q
    ram_attention += N * F / H  # K
    ram_attention += N * F / H  # V
    heads_per_device = math.ceil(H / D)
    H_local_size = N * F / H * heads_per_device
    ram_attention += H_local_size  # H

    ram_head = H_local_size  # H
    ram_head += (N*F - H_local_size) * (1-pruning_ratio)  # H_received
    ram_head += N * F / D  # O (added to X)

    ram = max(ram_attention, ram_head)

    flash = 0
    heads_per_device = math.ceil(H / D)
    F_local = F * heads_per_device / H
    flash += F_local * (F_local + (F-F_local) * (1-pruning_ratio)) * 3  # Wq, Wk, Wv
    flash += F_local * (F_local + (F-F_local) * (1-pruning_ratio))  # Wo

    com = 0
    com += N * F * (1 - pruning_ratio)   # X
    com += N * F * (1 - pruning_ratio)   # H
    com += N * F * (1 - pruning_ratio)   # O
    com += N * F * (1 - pruning_ratio)   # Residual

    return ram, flash, com




if __name__ == "__main__":

    F = 128
    N = 64
    H = 16
    
    approaches = ["Hu_2024", "Liu_2025", "Bochem_2025", "CATS", "CATS50", "CATS75"]
    resources = {a: {"num_devices": [], "RAM": [], "Flash": [], "Com": []} for a in approaches}

    # Sweep over D (num_devices)
    for D in range(1, 17):  # D from 1 to 16
        # Calculate resources for each approach
        ram_hu, flash_hu, com_hu = calculate_resources_Hu_2024(D, N, F, H)
        ram_liu, flash_liu, com_liu = calculate_resources_Liu_2025(D, N, F, H)
        ram_bochem, flash_bochem, com_bochem = calculate_resources_Bochem_2025(D, N, F, H)
        ram_cats, flash_cats, com_cats = calculate_resources_cats(D, N, F, H, pruning_ratio=0.0)
        ram_cats50, flash_cats50, com_cats50 = calculate_resources_cats(D, N, F, H, pruning_ratio=0.5)
        ram_cats75, flash_cats75, com_cats75 = calculate_resources_cats(D, N, F, H, pruning_ratio=0.9)
        
        # Store results
        resources["Hu_2024"]["num_devices"].append(D)
        resources["Hu_2024"]["RAM"].append(ram_hu)
        resources["Hu_2024"]["Flash"].append(flash_hu)
        resources["Hu_2024"]["Com"].append(com_hu)
        
        resources["Liu_2025"]["num_devices"].append(D)
        resources["Liu_2025"]["RAM"].append(ram_liu)
        resources["Liu_2025"]["Flash"].append(flash_liu)
        resources["Liu_2025"]["Com"].append(com_liu)
        
        resources["Bochem_2025"]["num_devices"].append(D)
        resources["Bochem_2025"]["RAM"].append(ram_bochem)
        resources["Bochem_2025"]["Flash"].append(flash_bochem)
        resources["Bochem_2025"]["Com"].append(com_bochem)
        
        resources["CATS"]["num_devices"].append(D)
        resources["CATS"]["RAM"].append(ram_cats)
        resources["CATS"]["Flash"].append(flash_cats)
        resources["CATS"]["Com"].append(com_cats)

        resources["CATS50"]["num_devices"].append(D)
        resources["CATS50"]["RAM"].append(ram_cats50)
        resources["CATS50"]["Flash"].append(flash_cats50)
        resources["CATS50"]["Com"].append(com_cats50)

        resources["CATS75"]["num_devices"].append(D)
        resources["CATS75"]["RAM"].append(ram_cats75)
        resources["CATS75"]["Flash"].append(flash_cats75)
        resources["CATS75"]["Com"].append(com_cats75)

    # Save each approach as CSV
    for approach_name, data in resources.items():
        df = pd.DataFrame(data)
        df.to_csv(f"/home/alex/Documents/009_Paper/papers-dsme-nes/distributed_neural_network_inference/plot_data/resource_{approach_name}.csv", index=False)

    # main()