import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import os
import logging
import pickle
import subprocess
import sys
from tqdm import tqdm

import hydra

from model.vit import VIT
from model import quantization

from export_to_hardware import convert_to_python, run_export
import pandas as pd
import shutil


def generate_code_for_config(base_path, cfg: omegaconf.OmegaConf, num_features: int, input_length: int, scalings_weight=None, scalings_activation=None):
    """
    Generate C code for a given configuration.
    
    Args:
        cfg: Hydra configuration object
        num_features: Number of features for attention/residual/head layers
        input_length: Length of input X (number of patches)
    """
    # logger = logging.getLogger(__name__)

    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 3)
    
    # Update config with new feature dimensions
    cfg.model.num_features_attention = num_features
    cfg.model.num_features_residual = num_features
    cfg.model.num_features_head = num_features

    cfg.model.input_patch_length = 1   # to avoid that the input data has an overly large influence
    
    # Create output path
    path_output = f"{base_path}/firmware/distributed_inference"
    
    # Create model (no loading from file)
    model = VIT(cfg, keys[0])

    if cfg.per_step_pruning_ratio > 0.0:
        model = model.prune_step(cfg.per_step_pruning_ratio)
    
    # Create random example input with the specified length
    # Input shape: (input_length, num_features)
    X = 2 * jax.random.uniform(keys[2], (input_length, model.input_dim)) - 1.0
    
    run_export(model, X, scalings_weight, scalings_activation, cfg, path_output)
    
    return True


def compile_firmware(base_path, device_id: int, board_type: str = "GPI_ARCH_BOARD_nRF_PCA10056"):
    """
    Compile firmware for a specific device.
    
    Args:
        device_id: Device ID to compile for (1-16)
        board_type: Board type (default: GPI_ARCH_BOARD_nRF_PCA10056)
    
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # logger.info(f"Compiling firmware for device {device_id}...")
    
    # Build command similar to build_and_flash.bash
    build_command = [
        "/opt/SEGGER/segger_embedded_studio_8.22a/bin/emBuild",
        "-rebuild",
        "-verbose",
        "-config", "Debug",
        "-D", f"EXT_THIS_NODE_ID={device_id}",
        "-D", f"EXT_BOARD={board_type}",
        f"{base_path}/firmware/firmware.emProject"
    ]
    
    # Change to hardware_implementation directory
    hw_impl_dir = os.path.join(os.path.dirname(__file__), "hardware_implementation")
    
    try:
        result = subprocess.run(
            build_command,
            cwd=hw_impl_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # if result.stdout:
        #     logger.info(result.stdout)
        # if result.stderr:
        #     logger.warning(f"Build stderr: {result.stderr}")
        
        if result.returncode != 0:
            # logger.error(f"Compilation failed for device {device_id} with return code {result.returncode}")
            return False
        
        # logger.info(f"Compilation successful for device {device_id}")
        return True
        
    except subprocess.TimeoutExpired:
        # logger.error(f"Compilation timeout for device {device_id}")
        return False
    except Exception as e:
        # logger.error(f"Compilation error for device {device_id}: {e}")
        return False


@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
def main(cfg: omegaconf.OmegaConf):
    recalculate = True
    paper_plot_data_dir = os.path.expanduser("~/Documents/009_Paper/papers-dsme-nes/distributed_neural_network_inference/plot_data")
    
    # copy firmware code to tmp directory
    os.makedirs(os.path.expanduser("~/tmp"), exist_ok=True)
    firmware_dir = os.path.join(os.path.expanduser("~/tmp"), f"firmware_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio*100)}")
    os.makedirs(firmware_dir, exist_ok=True)
    # Copy firmware directory contents
    hw_impl_dir = os.path.join(os.path.dirname(__file__), "hardware_implementation")
    
    shutil.copytree(hw_impl_dir, firmware_dir, dirs_exist_ok=True)
    
    if recalculate:
        jax.config.update('jax_platform_name', 'cpu')
        
        # calculate scalings once.
        key = jax.random.PRNGKey(1)
        keys = jax.random.split(key, 3)
        model = VIT(cfg, keys[0])
        X = 2 * jax.random.uniform(keys[2], (10, model.input_dim)) - 1.0
        Y, record_activations = model.decoder(X, None, None, True, keys[0], True)
        
        # Calculate scalings
        scalings_activation = quantization.calculate_scalings(record_activations, num_bits=8)
        scalings_weight = quantization.calculate_scalings(model.get_weight_dict(), num_bits=8)

        num_features_step_size = 16
        num_features_start = 32
        num_features_end = 2048 # 512

        input_length_step_size = 32
        input_length_start = 32
        input_length_end = 1024

        
        df = pd.DataFrame(columns=["num_features", "input_length", "compilation_success"])
        stop_sweep = False
        for num_features in range(num_features_start, num_features_end, num_features_step_size):
            for input_length in range(input_length_start, input_length_end, input_length_step_size):
                print(f"Testing num_features={num_features}, input_length={input_length}...")
                generate_code_for_config(firmware_dir,
                                        cfg, 
                                        num_features=num_features, 
                                        input_length=input_length, 
                                        scalings_weight=scalings_weight, 
                                        scalings_activation=scalings_activation)
                success = compile_firmware(base_path=firmware_dir, device_id=1, board_type="GPI_ARCH_BOARD_nRF_PCA10056")
                df = pd.concat([df, pd.DataFrame.from_records([{
                    "num_features": num_features,
                    "input_length": input_length,
                    "compilation_success": success
                }])], ignore_index=True)
                if input_length == 32 and not success:
                    # If even the smallest input_length fails, stop increasing num_features
                    stop_sweep = True
                    break
                if not success:
                    # No need to test larger input lengths if this one failed
                    break
            if stop_sweep:
                break

        # Save results to CSV
        df.to_csv(os.path.join(paper_plot_data_dir, f"compilation_results_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio*100)}.csv"), index=False)
    else:
        # Load results from CSV
        df = pd.read_csv(os.path.join(paper_plot_data_dir, f"compilation_results_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio*100)}.csv"))

    # Find the boundary: max successful num_features for each input_length
    boundary_data = []
    for input_len in sorted(df["input_length"].unique()):
        input_df = df[df["input_length"] == input_len]
        successful = input_df[input_df["compilation_success"] == True]
        if not successful.empty:
            max_features = successful["num_features"].max()
            boundary_data.append({"input_length": input_len, "num_features": max_features})
    
    boundary_df = pd.DataFrame(boundary_data)
    boundary_df.to_csv(os.path.join(paper_plot_data_dir, f"compilation_boundary_{cfg.num_devices}_{int(cfg.per_step_pruning_ratio*100)}.csv"), index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot successful compilations in green
    success_df = df[df["compilation_success"] == True]
    ax.scatter(success_df["input_length"], success_df["num_features"], 
               c='green', marker='o', s=50, alpha=0.6, label='Success')

    # Plot failed compilations in red
    fail_df = df[df["compilation_success"] == False]
    ax.scatter(fail_df["input_length"], fail_df["num_features"], 
               c='red', marker='x', s=50, alpha=0.6, label='Failed')
    
    # Plot the boundary line
    if not boundary_df.empty:
        ax.plot(boundary_df["input_length"], boundary_df["num_features"], 
                'k-', linewidth=2, label='Boundary')

    ax.set_xlabel('Input Length')
    ax.set_ylabel('Number of Features')
    ax.set_title('Compilation Success by Input Length and Number of Features')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('compilation_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()