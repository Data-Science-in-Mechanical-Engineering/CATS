"""
Standalone export script that runs without JAX dependencies.
This script is called via subprocess from export_to_hardware.py to avoid
JAX/multiprocessing conflicts.
"""
import pickle
import sys
import os

# DO NOT IMPORT JAX HERE - that's the whole point!
import numpy as np

# Import only the export utilities
from utils import export_utils


def main():
    if len(sys.argv) != 2:
        print("Usage: python export_standalone.py <pickle_file_path>")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Starting export...")
    export_utils.export_transformer(
        name=data['name'],
        weight_dict=data['weight_dict'],
        pruning_dict=data['pruning_dict'],
        num_heads=data['num_heads'],
        o_split_input=data['o_split_input'],
        scalings_weight=data['scalings_weight'],
        scalings_activation=data['scalings_activation'],
        num_devices=data['num_devices'],
        example_input=data['example_input'],
        num_bits=data['num_bits'],
        attention_only=data['attention_only'],
        path_output=data['path_output']
    )
    
    print("Export completed successfully!")
    
    # Clean up the pickle file
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
        print(f"Cleaned up temporary file: {pickle_path}")


if __name__ == "__main__":
    main()
