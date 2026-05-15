# distributed_transformer_inference

This repos is the accompanying repo of our paper *Going beyond the Edge: Distributed Inference of Transformer Models on Ultra-low-power Wireless Devices*

## Installation
The code has been tested with Python 3.12. Install all packages with

```bash
pip install -r requirements.txt
```

## Entrance points
### [train.py](train.py)

Starts the training. We use hydra (https://hydra.cc/) for configuring our training runs. The parameters can be found in [parameters/](parameters). The folder [parameters/run](parameters/run) contains the dataset-specific parameter changes.

Example usage:
```bash
python train.py "+run=cat_vs_dog" "dataset.dataset_base_path=<path to dataset>" "root_dir=<path to logging files>"
```
### [export_to_hardware.py](export_to_hardware.py)

Exports a given neural network to hardware code.

Example usage:
```bash
python export_to_hardware.py "+run=cat_vs_dog" "dataset.dataset_base_path=<path to dataset>" "root_dir=<path to logging files>"
```

### [hardware_implementation/](ardware_implementation/)

Our code for the hardware implementation. Open [hardware_implementation/firmware/firmware.emProject](hardware_implementation/firmware/firmware.emProject) in Segger embedded studio (https://www.segger.com/products/development-tools/embedded-studio/). You can compile and flash the code using [hardware_implementation/build_and_flash.bash](hardware_implementation/build_and_flash.bash).

The implementation of the layers can be found in [hardware_implementation/firmware/distributed_inference/](hardware_implementation/firmware/distributed_inference/).