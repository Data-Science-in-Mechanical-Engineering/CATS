The scripts directory contains various scripts to parse, evaluate, and visualize Mixer executions.

# Python setup

- suggestion: use a virtual environment (https://docs.python.org/3/tutorial/venv.html)
	- create venv: python3 -m venv .venv
	- activate venv: source .venv/bin/activate
- install requirements: pip install -r requirements.txt

# Scripts

## evaluation/MixerLogParser.py

This is a log parser script. It reads the logs from testbeds (local, graz, flocklab) and produces logs in a standardized format for futher evaluation scripts. The new log is located in the log directory at generated_logs/log_formatted.

- MixerLogParser.py [-h] [--lvl {INFO,DEBUG}] path files description {flocklab,graz,local}
- example: python3 MixerLogParser.py test_data "serial.csv" "test description" flocklab

## evaluation/evaluation.py

Script to evaluate Mixer rounds. MixerLogParser must be used before to generate standardized logs. Generates plots in the log directory at plots/.

- evaluation.py [-h] [--lvl {INFO,DEBUG}] [--all] [--force] path
- example: python3 evaluation.py test_data/

## evaluation/eval_metric.py

Script to evaluate a user-defined metric. The metric is described via regular expressions and currently expected to be an integer. MixerLogParser must be used before to generate standardized logs.

- eval_metric.py [-h] [--lvl {INFO,DEBUG}] [--all] path metric pattern
- example: python3 eval_metric.py test_data "BrokenRX" "num_rx_broken: (?P<metric>\d+)\n"
	- matches the pattern "num_rx_broken: (?P<metric>\d+)\n" against all log entries (the term "metric" must be used!)
	- creates result as a violin plot
Outdated script to generate an overview table for multiple different experiments.

## visualization/MixerVisualization.py

This script visualizes information about the Mixer round and allows to inspect what happened in each slot. It shows some internal state of nodes (e.g., its current rank) and the packet exchange between nodes.

1. python3 MixerVisualization.py
2. Click "Load Log" and select log directory (e.g., visualization/test_data/log/) and wait until "Loaded Log: ..." shows up in the upper left corner (log parsing may take a bit of time).
3. Click "Load Node Layout" (e.g., visualization/test_data/node_layout_graz).
5. Select round, select slot and press "Apply".
	- slot selection can also be a range (e.g., 1-50)

- nodes can be moved around and new layouts can be saved
- node information:

	|---------------------------------------------|
	| physical node ID     |        rank          |
	|----------------------|----------------------|
	| transmit probability |      neighbors       | 
	|---------------------------------------------|

	- white box means nothing received so far (not initiated)
	- dark green means successful reception and rank up (something innovative was received)
	- light green means successful reception of already known information
	- blue means transmitting
	- check options on the top of the window for more information (e.g., show neighbors of specific nodes) 
- lines show successful packet transmissions (blue node transmitted and green nodes received)
	- different packets use different colors, however, currently there is a limited number of colors so they repeat if there are a lot of transmissions shown at the same time (this especially happens when showing slot ranges)

## calculate_slot_time.py

Takes the message size and number of messages (generation size) as input and outputs the minimal slot length for Mixer (BLE 2M mode only!).
