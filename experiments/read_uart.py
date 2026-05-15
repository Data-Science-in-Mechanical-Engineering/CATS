import serial
import numpy as np
import pandas as pd

import time

import sys
import glob
import serial

import psutil, os

import argparse


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def extract_duration(line):
    # Extract the duration value between ": " and " us"
    start_idx = line.find(": ") + 2
    end_idx = line.find(" us")
    if start_idx != 1 and end_idx != -1:  # Check if both markers were found
        return int(line[start_idx:end_idx]) / 1000
    return 0


if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(15)

    max_num_samples = 20
    num_nodes = 10
    num_triggered_devices = 10
    quantize = False
    seed = 1
    name = "HAR"
        
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--port', dest='port', type=str, help='serial port')
    parser.add_argument('--list_ports', action='store_true', help='list available serial ports')
    args = parser.parse_args()
    
    if args.list_ports:
        print(serial_ports())
        exit()
        
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--port', dest='port', type=str, help='serial port')
    args = parser.parse_args()
    ser = serial.Serial(args.port, baudrate=115200)
    residual_block_durations = []
    attention_block_durations = []
    communication_durations = []
    print("Connected.")
    with open(f'/data/distributed_transformer/logs{args.port.replace("/", "_")}.txt', 'w') as f:
        while True:
            line = ser.readline().decode('utf-8').strip()
            f.write(line + '\n')
            if line.startswith("Residual block computing duration:"):
                residual_block_durations.append(extract_duration(line))
            elif line.startswith("Attention block computing duration"):
                attention_block_durations.append(extract_duration(line))
            elif line.startswith("Communication duration:"):
                communication_duration = extract_duration(line)
                print(communication_duration)
                communication_durations.append(communication_duration)

                if len(communication_durations) >= max_num_samples and False:
                    break
            elif line.startswith("Message loss this round:"):
                message_loss = int(line.split(": ")[1]) / 10000.0
                print(f"Current message loss: {message_loss}")
            elif line.startswith("slot_off:"):
                print(line)
            # print(line)

    def mean(data):
        return sum(data) / len(data)
    print(f"Mean residual block duration: {mean(residual_block_durations):.2f} us")
    print(f"Mean attention block duration: {mean(attention_block_durations):.2f} us")
    print(f"Mean communication duration: {(mean(communication_durations) * 2):.2f} us")

    import matplotlib.pyplot as plt

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot residual block durations
    ax1.plot(residual_block_durations, 'b-', linewidth=1)
    ax1.set_title('Residual Block Computing Duration')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Duration (us)')
    ax1.grid(True)
    
    # Plot attention block durations
    ax2.plot(attention_block_durations, 'r-', linewidth=1)
    ax2.set_title('Attention Block Computing Duration')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Duration (us)')
    ax2.grid(True)
    
    # Plot communication durations
    ax3.plot(communication_durations, 'g-', linewidth=1)
    ax3.set_title('Communication Duration')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Duration (us)')
    ax3.grid(True)
    
    plt.show()
