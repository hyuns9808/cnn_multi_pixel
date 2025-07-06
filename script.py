# Imports
# 0) Necessary imports
import argparse
import seaborn as sns
import numpy as np
from collections import defaultdict
import os
import re

# 1) Hyperparameter settings
# Check "hyperparams.py" file to change settings 
# ADC hyperparameters
from hyperparams import adc_num, adc_bitwidth, train_batch, split_digital, normalized_digital
# Directory hyperparameters
from hyperparams import sub_folder, sub_folder_name, train_trace_folder_names, test_trace_folder_names, trace_type, file_pattern
# 2) File imports
from src.cnn_multi_pixel.preprocessing_exp import get_avg_exponent
from src.cnn_multi_pixel.preprocessing_array import create_trace_arrays

if __name__ == "__main__":
    # 0) Get inputs
    parser = argparse.ArgumentParser(
                    prog='multi_ADC',
                    description='Attacks multiple ADCs using either a bitwise or singular attack.',
                    epilog='Bitwise == 1 ADC per 8 bits, singular == 1 ADC for all bits')
    # -c/--cli: Set to "True" to read inputs straight from cli
    parser.add_argument('-c', '--cli', action='store_true')  # on/off flag
    parser.add_argument('-f', '--filename', help='Path to the input file')
    args = parser.parse_args()

    # CLI-mode
    if args.cli:
        print("CLI")
    # File read mode
    else:       
        print("Read file")
    
    # ASSUMING FILE INPUT HERE
    trace_root = 'root'
    train_folder_list = []
    test_folder_list = []

    # 1) Preprocessing
    # 1-1) Get average exponent from all files, BOTH training AND testing
    print("(1) Obtaining average exponent from both training and testing files...")
    avg_exponent = get_avg_exponent(trace_root, train_folder_list, test_folder_list)
    print(f"\tObtained average exponent value: {avg_exponent}")
    # 1-2) Read files, store RAW values into np arrays
    print(f"(2) Creating trace arrays...")
    train_traces, test_traces = create_trace_arrays(trace_root, train_folder_list, test_folder_list, avg_exponent)

    # 2) Creating dataset/dataloaders
    # 3) Create CNN
    # 4) Run training
    # 5) Run testing