# Imports
# 0) Necessary imports
import argparse
# not usre if i need these, but leave here
import seaborn as sns
import numpy as np
from collections import defaultdict
import os
import re

# 1) Hyperparameter settings
# Check "hyperparams.py" file to change settings 
# ADC hyperparameters
from .src.cnn_multi_pixel.hyperparams import adc_num, adc_bitwidth, train_batch, split_digital, normalized_digital
# Directory hyperparameters
from .src.cnn_multi_pixel.hyperparams import sub_folder, sub_folder_name, train_trace_folder_names, test_trace_folder_names, trace_type, file_pattern


if __name__ == "__main__":
    # 0) Get inputs
    parser = argparse.ArgumentParser(
                    prog='multi_ADC',
                    description='Attacks multiple ADCs using either a bitwise or singular attack.',
                    epilog='Bitwise == 1 ADC per 8 bits, singular == 1 ADC for all bits')
    # -c/--cli: Set to "True" to read inputs straight from cli
    parser.add_argument('-c', '--cli', action='store_True')  # on/off flag
    parser.add_argument('-f', '--filename', help='Path to the input file')
    args = parser.parse_args()

    # CLI-mode
    if args.cli:
        print("CLI")
    # File read mode
    else:       
        print("Read file")
    
    # ASSUMING FILE INPUT HERE
    root_dir = 'root'
    folder_list = []

    # 1) Preprocessing
    # 1-1) Get average exponent from all training files
    
    # 1-2) Read files, store normalized values into np arrays
    preprocess_traces(root_dir, folder_list)

    # 2) Creating dataset/dataloaders
    # 3) Create CNN
    # 4) Run training
    # 5) Run testing