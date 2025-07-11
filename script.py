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
from src.cnn_multi_pixel.preprocessing.preprocessing_array import get_train_test_traces

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
    train_dict = {}
    test_dict = {}
    # Select if initial or additional run
    is_initial = True
    
    # 1) Check if there are any saved raw or subsampled datasets that correspond to required folder
    # IMPLEMENT AFTER dataset_raw
    '''
    2) Convert all training/testing folders into raw traces
    Each dictionary:
        Key: string; Folder name
        Value: dictionary; Each entry represents a file within the target folder
            Key: int; Digital value of file
            Value: list of tuples; list of time_val_tuple tuples of converted files
            time_val_tuple[0]: np.float64; time value
            time_val_tuple[1]: np.float64; trace value
    '''
    train_dict, test_dict = get_train_test_traces(trace_root, train_dict, test_dict, file_pattern)
    # 3) Create and save raw dataset
    # 4) Run subsamplig
    # 5) Create and save subsampled dataset
    # 6) Normalize all datasets
    if is_initial:
        print("lol")
    else:
        print("lol")
    # 7) Create dataloader
    # 8) Create CNN
    # 9) Run training
    # 10) Run testing
    # 11) Print results
