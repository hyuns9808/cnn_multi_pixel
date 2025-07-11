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
from src.cnn_multi_pixel.preprocessing.preprocessing_array import create_train_test_raw_traces
from src.cnn_multi_pixel.dataset.dataset_raw import create_and_save_raw_dataset
from src.cnn_multi_pixel.dataset.dataset_sampled import create_and_save_sampled_dataset

if __name__ == "__main__":
    # 0) Get inputs
    # AFTER ALL LOGIC IS FINALIZED, TRY TO USE KAREEM'S HYPERPARAMETER SETTINGS
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
    
    # DIRECTORY PATH SETTER
    # NEED ADDITIONAL FUNCTION; DO AT END AFTER ALL LOGIC IS FINALIZED
    trace_root = 'root'
    raw_dataset_root = 'root'
    sample_dataset_root = 'root'
    
    # ASSUMING FILE INPUT HERE
    train_dict = {}
    test_dict = {}
    # Select if initial or additional run
    is_initial = True
    
    # 1) Check if there are any saved raw or subsampled datasets that correspond to required folder
    # NEED ADDITIONAL FUNCTION; ADD SAVED-NEW CHECKER FUNCTION
    train_folders_saved = {}
    train_folders_new = {}
    test_folders_saved = {}
    test_folders_new = {}
    # 2) Convert all training/testing folders into raw traces
    '''
    Both train_new_dict & test_new_dict:
        Key: string; Folder name
        Value: dictionary; Each entry represents a file within the target folder
            Key: int; Digital value of file
            Value: list of tuples; list of time_val_tuple tuples of converted files
            time_val_tuple[0]: np.float64; time value
            time_val_tuple[1]: np.float64; trace value
    '''
    train_new_dict, test_new_dict = create_train_test_raw_traces(trace_root, train_folders_new, test_folders_new, file_pattern)
    # 3) Create and save raw datasets
    '''
    Both train_datasets_new & test_datasets_new: list of new raw datasets
    '''
    train_datasets_new, test_datasets_new = create_and_save_raw_dataset(raw_dataset_root, train_new_dict, test_new_dict)
    # 4) For each new raw dataset, create and save a subsampled dataset
    '''
    Both train_datasets_new & test_datasets_new: list of new sampled datasets
    '''
    train_datasets_new_sampled, test_datasets_new_sampled = create_and_save_sampled_dataset(sample_dataset_root, train_datasets_new, test_datasets_new)
    # 5) Form both training and testing by combining both saved and new datasets
    # 6) Normalize all datasets
    if is_initial:
        print("lol")
    else:
        print("lol")
    # 7) Create dataloader - APPLY BITWISE/SINGLE MODE SETTER HERE
    
    # Reuse Kareem's code as much as possible from here
    # 8) Create CNN
    # 9) Run training
    # 10) Run testing
    # 11) Print results
