# Imports
# 0) Necessary imports
import argparse
import seaborn as sns
import numpy as np
from collections import defaultdict

import re
import os
import time

import torch
import matplotlib
import matplotlib.pyplot as plt

# 1) File imports
from src.cnn_multi_pixel.setup.setup_directories import create_directories
from src.cnn_multi_pixel.preprocessing.preprocessing_array import create_raw_traces
from src.cnn_multi_pixel.dataset.dataset_raw import create_and_save_raw_datasets
from src.cnn_multi_pixel.dataset.dataset_sampled import create_and_save_sampled_datasets


# Plot settings
plt.rcParams.update({'font.size': 10})
FIGX = 8
FIGY = 8

# Get CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cpuonly", const=True, default=False, action='store_const', help="Don't use GPU even if available")
parser.add_argument("-n", "--nowrite", const=True, default=False, action='store_const', help="Don't write any outputs")
parser.add_argument("-f", "--force", const=True, default=False, action='store_const', help="Overwrite output files")
parser.add_argument("-p", "--preview", const=True, default=False, action='store_const', help="Don't run anything only list runs that would occur")
parser.add_argument("-x", "--headless", const=True, default=False, action='store_const', help="Do not open any gui's")
parser.add_argument("-t", "--test", type=str, default="", help="Limit run tests to those whose description matches the specified regex")
parser.add_argument("-r", "--repeat", type=int, default=1, help="Rerun training/test this many times. requires -f flag to work properly")
parser.add_argument("--nndebug", const=True, default=False, action='store_const', help="Print information about cnn creation.")
parser.add_argument("-i", "--json",   type=str, default='regression.json', help="JSON description of CNNs")
parser.add_argument("-o", "--output", type=str, default='outputs', help="Directory for output files")

# Base Paths
pwd      = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(os.path.dirname(pwd))



if __name__ == "__main__":
    args = parser.parse_args()
    if not args.nowrite:
        os.makedirs(args.output, exist_ok=True)
    
    # DIRECTORY PATH SETTER
    # NEED ADDITIONAL FUNCTION; DO AT END AFTER ALL LOGIC IS FINALIZED
    cwd = os.getcwd()
    # 1) Initialize all necessary directories
    trace_root, datasets_raw_root, datasets_sampled_root, cnn_settings_root = create_directories(cwd)

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
    train_new_dict = create_raw_traces(trace_root, train_folders_new)
    test_new_dict = create_raw_traces(trace_root, test_folders_new)
    # 3) Create and save raw datasets
    train_datasets_new = create_and_save_raw_datasets(datasets_raw_root, train_new_dict)
    test_datasets_new = create_and_save_raw_datasets(datasets_raw_root, test_new_dict)
    # 4) For each new raw dataset, create and save a subsampled dataset
    train_datasets_new_sampled, test_datasets_new_sampled = create_and_save_sampled_datasets(datasets_sampled_root, train_datasets_new, test_datasets_new)
    # 5) Form both training and testing by combining both saved and new datasets
    # 6) Normalize all datasets
    # 7) Create dataloader - APPLY BITWISE/SINGLE MODE SETTER HERE
    
    # Reuse Kareem's code as much as possible from here
    # 8) Create CNN
    # 9) Run training
    # 10) Run testing
    # 11) Print results