from collections import defaultdict
import numpy as np
import re
import os

''' 
Function that extracts all exponent values from a SINGLE power trace file
Input:
    1) file_path: string; name of power trace file
Returns:
    1) exponents: list; all exponent values from file
'''
def extract_exponents_from_file(file_path):
    exponents = []
    pattern = re.compile(r'^\s*time\s+-i\(vdd\)\s*$')
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and pattern.match(line):
                continue  # Skip function call for first line if it matches
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip lines with fewer than 2 columns
            value_str = parts[1]
            if 'e' not in value_str:
                print(f"Skipping non-scientific value '{value_str}' in {file_path}")
                continue
            try:
                exponent = int(value_str.split('e')[1])
                exponents.append(exponent)
            except ValueError:
                print(f"Invalid exponent format in '{value_str}' from {file_path}")
                continue
    return exponents

''' 
Function that runs through all trace files within given folder directory
and gets 1) sum of all exponent values, 2) number of traces
Input:
    1) trace_folder_dir: string; raw path to trace folder directory
    2) target_folders: array; list of trace folders to get avg exponent
Returns:
    1) exp_results: dictionary;
    key = each folder
    items = list of (sum(exponent), len(exponent)) tuples per files in folder
    2) min_exp: int; minimum exponent value
    3) min_exp: int; maximum exponent value
'''
def extract_exponents_from_folders(trace_folder_dir, target_folders):
    exp_results = defaultdict(list)
    # Values used to check max/min exponent of all folders
    # Used to alert user if average value is too off from these values
    min_exp = None
    max_exp = None

    # Get exp_results: dictionary of exponent values per folder
    # key = each folder
    # items = list of (sum(exponent), len(exponent)) tuples per files in folder
    for folder in target_folders:
        folder_path = os.path.join(trace_folder_dir, folder)
        if os.path.exists(folder_path):
            folder_name = os.path.basename(folder_path)
            print(f"Handling: {folder_name}...")
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    # file_exponents = list of exponent values from given file 
                    file_exponents = extract_exponents_from_file(file_path)
                    if min_exp is None or min_exp > min(file_exponents):
                        min_exp = min(file_exponents)
                    if max_exp is None or max_exp < max(file_exponents):
                        max_exp = max(file_exponents)
                    exp_results[folder_name].append((sum(file_exponents), len(file_exponents)))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
        
    return exp_results, min_exp, max_exp

''' 
Function that gets final avg exponent from both training/testing trace files
Input:
    1) trace_root: string; raw path to trace folder directory
    2) train_folder_list: array; list of training trace folders to get avg exponent
    2) test_folder_list: array; list of testing trace folders to get avg exponent
Returns:
    1) avg_exponent: int; final avg exponent value used to normalize all trace values
'''
def get_avg_exponent(trace_root, train_folder_list, test_folder_list):
    # Combine train_folder_list and test_folder_list and get all exponents
    exp_results, min_exp, max_exp = extract_exponents_from_folders(trace_root, train_folder_list + test_folder_list)
    # Get final result
    tot_exp_val = 0
    tot_exp_len = 0
    final_avg = 0
    # avg_results: key = folder, val = list of its avg exponent values
    avg_results = {}
    for folder in exp_results.keys():
        dir_exp_val = 0
        dir_exp_len = 0
        for vals in exp_results[folder]:
            dir_exp_val += vals[0]
            dir_exp_len += vals[1]
        avg_results[folder] = np.float64(dir_exp_val / dir_exp_len)
        tot_exp_val += dir_exp_val
        tot_exp_len += dir_exp_len
    final_avg = int(tot_exp_val/tot_exp_len)

    # Alert user if average value has extreme decrepancy from min/max values
    # CHANGE 'decrepancty_val' VAR TO DECREASE/INCREASE DECREPANCY VALUE
    decrepancty_val = 3
    if(abs(min_exp + max_exp) / 2 - final_avg >= decrepancty_val):
        print("WARNING: Resulting avg exponent value has large differences with min/max exponents")
        print(f"\tResulting avg exponent: {final_avg}")
        print(f"\tSmallest exponent: {min_exp}")
        print(f"\tLargest exponent: {max_exp}")
    return final_avg