from collections import defaultdict
import numpy as np
import os

''' 
Function that extracts all exponent values from a SINGLE power trace file
Input:
    1) filename: string; name of power trace file
Returns:
    1) avg_exponent: float; average exponent of all power trace values
'''
def extract_exponents_from_file(file_path):
    exponents = []
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
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
Function that runs through all trace files within given folder directory to get average
exponent values
Input:
    1) trace_folder_dir: string; raw path to trace folder directory
    2) target_folders: array; list of trace folders to be used in training/testing
Returns:
    1) final_avg: int; average exponent value of all folders
'''
def get_avg_exponent(trace_folder_dir, target_folders):
    exp_results = defaultdict(list)
    min_val = None
    max_val = None

    # exp_results: key = each dir, items = list of (sum(exponent), len(exponent)) tuples per file
    for folder in target_folders:
        folder_path = os.path.join(trace_folder_dir, folder)
        if os.path.exists(folder_path):
            folder_name = os.path.basename(folder_path)
            print(f"Handling: {folder_name}...")
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    temp = extract_exponents_from_file(file_path)
                    if min_val is None or min_val > min(temp):
                        min_val = min(temp)
                    if max_val is None or max_val < max(temp):
                        max_val = max(temp)
                    exp_results[folder_name].append((sum(temp), len(temp)))
        else:
            print(f"Trace folder does not exist: {folder_path}")

    tot_exp_val = 0
    tot_exp_len = 0
    final_avg = 0
    # avg_results: key = dir, val = list of its avg exponent values
    avg_results = {}
    for k in exp_results.keys():
        dir_exp_val = 0
        dir_exp_len = 0
        for vals in exp_results[k]:
            dir_exp_val += vals[0]
            dir_exp_len += vals[1]
        avg_results[k] = np.float64(dir_exp_val / dir_exp_len)
        tot_exp_val += dir_exp_val
        tot_exp_len += dir_exp_len
    final_avg = int(tot_exp_val/tot_exp_len)
    print(f"FINAL AVERAGE VALUE: {final_avg}")
    print(f"SMALLEST VALUE: {min_val}")
    print(f"LARGEST VALUE: {max_val}")
    return final_avg