from collections import defaultdict
import numpy as np
import re
import os

def get_matching_files(trace_folder_dir, train_folder_list, test_folder_list, file_pattern):
    ''' 
    Function that gets all files from given folders that match the set file name format.
    Input:
        1) trace_folder_dir: string; raw path to trace folder directory
        2) train_folder_list: array; list of training trace folders to get avg exponent
        3) test_folder_list: array; list of testing trace folders to get avg exponent
        4) file_pattern: string; RegEx expression used to 
    Returns:
        1) train_match_dict: dictionary; all files with names that match given format
        Key: string; folder name
        Value: array of strings; array of names of all files that match given format of key folder
        2) test_match_dict: dictionary; all files with names that match given format
        Key: string; folder name
        Value: array of strings; array of names of all files that match given format of key folder
    '''
    train_match_dict = defaultdict(list)
    test_match_dict = defaultdict(list)
    pattern = re.compile(rf"{file_pattern}")
    # First get matches in training folder
    for folder_name in train_folder_list:
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            for file in os.listdir(folder_path):
                if pattern.match(file):
                    train_match_dict[folder_name].append(file)
        else:
            raise KeyError(f"Trace folder for TRAINING does not exist: {folder_path}")
    # Next get matches in testing folder
    for folder_name in test_folder_list:
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            for file in os.listdir(folder_path):
                if pattern.match(file):
                    test_match_dict[folder_name].append(file)
        else:
            raise KeyError(f"Trace folder for TESTING does not exist: {folder_path}")
        
    return train_match_dict, test_match_dict

def extract_exponents_from_file(file_path):
    ''' 
    Function that extracts all exponent values from a SINGLE power trace file
    Input:
        1) file_path: string; name of power trace file
    Returns:
        1) exponents: list; all exponent values from file
    '''
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

def extract_exponents_from_folders(trace_folder_dir, match_dict, exp_results, min_exp, max_exp):
    ''' 
    Function that runs through all trace files within given folder directory
    and gets 1) sum of all exponent values, 2) number of traces
    Input:
        1) trace_folder_dir: string; raw path to trace folder directory
        2) train_match_dict: dictionary; all files with names that match given format
        Key: string; folder name
        Value: array of strings; array of names of all files that match given format of key folder
        3) test_match_dict: dictionary; all files with names that match given format
        Key: string; folder name
        Value: array of strings; array of names of all files that match given format of key folder
    Returns:
        1) exp_results: dictionary;
        key = each folder
        items = list of (sum(exponent), len(exponent)) tuples per files in folder
        2) min_exp: int; minimum exponent value
        3) min_exp: int; maximum exponent value
    '''
    # Get avg exponent in training traces
    for folder_name, file_list in match_dict.items():
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            folder_sum = 0
            folder_len = 0
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                # file_exponents = list of exponent values from given file 
                file_exponents = extract_exponents_from_file(file_path)
                folder_max = max(file_exponents)
                folder_min = min(file_exponents)
                exp_results[folder_name].append((sum(file_exponents), len(file_exponents)))
                folder_sum += sum(file_exponents)
                folder_len += len(file_exponents)
            print(f"\t\tFolder \"{folder_name}\" exponent results:")
            print(f"\t\tAverage exponent: {int(folder_sum / folder_len)}")
            print(f"\t\tMax exponent: {folder_max}")
            print(f"\t\tMinimum exponent: {folder_min}")
            
            if max_exp is None or max_exp < folder_max:
                max_exp = folder_max
            if min_exp is None or min_exp > folder_min:
                min_exp = folder_min
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
        
    return exp_results, min_exp, max_exp

def get_avg_exponent(trace_folder_dir, train_folder_list, test_folder_list):
    ''' 
    Function that gets final avg exponent from both training/testing trace files
    Input:
        1) trace_folder_dir: string; raw path to trace folder directory
        2) train_folder_list: array; list of training trace folders to get avg exponent
        3) test_folder_list: array; list of testing trace folders to get avg exponent
    Returns:
        1) avg_exponent: int; final avg exponent value used to normalize all trace values
    '''
    print("Obtaining avg exponent value for both train/test files...")
    # exp_results: dictionary;
    # Key: each folder
    # Value: list of (sum(exponent), len(exponent)) tuples per files in folder
    exp_results = defaultdict(list)
    # Values used to check max/min exponent of all folders
    # Used to alert user if average value is too off from these values
    min_exp = None
    max_exp = None

    # Combine train_folder_list and test_folder_list and get all exponents
    print("Analyzing exponent values of training trace files...")
    exp_results, min_exp, max_exp = extract_exponents_from_folders(trace_folder_dir, train_folder_list, exp_results, min_exp, max_exp)
    print("Analyzing exponent values of testing trace files...")
    exp_results, min_exp, max_exp = extract_exponents_from_folders(trace_folder_dir, test_folder_list, exp_results, min_exp, max_exp)
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