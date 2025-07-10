import numpy as np
import re
import os
from collections import defaultdict

from torch import norm

''' 
Function that extracts all values from a SINGLE power trace file and saves it as a np.array
Does NOT normalize the values; stores RAW traces
Input:
    1) file_path: string; name of power trace file
Returns:
    1) np.array(trace_array): np.array; np.array of all values saved as np.float32
'''
def create_array(file_path):
    trace_array = []
    pattern = re.compile(r'^\s*time\s+-i\(vdd\)\s*$')
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
             # Skip function call for first line if it matches
            if i == 0 and pattern.match(line):
                continue
            # Skip lines with fewer than 2 columns
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Check if obtained value is in valid scientific form
            value_str = parts[1]
            if 'e' not in value_str:
                print(f"Skipping non-scientific value '{value_str}' in {file_path}")
                continue
            # Cast to np.float32
            try:
                value = np.float32(value_str)
            except ValueError:
                raise(f"Invalid format in '{value_str}' from {file_path}: Cannot store as np.float32.")
            trace_array.append(value)
    return np.array(trace_array)

def create_raw_trace_arrays(trace_root, train_dict, test_dict, file_pattern):
    ''' 
    Function that converts text trace values to list of np arrays for each train/test dataset
    Input:
        1) trace_root: string; Raw path to trace folder directory
        2) train_folder_list: array; List of training trace folders to be converted
        3) test_folder_list: array; List of testing trace folders to be converted
        4) file_pattern: string; RegEx pattern for file name where single resulting group is the digital value
    Returns:
        1) train_traces: dictionary;
        Key: string; Training folder name used to create trace arrays
        Value: tuple; tuple[0] = digital value; parsed using file_pattern group
        tuple[1] = array of np.float32; raw trace values converted to np.float32
        2) test_traces: dictionary;
        Key: string; Testing folder name used to create trace arrays
        Value: tuple; tuple[0] = digital value; parsed using file_pattern group
        tuple[1] = array of np.float32; raw trace values converted to np.float32
    '''
    train_traces = defaultdict(list)
    test_traces = defaultdict(list)
    # First get traces for all training folders
    for folder_name, file_list in train_dict.items():
        folder_path = os.path.join(trace_root, folder_name)
        if os.path.exists(folder_path):
            print(f"Handling: {folder_name}...")
            folder_traces = []
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                folder_traces.append(create_array(file_path))
                match = re.match(file_pattern, file_name)
                if match:
                    raw_digital_val = match.group(1)
                else:
                    print(f"ERROR: Invalid file name format for - \"{file_name}\"")
                    continue
                train_traces[folder_name].append((raw_digital_val, folder_traces))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")  

    # Next, get traces for all testing folders
    for folder_name, file_list in test_dict.items():
        folder_path = os.path.join(trace_root, folder_name)
        if os.path.exists(folder_path):
            print(f"Handling: {folder_name}...")
            folder_traces = []
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                folder_traces.append(create_array(file_path))
                match = re.match(file_pattern, file_name)
                if match:
                    raw_digital_val = match.group(1)
                else:
                    print(f"ERROR: Invalid file name format for - \"{file_name}\"")
                    continue
                test_traces[folder_name].append((raw_digital_val, folder_traces))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")  
    
    return train_traces, test_traces