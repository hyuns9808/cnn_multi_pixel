import numpy as np
import re
import os

''' 
Function that extracts all exponent values from a SINGLE power trace file
Input:
    1) file_path: string; name of power trace file
Returns:
    1) np.array(norm_array): np.array; np.array of all normalized values
'''
def create_norm_array(file_path, avg_exponent):
    norm_array = []
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
                value = int(value_str.split('e')[0])
                exponent = int(value_str.split('e')[1])
            except ValueError:
                print(f"Invalid exponent format in '{value_str}' from {file_path}")
                continue
            norm_array.append(np.float32(value*pow(10,exponent-avg_exponent)))
    return np.array(norm_array)



''' 
Function that creates list of np arrays for each train/test dataset
based on avg exponent value
Input:
    1) trace_root: string; raw path to trace folder directory
    2) train_folder_list: array; list of training trace folders to get avg exponent
    3) test_folder_list: array; list of testing trace folders to get avg exponent
    4) avg_exponent: 
Returns:
    1) avg_exponent: int; final avg exponent value used to normalize all trace values
'''
def create_trace_arrays(trace_root, train_folder_list, test_folder_list, avg_exponent):
    train_traces = []
    test_traces = []
    # First get traces for all training folders
    for train_folder in train_folder_list:
        folder_path = os.path.join(trace_root, train_folder)
        if os.path.exists(folder_path):
            print(f"Handling: {train_folder}...")
            folder_traces = []
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    # file_exponents = list of exponent values from given file 
                    folder_traces.append(create_norm_array(file_path))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
        train_traces.append(folder_traces)

    # Next, get traces for all testing folders
    for test_folder in test_folder_list:
        folder_path = os.path.join(trace_root, test_folder)
        if os.path.exists(folder_path):
            print(f"Handling: {test_folder}...")
            folder_traces = []
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    # file_exponents = list of exponent values from given file 
                    folder_traces.append(create_norm_array(file_path))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
        test_traces.append(folder_traces)
    
    return train_traces, test_traces