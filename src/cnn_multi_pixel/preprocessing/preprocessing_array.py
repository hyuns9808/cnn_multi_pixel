import numpy as np
import re
import os
from collections import defaultdict

from torch import norm

''' 
Function that extracts all time and trace values from a single power trace FILE and saves it as a list of (time, value) tuples
Does NOT normalize the values; stores RAW traces
Input:
    1) file_path: string; full path to file
Returns:
    1) trace_array: list of (time, value) tuples
    time_val_tuple[0]: np.float64; time value
    time_val_tuple[1]: np.float64; trace value
'''
def convert_file(file_path):
    trace_array = []
    first_line_pattern = re.compile(r'^\s*time\s+-i\(vdd\)\s*$')
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
             # Skip function call for first line if it matches
            if i == 0 and first_line_pattern.match(line):
                continue
            # Skip lines with fewer than 2 columns
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            time_str = parts[0]
            value_str = parts[1]
            # Check if obtained value is in valid scientific form
            if 'e' not in value_str:
                print(f"Skipping non-scientific value '{value_str}' in {file_path}")
                continue
            # Cast both values to np.float64 for best possible accuracy
            try:
                time = np.float64(time_str)
                value = np.float64(value_str)
            except ValueError:
                raise(f"Invalid format in '{value_str}' from {file_path}: Cannot store as np.float64.")
            time_val_tuple = (time, value)
            trace_array.append(time_val_tuple)
    return trace_array

''' 
Function that extracts all trace files from a single trace folder and stores them into a dictionary
Does NOT normalize the values; stores RAW traces
Input:
    1) trace_root: string; Full path to trace folder directory
    2) folder_name: string; Name of target folder
    3) valid_file_pattern: string; RegEx expression for correct file naming convention
Returns:
    1) folder_name: string; Name of converted folder
    2) folder_dict: dictionary; Each entry represents a single file within the folder
    Key: int; Digital value of file
    Value: list of tuples; list of time_val_tuple tuples of converted files
    time_val_tuple[0]: np.float64; time value
    time_val_tuple[1]: np.float64; trace value
'''
def convert_folder(trace_root, folder_name, valid_file_pattern):
    folder_dict = {}
    folder_path = os.path.join(trace_root, folder_name)
    if os.path.exists(folder_path):
        print(f"\tHandling folder \"{folder_name}\"...")
        for file in os.listdir(folder_path):
            match = valid_file_pattern.search(file)
            if match:
                digital_val = match.group(0)
                folder_dict[digital_val] = convert_file(os.path.join(folder_path, file))
    else:
        raise FileNotFoundError(f"ERROR - The following trace folder is missing: {folder_name} ")
    return folder_dict


def get_train_test_traces(trace_root, train_dict, test_dict):
    ''' 
    Function that gets traces from all training/testing folders
    Input:
        1) trace_root: string; Full path to trace folder directory
        2) train_pattern_dict: dictionary;
        Key: string; Folder name
        Value: string; RegEx pattern for correct file name where single resulting group is the digital value
        3) test_pattern_dict: dictionary;
        Key: string; Folder name
        Value: string; RegEx pattern for correct file name where single resulting group is the digital value
    Returns:
        1) train_traces: dictionary;
        Key: string; Training folder name
        Value: dictionary; Each entry represents a file within the target folder
            Key: int; Digital value of file
            Value: list of tuples; list of time_val_tuple tuples of converted files
            time_val_tuple[0]: np.float64; time value
            time_val_tuple[1]: np.float64; trace value
        2) test_traces: dictionary;
        Key: string; Testing folder name
        Value: dictionary; Each entry represents a file within the target folder
            Key: int; Digital value of file
            Value: list of tuples; list of time_val_tuple tuples of converted files
            time_val_tuple[0]: np.float64; time value
            time_val_tuple[1]: np.float64; trace value
    '''
    train_traces = {}
    test_traces = {}
    # First get traces for all training folders
    for folder_name, file_pattern in train_dict.items():
        train_traces.append(convert_folder(trace_root, folder_name, file_pattern))

    # Next, get traces for all testing folders
    for folder_name, file_pattern in test_dict.items():
        test_traces.append(convert_folder(trace_root, folder_name, file_pattern))
    
    return train_traces, test_traces