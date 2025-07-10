from collections import defaultdict
import numpy as np
import re
import os

from normalizer import LogNormalizer

def extract_values_from_file(file_path):
    ''' 
    Function that extracts all values from a SINGLE power trace file
    Input:
        1) file_path: string; name of power trace file
    Returns:
        1) values: list; all trace values(np.float64) from file
    '''
    values = []
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
                # Using float64 for maximum accuracy in log transform/min-max norm
                values.append(np.float64(value_str))
            except ValueError:
                print(f"Invalid exponent format in '{value_str}' from {file_path}")
                continue
    return values

def create_folder_array(trace_folder_dir, match_dict):
    ''' 
    Function that runs through all trace files within given folder directory
    and creates list of trace values
    Input:
        1) trace_folder_dir: string; raw path to trace folder directory
        2) match_dict: dictionary; all files with names that match given format
        Key: string; folder name
        Value: array of strings; array of names of all files with valid file names of naming convention within key folder
    Returns:
        1) folder_file_values: array of array of floats; array of all trace value arrays within given folder
    '''
    folder_file_values = []
    # Get avg exponent in training traces
    for folder_name, file_list in match_dict.items():
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                folder_file_values.append(extract_values_from_file(file_path))
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
    return folder_file_values
        
def normalize_all(trace_folder_dir, train_folder_list, test_folder_list, file_pattern, normalizer):
    ''' 
    Function that normalizes all given traces
    Input:
        1) trace_folder_dir: string; Raw path to trace folder directory
        2) train_folder_list: array; List of training trace folders
        3) test_folder_list: array; List of testing trace folders
        4) global_minmax: tuple; Tuple containing saved global_min, global_max values
        Default is None; ONLY used when adding additional values 
    Returns:
        1) 
        2)
        3) global_max:
        4) global_min:
    '''
    all_vals = []
    # 1) Create dictionary where key is folder name and value is list of all matching file names per folder
    train_dict, test_dict = get_matching_files(trace_folder_dir, train_folder_list, test_folder_list, file_pattern)
    # 2) Per each folder in dictionary, change values to list of lists of trace np.float64 arrays
    for train_folder, train_folder_list in train_dict.items():
        train_dict[train_folder] = create_folder_array(trace_folder_dir, )
        
    # 2-1) If initial run or given normalizer is corrupt, create new normalizer based on given files
    if normalizer.global_min is None or normalizer.global_max is None:
        print("Normalizing all trace values...")
        new_normalizer = LogNormalizer()
        new_normalizer.fit_test_train(all_vals)
        train_dict, test_dict = new_normalizer.transform_test_train(train_dict, test_dict)
    # 2-2) If additional run to add more train/test files, use given normalizer to normalize new arrays
    else:
        return
    return