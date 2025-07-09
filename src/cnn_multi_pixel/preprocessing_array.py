import numpy as np
import re
import os

from torch import norm
from subsampler import sample_file

def create_array(file_path):
    ''' 
    Function that extracts all values from a SINGLE power trace file and saves it as a np.array
    Does NOT normalize the values; stores RAW traces
    Input:
        1) file_path: string; name of power trace file
    Returns:
        1) np.array(trace_array): np.array; np.array of all values saved as np.float32
    '''
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

def create_trace_arrays(trace_root, trace_dict, sampling=False, sample_info=None):
    ''' 
    Function that creates list of np arrays for given files
    Input:
        1) trace_root: string; Raw path to trace folder directory
        2) trace_dict: dictionary;
        Key: string; folder name
        Item: list; list of file names within folder
    Returns:
        1) traces: dictionary;
        Key: string; Folder name used to create trace arrays
        Value: tuple; tuple[0] = file name
        tuple[1] = array of np.float32; Array of normalized values converted to np.float32
    '''
    traces = {}
    # Get traces from all given folders
    for folder_name, file_list in trace_dict.items():
        folder_path = os.path.join(trace_root, folder_name)
        if os.path.exists(folder_path):
            print(f"Handling: {folder_name}...")
            folder_traces = []
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                if sampling == True:
                    try:
                        sample_interval, max_samples, sample_mode, column = sample_info
                        folder_traces.append(sample_file(file_path, sample_interval, max_samples, sample_mode, column))
                    except Exception as e:
                        raise(f"ERROR: {e}")
                else:
                    folder_traces.append(create_array(file_path))
            traces[folder_name] = (file_name, folder_traces)
        else:
            raise KeyError(f"Trace folder does not exist: {folder_path}")
    return traces