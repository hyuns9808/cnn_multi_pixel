import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

''' 
Class that creates a dataset with given traces
Each dataset represents a single folder
Input for initialization:
    1) strip_val: value before 1e
    2) e_val: 1e exponent value
    ex) if input string is "-2.17498915e-03", strip_val = "-2.17498915", e_val = "-03"
    3) e_exp: defined hyperparameter. Normalizes all values by pow(10, -1*e_exp)
    Default value is set to 4.
Returns:
    1) new_number: normalized float64 of string
'''
class TraceDataset(Dataset):
    # raw_traces: dictionary;
    # Key: folder name
    # Value: array of array of np.float32; each array is the raw np.float32 values from trace files
    raw_traces = {}
    trace_list = []

    # file_list: list of FILE NAMES that have been converted
    # cache: actual traces saved that can be reused
    def __init__(self, file_list, avg_exp, cache=True):
        self.file_list = file_list
        self.cache     = cache
        self.avg_exp   = avg_exp

    def __len__():
        return
    
    def __getitem__(self, index):
        return