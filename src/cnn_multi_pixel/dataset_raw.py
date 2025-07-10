import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

''' 
Class that creates a dataset from raw trace arrays. Each dataset represents a single trace folder.
Runs the following operations:
    1) If sampling is required, run subsampling
    2) Create labels for each trace input
Input for initialization:
    1) folder_name: string; Name of folder where traces originate
    2) file_list: list of strings; List of file names within the folder
    3) label_dict; dictionary;
    Key: File names
    Value: Label for corresponding file
    4) avg_exp: int; Average exponent value obtained during file read process
Returns:
    1) new_number: normalized float64 of string
'''
class TraceDatasetRaw(Dataset):
    labelled_traces = {}
    trace_list    = []

    # file_list: list of FILE NAMES that have been converted
    def __init__(self, folder_name, file_list, label_dict, avg_exp):
        self.folder_name = folder_name
        self.file_list   = file_list
        self.label_dict  = label_dict
        self.avg_exp     = avg_exp

    def __len__():
        return
    
    def __getitem__(self, index):
        return