import numpy as np
from torch.utils.data import Dataset

''' 
Class that creates a dataset from (time, value) tuples. Each dataset represents a single trace folder.
    Label: Digital value
    Item:  Full list of (time, value) tuples
Input for initialization:
    1) folder_name: string; Name of folder where traces originate
    2) label_dict; dictionary;
    Key: string; file name
    Value: list of tuples; np.float64; trace value array
Returns:
    1) raw_dataset
'''
class TraceDatasetRaw(Dataset):
    def __init__(self, folder_name, time_val_dict):
        self.folder_name    = folder_name
        self.time_val_dict  = time_val_dict

    # Number of total files within folder
    def __len__(self):
        return len(self.time_val_dict.keys())
    
    # Get item via digital value
    def __getitem__(self, index):
        return self.time_val_dict[index]