import numpy as np
import os
import torch
from torch.utils.data import Dataset

class TraceDatasetRaw(Dataset):
    ''' 
    Class used to create a raw dataset. Each dataset represents a single trace folder.
    Dataset values via __getitem__:
        Index: int; Full digital value.
        Item:  list of tuples; Full list of (time, value) tuples of corresponding digital value.
    Attributes:
        1) self.folder_name: string; Name of folder. Used to identify raw dataset.
        2) self.digital_dict: dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    Input for initialization:
        1) folder_name: string; Name of folder where traces originate
        2) label_dict; dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    '''
    def __init__(self, folder_name, digital_to_time_val_dict):
        self.folder_name    = folder_name
        self.raw_dict       = digital_to_time_val_dict

    # Number of total files within folder
    def __len__(self):
        return len(self.raw_dict.keys())
    
    # Get item via digital value
    def __getitem__(self, digital_value):
        return self.raw_dict[digital_value]
    
    # Get full raw_dict
    def get_raw_dict(self):
        return self.raw_dict
    
def create_and_save_raw_dataset(raw_save_dir, train_raw_dict, test_raw_dict):
    ''' 
    Function used to create and save raw datasets for both new training and testing trace folders.
    Input:
        1) raw_save_dir: string; Full path to directory where all raw datasets are stored as .pt files.
        2) train_raw_dict; dictionary;
            Key: string; Target folder name
            Value: dictionary; Contents of all files within folder
                Key: string; file name
                Value: list of tuples; np.float64; trace value array
        3) test_raw_dict; dictionary;
            Key: string; Target folder name
            Value: dictionary; Contents of all files within folder
                Key: string; file name
                Value: list of tuples; np.float64; trace value array
    Returns:
        1) new_train_datasets: list of TraceDatasetRaw objects; list of newly created raw datasets for training
        2) new_test_datasets: list of TraceDatasetRaw objects; list of newly created raw datasets for testing
    '''
    new_train_datasets = []
    new_test_datasets = []
    for folder_name, raw_dict in train_raw_dict.items():
        new_dataset = TraceDatasetRaw(folder_name, raw_dict)
        new_dataset_path = os.path.join(raw_save_dir, f"{folder_name}.pt")
        torch.save(new_dataset, new_dataset_path)
        new_train_datasets.append(new_dataset)
    for folder_name, raw_dict in test_raw_dict.items():
        new_dataset = TraceDatasetRaw(folder_name, raw_dict)
        new_dataset_path = os.path.join(raw_save_dir, f"{folder_name}.pt")
        torch.save(new_dataset, new_dataset_path)
        new_test_datasets.append(new_dataset)
    return new_train_datasets, new_test_datasets

def load_raw_dataset(folder_name):
    ''' 
    Function used to load saved raw datasets.
    Input:
        1) raw_save_dir: string; Full path to directory where all raw datasets are stored as .pt files.
        2) label_dict; dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    '''
    return