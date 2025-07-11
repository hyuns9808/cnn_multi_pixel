import numpy as np
import os
import torch
from torch.utils.data import Dataset
from src.cnn_multi_pixel.helper_functions.subsampler import sample_raw_dataset

class TraceDatasetSampled(Dataset):
    ''' 
    Class used to create sampled datasets. Each dataset represents a single trace folder.
    Dataset values via __getitem__:
        Index: int; Full digital value.
        Item:  list of np.float32; Full list of trace values of corresponding digital value.
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
    def __init__(self, folder_name, digital_to_val_dict):
        self.folder_name    = folder_name
        self.sampled_dict   = digital_to_val_dict

    # Number of total files within folder
    def __len__(self):
        return len(self.sampled_dict.keys())
    
    # Get item via digital value
    def __getitem__(self, digital_value):
        return self.sampled_dict[digital_value]
    
    # Get full sampled_dict
    def get_sampled_dict(self):
        return self.sampled_dict
    
def create_and_save_sampled_dataset(raw_sampled_dir, train_raw_datasets, test_raw_datasets, sample_info):
    ''' 
    Function used to create and save sampled datasets for both new training and testing trace folders.
    Assumed to be run right after 'create_and_save_raw_dataset'
    Input:
        1) raw_sampled_dir: string; Full path to directory where all sampled datasets are stored as .pt files.
        2) label_dict; dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    '''
    new_train_datasets = []
    new_test_datasets = []
    for raw_dataset in train_raw_datasets:
        sampled_dataset = sample_raw_dataset(raw_dataset)
        new_dataset_path = os.path.join(raw_save_dir, f"{folder_name}.pt")
        torch.save(new_dataset, new_dataset_path)
        new_train_datasets.append(new_dataset)
    for folder_name, file_dict in test_dict.items():
        new_dataset = TraceDatasetRaw(folder_name, file_dict)
        new_dataset_path = os.path.join(raw_save_dir, f"{folder_name}.pt")
        torch.save(new_dataset, new_dataset_path)
        new_test_datasets.append(new_dataset)
    return new_train_datasets, new_test_datasets

def load_sampled_dataset(folder_name):
    ''' 
    Function used to load saved raw datasets.
    Input:
        1) raw_save_dir: string; Full path to directory where all raw datasets are stored as .pt files.
        2) label_dict; dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    '''
    return