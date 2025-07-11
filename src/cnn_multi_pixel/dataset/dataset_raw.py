import numpy as np
import os
import gzip
import pickle
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
    
    # Get folder name
    def get_folder_name(self):
        return self.folder_name
    
    # Get full raw_dict
    def get_raw_dict(self):
        return self.raw_dict
    
def create_and_save_raw_datasets(raw_save_dir, total_raw_dict):
    ''' 
    Function used to create and save raw datasets for both new training and testing trace folders.
    Input:
        1) raw_save_dir: string; Full path to directory where all raw datasets are stored as .gz files.
        2) total_raw_dict; dictionary;
            Key: string; Target folder name
            Value: dictionary; Contents of all files within folder
                Key: string; file name
                Value: list of tuples; np.float64; trace value array
    Returns:
        1) new_datasets: list of TraceDatasetRaw objects; list of newly created raw datasets
    '''
    new_raw_datasets = []
    for folder_name, raw_dict in total_raw_dict.items():
        new_dataset = TraceDatasetRaw(folder_name, raw_dict)
        new_dataset_path = os.path.join(raw_save_dir, f"{folder_name}_raw.gz")
        save_data = {"folder_name": new_dataset.get_folder_name(), "raw_dict": new_dataset.get_raw_dict()}
        with gzip.open(new_dataset_path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        new_raw_datasets.append(new_dataset)
    return new_raw_datasets

def load_raw_datasets(raw_save_dir, folder_list):
    ''' 
    Function used to load saved raw datasets.
    Input:
        1) raw_save_dir: string; Full path to directory where all raw datasets are stored as .gz files.
        2) folder_list: list of strings; List of all raw dataset folder names to load
    Returns:
        1) load_dataset_list: list of datasets; List of all loaded raw datasets
    '''
    load_dataset_list = []
    for folder_name in folder_list:
        load_data_path = os.path.join(raw_save_dir, folder_name + "_raw.gz")
        with gzip.open(load_data_path, "rb") as f:
            load_data = pickle.load(f)
            load_dataset = TraceDatasetRaw(load_data["folder_name"], load_data["raw_dict"])
        load_dataset_list.append(load_dataset)
    return load_dataset_list