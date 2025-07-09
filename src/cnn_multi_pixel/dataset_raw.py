import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
# subsampler is for SINGLE file
from subsampler import sample_file
import re

# namedtuple used for fast trace identification
TraceInfo = namedtuple("TraceInfo", ["trace", "start", "stop"])
DTYPE = np.float32

class TraceDatasetRaw(Dataset):
    ''' 
    Class that creates a base dataset from raw trace np arrays.
    Assumes the given raw trace np array is subsampled and ready-to-go.
    Each dataset represents a single trace folder.
    Each trace is labelled with the full digital value given with the input folder.
    '''
    def __init__(self, folder_name, file_list, label_dict, avg_exp, is_sampled=False, device='cuda'):
        '''
        Input for initialization:
            1) folder_name: string; Name of folder where traces originate
            2) file_list: list of strings; List of file names within the folder
            3) label_dict; dictionary;
            Key: File names
            Value: Label for corresponding file
            4) avg_exp: int; Average exponent value obtained during file read process
        '''
        self.folder_name = folder_name
        self.file_list   = file_list
        self.label_dict  = label_dict
        self.avg_exp     = avg_exp
        self.is_sampled  = is_sampled
        self.device      = device
        self.start       = 0
        self.end         = 1

    def __len__(self):
        return self.end - self.start
    
    # Getter functions
    def __getitem__(self, index):
        return

    def get_info(self, index):
        return self.file_list[index]
    
    def get_label(self, label): return DTYPE(label)

    def get_by_label(self, label, index=0):
        index = self.label_dict[label][index]
        return self.get_item(index)[0]
    
    # Setter functions
    def set_range(self, start, end):
        self.start = start
        self.end   = end
        return self

    def set_prop_range(self, test, proportion):
        width = int(len(self.file_list) * proportion)
        start = len(self.file_list) - width if test else 0
        end   = start + width
        self.set_range(start, end)
        return self