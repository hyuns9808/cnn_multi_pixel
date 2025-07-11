import numpy as np
# Change based on CNN pickle implementation
# ALSO need to change normalization on SAMPLED DATASET
import json

# Class used to normalize trace arrays
class LogNormalizer:
    def __init__(self, clip_min=1e-15):
        self.global_min = None
        self.global_max = None
        # Using 1e-15 as clipping np.float64 values
        self.clip_min = clip_min

    def fit_list(self, array_list):
        all_vals = np.concatenate(array_list)
        log_vals = np.log10(np.clip(all_vals, self.clip_min, None))
        self.global_min = log_vals.min()
        self.global_max = log_vals.max()
        
    def fit_test_train(self, train_dict, test_dict):
        all_vals = []
        for trace_values in train_dict.values():
            all_vals.append(trace_values)
        for trace_values in test_dict.values():
            all_vals.append(trace_values)
        all_vals = np.concatenate(all_vals)
        log_vals = np.log10(np.clip(all_vals, self.clip_min, None))
        self.global_min = log_vals.min()
        self.global_max = log_vals.max()

    def transform_list(self, arr):
        log_arr = np.log10(np.clip(arr, self.clip_min, None))
        return (log_arr - self.global_min) / (self.global_max - self.global_min + self.clip_min)
    
    def transform_test_train(self, train_dict, test_dict):
        for trace_folder, trace_values in train_dict.items():
            log_vals = np.log10(np.clip(trace_values, self.clip_min, None))
            train_dict[trace_folder] = (log_vals - self.global_min) / (self.global_max - self.global_min + self.clip_min)
        for trace_folder, trace_values in test_dict.items():
            log_vals = np.log10(np.clip(trace_values, self.clip_min, None))
            train_dict[trace_folder] = (log_vals - self.global_min) / (self.global_max - self.global_min + self.clip_min)
        return train_dict, test_dict

    def save(self, path):
        if self.global_max is None:
            raise KeyError("ERROR: global_max has not been set")
        if self.global_min is None:
            raise KeyError("ERROR: global_min has not been set")
        with open(path, 'w') as f:
            json.dump({'min': float(self.global_min), 'max': float(self.global_max)}, f)

    def load(self, path):
        with open(path, 'r') as f:
            d = json.load(f)
        self.global_min = d['min']
        self.global_max = d['max']