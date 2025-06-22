from collections import defaultdict
import re
import os

''' 
Function that gets all files from given folders that match the set file name format.
Input:
    1) trace_folder_dir: string; raw path to trace folder directory
    2) train_folder_list: array; list of training trace folders to get avg exponent
    3) test_folder_list: array; list of testing trace folders to get avg exponent
    4) file_pattern: string; RegEx expression used to 
Returns:
    1) train_match_dict: dictionary; all files with names that match given format
    Key: string; folder name
    Value: array of strings; array of names of all files that match given format of key folder
    2) test_match_dict: dictionary; all files with names that match given format
    Key: string; folder name
    Value: array of strings; array of names of all files that match given format of key folder
'''
def get_matching_files(trace_folder_dir, train_folder_list, test_folder_list, file_pattern):
    train_match_dict = defaultdict(list)
    test_match_dict = defaultdict(list)
    pattern = re.compile(rf"{file_pattern}")
    # CHANGE SEARCH TO MATCH AFTER MEETING
    # First get matches in training folder
    for folder_name in train_folder_list:
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            for file in os.listdir(folder_path):
                if pattern.search(file):
                    train_match_dict[folder_name].append(file)
        else:
            raise KeyError(f"Trace folder for TRAINING does not exist: {folder_path}")
    # Next get matches in testing folder
    for folder_name in test_folder_list:
        folder_path = os.path.join(trace_folder_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"\tHandling folder \"{folder_name}\"...")
            for file in os.listdir(folder_path):
                if pattern.search(file):
                    test_match_dict[folder_name].append(file)
        else:
            raise KeyError(f"Trace folder for TESTING does not exist: {folder_path}")
        
    return train_match_dict, test_match_dict