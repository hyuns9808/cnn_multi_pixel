''' 
Hyperparameters used to define the nature of the power trace files and values used for creating the dataloaders.
Double-check if these values match the format of the power trace files; failure to do so may cause unforeseen errors which are extremely difficult to debug.
    1) adc_num: int; Number of ADCs to create.
    2) adc_bandwidth: int; Number of bits each ADC stores.
    3) train_batch: int; Training dataloader batch size
    4) split_digital: boolean; Set to 'True' if the dataloaders need the digital output values of individual ADCs.
    5) normalized_digital: boolean; Set to 'True' if the file names provide normalized digital values.
'''
adc_num = 5
adc_bitwidth = 8
train_batch = 512
split_digital = True
normalized_digital = True

'''
Hyperparameter used to desginate trace folders to be loaded.
All power trace files are assumed to be saved in a folder called 'trace_files' within the root directory of the notebook.
Define names of all folders within 'trace_files' to be added to the dataloaders within this hyperparameter.
    1) sub_folder: boolean; Set to 'True' trace folders are stored in a specific
    folder within the parent directory.
    2) sub_folder_name: string; sub_folder_name: string; Name of sub-folder.
    If trace folders are stored in '/trace_folders/analog_5px,'
    set 'sub_folder' to 'True' and 'sub_folder_name' to 'analog_5px.'
    If there no subfolders, this value is not used.
    3) train_trace_folder_names: array of strings; Add names of folderss used for TRAINING dataset to this array.
    4) test_trace_folder_names: array of strings; Add names of folders used for TESTING dataset to this array.
    5) trace_type: string; Defines the type of trace file to be used.
    6) file_pattern: string; RegEx of file names to be checked. Number of groups MUST match 'adc_num'
'''
sub_folder = True
sub_folder_name = 'analog_5px'
train_trace_folder_names = ['analog_5px_tt_px','analog_5px_tt_pm','analog_5px_tt_xm','analog_5px_tt_xx']
test_trace_folder_names = ['analog_5_tt_x']
trace_type = "lin"
file_pattern = trace_type + "_s[^_]*_(?:\\d+(?:\\.\\d+)?_){4}\\d+(?:\\.\\d+)?\\.txt$"