import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

''' 
Helper function used to normalize string values with given average exponent value
Input for initialization:
    1) strip_val: value before 1e
    2) e_val: 1e exponent value
    ex) if input string is "-2.17498915e-03", strip_val = "-2.17498915", e_val = "-03"
    3) avg_exp: obtained average exponent value.
Returns:
    1) new_number: normalized float32 value
'''
def process_string(strip_val, e_val, avg_exp):
    # e_diff: exponent difference from base -8 value
    e_diff = int(e_val) + avg_exp
    is_neg = strip_val.startswith('-')

    # Remove negative sign for processing
    if is_neg:
        strip_val = strip_val[1:]
    
    int_part, dec_part = strip_val.split('.')
    new_number = int_part + dec_part

    if e_diff < 0:
        new_number = '0' * abs(e_diff - 1) + new_number
        new_number = '0.' + new_number
    else:
        if e_diff + 1 < len(new_number):
            new_number = new_number[:e_diff + 1] + '.' + new_number[e_diff + 1:]
        else:
            new_number = new_number.ljust(e_diff + 1, '0')
    new_number = np.float32(new_number)

    return new_number * -1 if is_neg else new_number

''' 
Class that creates a dataset with given traces
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
    # cached_traces, trace_list = used to store already created traces
    # reused based on digital value, prevents rep calls on load_traces
    cached_traces = {}
    trace_list    = []

    # file_list: list of FILE NAMES that have been converted
    # cache: actual traces saved that can be reused
    def __init__(self, file_list, avg_exp, cache=True):
        self.file_list = file_list
        self.cache     = cache
        self.avg_exp   = avg_exp
    
    def __len__(self):
        return len(self.file_list)

    # input: index, used for finding file in file_list
    def __getitem__(self, index):
        fname, fpath, label = self.file_list[index]
        label = self.process_label(label)

        if self.cache and fname in self.cached_traces:
            return self.cached_traces[fname], label
        else:
            return self.load_trace(fname, fpath), label

    def get_info(self, index):
        return self.file_list[index]

    # opens single trace file, creates valu_arr, patches as tensor
    def load_trace(self, fname, fpath):
        with open(fpath, 'r') as file:
            header = file.readline()
            #time_arr = []
            valu_arr = []
            for line in file.readlines():
                _, value = line.strip().split()
                # Edge case: value is "-0.00000000e+00" or "0.00000000e+00"
                # Add more edge cases if needed
                if value in ["-0.00000000e+00", "0.00000000e+00"]:
                    valu_arr.append(np.float32(0))
                else:
                    try:
                        match = re.search(r"(?<=e-)\d+", value)
                        if match:
                            if value[0] == "-":
                                strip_val = value[0:11]
                                strip_val_e = value[12:15]
                            else:
                                strip_val = value[0:10]
                                strip_val_e = value[11:14]
                            valu_arr.append(process_string(strip_val, strip_val_e))
                    except ValueError as e:
                        print(f"Error parsing value '{value}': {e}")

        trace = np.array(valu_arr, dtype=np.float32)
        '''
        # Debugging scripts; do not erase
        print(f"\t{trace}")
        print(f"\tfname: {fname}")
        print(f"\tfpath: {fpath}\n")
        '''

        if self.cache: 
            self.cached_traces[fname] = trace
            self.trace_list.append(trace)

        return trace
    
    # label = PURE digital value as array
    def process_label(self, label):
        return label

    def cache_all(self):
        assert self.cache == True
        
        print("Caching all traces")
        for fname, fpath, label in self.file_list:
            self.load_trace(fname, fpath)
        print("DONE Caching all traces")

class TraceDatasetBW(TraceDataset):
    # bit_select = 0-7, 0 = LSB, 7 = MSB
    # adc_select = 0-4, 0 = ADC storing LSB, 4 = ADC storing MSB
    def __init__(self, file_list, bit_select, adc_select, cache=True, split_digital=False, normalize_digital=False):
        # if split_digital, need to create SEPARATE dataloaders PER ADC
        if split_digital:
            self.split_digital = True
            self.adc_num = adc_select
            self.bit_mask =  1 << bit_select
        else:
            self.split_digital = False
            self.adc_num = 0
            self.bit_mask =  1 << bit_select + adc_select * 8

        super().__init__(file_list, cache=cache) 
    
    # Uses bitwise on COMBINED label
    def process_label(self, label):
        # if split_digital, label = array
        if self.split_digital:
            try:
                label_num = label[self.adc_num]
            except IndexError:
                print(f"\tInvalid index; this may be caused due to bad hyperparameters.")
                print(f"\tadc_num: {adc_num}")
                print(f"\tsplit_digital: {split_digital}")
                print(f"\tnormalized_digital: {normalized_digital}")
        else:
            label_num = label
        return 1 if label_num & self.bit_mask else 0

class TraceDatasetBuilder:
    def __init__(self, adc_bitwidth=8, cache=True):
        self.file_list = []
        self.cache = cache
        self.adc_bits = adc_bitwidth

        self.dataset = None
        self.dataloader = None
        self.datasets = []
        self.dataloaders = []

    def add_files(self, directory, format, label_group):
        ''' Builds list of powertrace files
        Inputs:
            directory   : folder to search for files
            format      : regular expression to match filenames
            label_group : group index for digital output label corresponding to trace
        Outputs:
            list        : [(file_name, file_path, label) ... ]
        '''
        format = re.compile(format)
        fnames = os.listdir(directory)

        for fname in fnames:
            if match := format.match(fname):
                fpath = os.path.join(directory, fname)
                # IF split_digital, return ARRAY of digital values
                # returns: [[adc_num digital values], ...] (2D array)
                # ORDER: MSB values FIRST
                if split_digital:
                    # dvalue: ordered by FILE NAMING order
                    # if normalized, multiply 256 to get original value
                    if normalized_digital:
                        dvalue = [int(np.float64(i) * 256) for i in match.groups()]
                    # else, append as int
                    else:
                        dvalue = [int(i) for i in match.groups()]
                # ELSE, return ARRAY of SINGLE digital values
                # returns: [combined digital value, ...] (1D array)
                else:
                    dvalue = [0]
                    # if normalized, multiply 256 to get original value
                    if normalized_digital:
                        for i in match.groups():
                            dvalue[0] = dvalue[0] * 256 + int(np.float64(i) * 256)
                    # else, append as int 
                    else:
                        for i in match.groups():
                            dvalue[0]  = dvalue[0] * 256 + int(i)
                
                self.file_list.append((fname, fpath, dvalue))

    def build(self):
        # dataset = TraceDataset, trace - digital value label ONLY
        self.dataset = TraceDataset(self.file_list, cache=self.cache)
        # Append dataloaders IN LSB ORDER; dataloader[0] = adc[0], bit[0]
        # dataloader[39] = adc[4], bit[7]
        # adc_dataloader[adc_num] = [dataloader[adc_num*8+0], dataloader[adc_num*8+1], ..., dataloader[adc_num*8+7]]
        for adc in range(adc_num):
            for bit in range(self.adc_bits):
                self.datasets.append(TraceDatasetBW(self.file_list, bit, adc, cache=self.cache))
                
        if self.cache:
            self.dataset.cache_all()

    def build_dataloaders(self, **kwargs): # batch_size=256, shuffle=True
        self.dataloader = DataLoader(self.dataset, **kwargs)
        self.dataloaders = [DataLoader(dataset, **kwargs) for dataset in self.datasets]