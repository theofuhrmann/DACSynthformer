import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import dac
import torch.nn.functional as F  # for the integer to one-hot
import pandas as pd

# ------     -------------------------------------
# Length of num_classes + num_params must be divisible by numheads 
class_name_to_int = {
    'DSPistons': 0, 'pistons': 0,
    'DSWind': 1, 'wind': 1,
    'DSApplause': 2, 'applause': 2,
    'DSBugs': 3, 'bees': 3,
    #'ChirpPattern': 4, 'chirps': 4,
    'FM': 4, 'fm': 4,
    'TokWotalDuet': 5, 'toks': 5,
    'DSPeepers': 6, 'peeps': 6
}

int2classname = {
    0: 'pistons',
    1: 'wind',
    2: 'applause',
    3: 'bees',
    #4: 'chirps',
    4: 'fm',
    5: 'toks',
    6: 'peeps'
}
_, num_classes = list(class_name_to_int.items())[-1]
num_classes = num_classes + 1

def getNumClasses():
    return num_classes 

def onehot(class_name):
    class_num = class_name_to_int.get(class_name, -1)  # Return -1 if class_name not found
    if class_num == -1:
        print(f'class_name not found: {class_name}')
    return F.one_hot(torch.tensor(class_num), num_classes=num_classes).to(torch.float)

            
# ========   and the DataSet ===============================================================

class CustomDACDataset(Dataset):
    def __init__(self, data_dir, metadata_excel, transforms=None):
        """
        Args:
            data_dir (string): Directory with all the data files.
            metadata_excel (string): Path to the Excel file containing file metadata.
                                     The Excel file must have columns:
                                     'Full File Name', 'Class Name', and 'Param1'.
            transforms (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_dir = data_dir
        # Load metadata from the Excel file using Pandas
        self.metadata_df = pd.read_excel(metadata_excel)
        # Extract the list of file names from the 'Full File Name' column
        self.file_names = self.metadata_df["Full File Name"].tolist()
        # Create a mapping from file name to its metadata (as a dict)
        self.metadata_dict = self.metadata_df.set_index("Full File Name").to_dict(orient="index")
        self.transforms = transforms

    def __len__(self):
        return len(self.file_names)
    
    def extract_conditioning_vector(self, filename):
        """
        Retrieves the conditioning vector for a given file using metadata from the Excel file.
        Uses the 'Class Name' and 'Param1' columns.
        """
        metadata = self.metadata_dict.get(filename, None)
        if metadata is None:
            raise ValueError(f"Metadata for file {filename} not found in the Excel file")
        class_name = metadata["Class Name"]
        param_value = metadata["Param1"]
        one_hot_fvector = onehot(class_name)
        return torch.cat((one_hot_fvector, torch.tensor([param_value])))
                         
    # RETURNS input_data, target_data, conditioning_vector
    def __getitem__(self, idx):
        filename = self.file_names[idx]
        fpath = os.path.join(self.data_dir, filename)
        dacfile = dac.DACFile.load(fpath)  # Load the data file
        data = dacfile.codes

        # Assuming data is a tensor of shape [1, N, T],
        # Remove the first dimension to get a tensor of shape [N, T]
        data = data.squeeze(0)

        # The input is the data itself: all time steps except the last one
        input_data = data[:, :-1]
        # The target is the data shifted by one time step: all time steps except the first one
        target_data = data[:, 1:]
        
        condvect = self.extract_conditioning_vector(filename)
        
        # Transpose the data so we get [T, N] for the transformer
        return input_data.transpose(0, 1), target_data.transpose(0, 1), condvect
