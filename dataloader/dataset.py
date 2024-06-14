import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import dac
import torch.nn.functional as F # for the integer to one-hot

# --------- First, a few utilities ---------------------------------------------------------

#file names look like this: ClassName--ParamName-01.00--c-02--x-88.dac
def extract_classname_and_param_value(filename):
    # Regex pattern to match the required parts of the filename
    pattern = r'^([^--]+)--[^-]+-(\d+\.\d+)-'
    match = re.search(pattern, filename)

    if match:
        class_name = match.group(1)
        param_value = float(match.group(2))
        return class_name, param_value
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern")

# ------     -------------------------------------
# Length of num_classes + num_params must be divisible by numheads 
class_name_to_int = {
    'DSPistons': 0, 'pistons': 0,
    'DSWind': 1, 'wind': 1,
    'DSApplause': 2, 'applause': 2,
    'DSBugs': 3, 'bees': 3,
    'Foo': 4,
    'Bar': 5,
    'Baz': 6
}
_, num_classes = list(class_name_to_int.items())[-1] 
num_classes=num_classes+1

def getNumClasses() :
    return num_classes 

def onehot(class_name) :
    class_num=class_name_to_int.get(class_name, -1)  # Return -1 if class_name not found
    # Convert to one-hot encoded vector
    return F.one_hot(torch.tensor(class_num), num_classes=num_classes).to(torch.float)

            
# ========   and the DataSet ===============================================================

class CustomDACDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        Args:
            data_dir (string): Directory with all the data files.
        """
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.transforms=transforms

    def __len__(self):
        return len(self.file_names)
    
        
    def extract_conditioning_vector(self, filename):
        class_name, param_value = extract_classname_and_param_value(filename)
        one_hot_fvector=onehot(class_name)
        return torch.cat((one_hot_fvector, torch.tensor([param_value])))
                         

    # RETRUNS inputdata, outputdata, conditioning_vector
    def __getitem__(self, idx):
        fpath = os.path.join(self.data_dir, self.file_names[idx])
        dacfile=dac.DACFile.load(fpath)  # Load the data file
        data = dacfile.codes

        # Assuming data is a tensor of shape [1, N, T]
        # We remove the first dimension to get a tensor of shape [N, T]
        data = data.squeeze(0)

        # The input is the data itself
        input_data = data[:, :-1]  # All time steps except the last one
        # The target is the data shifted by one time step
        target_data = data[:, 1:]  # All time steps except the first one
        
        condvect=self.extract_conditioning_vector(self.file_names[idx])
        
        #print(f'condvect = {condvect}')

        # Transpose the last dimensions so we get [T, N] for the transformer
        return input_data.transpose(0, 1), target_data.transpose(0, 1), condvect

