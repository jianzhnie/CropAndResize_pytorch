from torch.utils.data import Dataset
import numpy as np 
import os 
import json
from PIL import Image
import torch 


class XMCData(Dataset):
    """
    Custom dataset for XMC
    input:
        data_dir: data root path
        json_path: detected bbox
        filename_path: filenamelist for train
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        filename_list = [x.strip() for x in os.listdir(data_dir)]

        self.X_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, self.X_train[index]))
        image = image.convert('RGB')

        image = np.array(image, dtype = np.float32)
        image = torch.from_numpy(image.transpose((2,0,1)))

        return image, self.X_train[index]

    def __len__(self):
        return self.length