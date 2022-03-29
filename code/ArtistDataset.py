from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
import cv2
import pandas as pd
import torch

# In[15]:
class ArtistDataset(Dataset):
    
    def __init__(self, base_path, metadata, transforms=None):
        self.base_path = base_path
        self.data = glob(self.base_path)
        self.metadata = metadata
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return img, torch.tensor(self.metadata[self.metadata.new_filename == img_path.split('\\')[-1]]['artist_cat'].to_numpy()[0])
    
    def __len__(self):
        return len(self.data)
        