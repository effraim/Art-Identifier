from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import pandas as pd
import torch
import os
import numpy
from sklearn.preprocessing import LabelEncoder
from random import shuffle


class ArtistData:
    def __init__(
        self,
        base_path,
        metadata_csv,
        val_perc,
        batch_size,
        sample=1,
        label_encode=False,
        train_transforms=None,
        val_transforms=None,
    ):
        # DATA SET UP
        self.base_path = base_path
        self.train_path = os.path.join(base_path, "train", "*")
        self.test_path = os.path.join(base_path, "test", "*")
        self.train_data = glob(self.train_path)
        self.test_data = glob(self.test_path)
        self.val_perc = val_perc
        self.batch_size = batch_size
        self.metadata_csv = metadata_csv
        self.sample = sample
        self.label_encode = label_encode
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        
        self.metadata = pd.read_csv(os.path.join(self.base_path, self.metadata_csv))
        #to split data
        self.valid_imgs = self.metadata['new_filename'].apply(lambda x: x.split('.')[0]).tolist()

        # load all images that are in the metadata column "new filename"
        self.train_data = list(filter(self.check_that_image_is_valid, self.train_data))
        self.test_data = list(filter(self.check_that_image_is_valid, self.test_data))

        if sample != 1:
            #choice picks random
            s = int(len(self.train_data) * sample) 
            s1 = int(len(self.test_data) * sample)
     
        
            self.train_data = numpy.random.choice(self.train_data, size= s, replace=False)
            # do the same for test data
            self.test_data = numpy.random.choice(self.test_data, size=s1, replace=False)

        # print(type(train_data))
        
        if self.label_encode == True:
            # The fit method(standardization) is calculating the mean and variance of
            # each of the features
            # present in our data. The transform
            # method is transforming all the features using the respective mean and variance.
            # labelencoder = Encode target labels with value between 0 and n_classes-1.
            enc = LabelEncoder()
            self.metadata["artist_cat"] = enc.fit_transform(
                self.metadata["artist"].astype(str)
            )
            self.metadata["style_cat"] = enc.fit_transform(
                self.metadata["style"].astype(str)
            )
            self.metadata["filename"] = enc.fit_transform(
                self.metadata["new_filename"].astype(str)
            )

        shuffle(self.train_data)
        shuffle(self.test_data)
        self.train_data, self.val_data = (
            self.train_data[int(len(self.train_data) * self.val_perc) :],
            self.train_data[: int(len(self.train_data) * self.val_perc)],
        )
        self.train_dataset = ArtistDataset(
            self.train_data, self.metadata, transforms=self.train_transforms
        )
        self.val_dataset = ArtistDataset(
            self.val_data, self.metadata, transforms=self.val_transforms
        )
        self.test_dataset = ArtistDataset(
            self.test_data, self.metadata, transforms=self.val_transforms
        )

    def train_loader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_loader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_loader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True
        )
    
    #to split data
    def check_that_image_is_valid(self, img):
        img_num = img.split("\\")[-1].split('.')[0]
        return img_num in self.valid_imgs



class ArtistDataset(Dataset):
    def __init__(self, data, metadata, transforms=None):
        self.data = data
        self.metadata = metadata
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
            
        return (
            img,
            torch.tensor(
                self.metadata[self.metadata.new_filename == img_path.split("\\")[-1]]["artist_cat"].to_numpy()[0]
            ),
        )

    def __len__(self):
        return len(self.data)
