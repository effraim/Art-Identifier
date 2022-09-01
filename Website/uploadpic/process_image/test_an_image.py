#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import timm
import torchvision 
from torch import nn
from albumentations import augmentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms


# In[2]:


import torch
import albumentations
import os
import cv2
import csv
import timm
import pickle
import statistics
import numpy as np
import pandas as pd
from zipfile import ZipFile
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from zipfile import ZipFile
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import augmentations as A
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms
from torchmetrics.functional import accuracy


def results_pic(pic):
    # In[3]:

    meta_path = os.path.join("uploadpic","csv","metadata_preprocessed_200.csv")
    metadata = pd.read_csv(meta_path)


    # In[4]:


    enc = LabelEncoder()

    metadata['artist_cat'] = enc.fit_transform(metadata['artist'].astype(str))


    # In[5]:


    #print(metadata['artist_cat'])


    # In[6]:


    model = timm.create_model(model_name='resnet50d', pretrained=True)
    model.fc = nn.Sequential(
    #nn.Dropout(0.5),
    nn.Linear(in_features=2048, out_features=2319)
    )
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(r'resnet50d_18_an_200.pth'))
    else:
        model.load_state_dict(torch.load(r'resnet50d_18_an_200.pth', map_location=torch.device('cpu')))


    # In[7]:


    model.eval() 


    # In[8]:


    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] )
    ])


    # In[9]:


    img = Image.open(pic).convert('RGB')  # Load image as PIL.Image


    # In[10]:


    #plt.imshow(img)
    #plt.show()
    

    # In[11]:




    img_t = transform(img)


    # In[12]:


    batch_t = torch.unsqueeze(img_t, 0)


    # In[13]:


    out = model(batch_t)
    out = torch.softmax(out, dim=1).detach().numpy()#try
    #pred = torch.argmax(out, dim=1)
    preds = np.argpartition(out, -5)[0][-5:]#try
    finalpreds = enc.inverse_transform(preds).tolist()[::-1]
    #x = x.unsqueeze(0)  # Add batch dimension

    #output = model(x)  # Forward pass
    #pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    



    #predictions_test = enc.inverse_transform(pred) #reverse enc

    return finalpreds
    


