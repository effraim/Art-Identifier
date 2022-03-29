import wandb
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
from ArtistDataset import ArtistDataset
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


def main():
    
    #Weight and Biases
    torch.manual_seed(0) # to fix the split result
    
    wandb.init(project="my-test-project", entity="frem",save_code=True)
    
    CONFIG =  dict(
        model_conf= "resnet50",
        lr_conf= 0.001,
        max_epochs= 100,
        batch_size= 32,
        optimizer= "Adam",
        loss_conf= "CrossEntropyLoss"
    )
    wandb.config = CONFIG
    
    print("\tWANDB SET UP DONE")
    
    
    # TRANSFORMS
    transforms = albumentations.Compose(
        [A.Normalize(),
         A.Downscale(p=0.3),#cuz overfit
         A.Blur(blur_limit=5,p=0.4),#cuz overfit
         A.Flip(),
         A.geometric.Resize(224, 224),
         ToTensorV2()]
    )
    
    val_transforms = albumentations.Compose(
        [A.Normalize(),
         A.geometric.Resize(224, 224),
         ToTensorV2()]
    )
    
    # DATA SET UP
    base_path = "D:\Art DataBase"
    train_path = os.path.join(base_path, 'train', '*')
    test_path = os.path.join(base_path, 'test', '*')
    test_data = glob(test_path)
    train_data = glob(train_path)
    # print(type(train_data))
    metadata = pd.read_csv(os.path.join(base_path, 'metadata.csv'))
    #The fit method(standardization) is calculating the mean and variance of 
    #each of the features
    #present in our data. The transform
    #method is transforming all the features using the respective mean and variance.
    #labelencoder = Encode target labels with value between 0 and n_classes-1.
    enc = LabelEncoder()
    metadata['artist_cat'] = enc.fit_transform(metadata['artist'].astype(str))
    metadata['style_cat'] = enc.fit_transform(metadata['style'].astype(str))
    metadata['filename'] = enc.fit_transform(metadata['new_filename'].astype(str))
    
    train_dataset = ArtistDataset(train_path, metadata, transforms=transforms)
    test_dataset = ArtistDataset(test_path, metadata, transforms=transforms)
    val_dataset = ArtistDataset(train_path,metadata, transforms=val_transforms)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False)
    print("\tDATA SET UP DONE")
    
    
    # MODEL SET UP
    gpu = torch.device('cuda:0')
    model = timm.create_model(model_name=CONFIG["model_conf"], pretrained=True)
    last_fc = nn.Linear(in_features=2048, out_features=2319)
    model.fc = last_fc
    params = model.parameters()
    model = model.to(gpu)
    print("\tMODEL SET UP DONE")
    
    
    # TRAINING SET UP
    #optimizer = optim.SGD(params, lr = 1e-2)
    
    optimizer = torch.optim.Adam(params, lr=CONFIG["lr_conf"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss = nn.CrossEntropyLoss()
    
    # Training and Validation
    print("\tSTARTING TRAINING AND VALIDATION LOOP")
    n_epochs = 30
    model.load_state_dict(torch.load(r'D:\Art DataBase\models\desert universe\resnet50_6.pth'))
    for epoch in range(n_epochs):
        # TRAIN LOOP
        model = model.train() #because of dropout in train we want it
        train_losses = list()
        train_accuracies = list()
        wandb.watch(model)
        for i,data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(gpu), labels.to(gpu)
    
            labels = labels.long()
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            outputs = model(inputs)
    
            J = loss(outputs, labels)
    
            J.backward()
            optimizer.step()
    
            train_batch_loss = J.cpu().item()
            train_losses.append(train_batch_loss)
    
            train_batch_acc = labels.eq(outputs.detach().argmax(dim=1)).float().mean().cpu()
            train_accuracies.append(train_batch_acc)
    
            wandb.log({
                "train_batch_loss": train_batch_loss,
                "train_batch_acc": train_batch_acc
            })
    
        wandb.log({
            "epoch_train_loss": torch.tensor(train_losses).mean(),
            "epoch_train_acc": torch.tensor(train_accuracies).mean()
        }) 
    
    
        # VALIDATION LOOP
        model = model.eval()#because of dropout in eval we dont
        val_losses = list()
        val_accuracies = list()
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                inputs, labels = data
                labels = labels.long()
                inputs, labels = inputs.to(gpu), labels.to(gpu)
                outputs = model(inputs)
    
                #2 compute objecttive function (howw well the network does)
                J = loss(outputs, labels)
    
                batch_val_loss = J.cpu().item()
                val_losses.append(batch_val_loss)
    
                batch_val_acc = labels.eq(outputs.detach().argmax(dim=1)).float().mean().cpu()
                val_accuracies.append(batch_val_acc)
                
                #EarlyStopping(losses_eval)
                patience=5 
                min_delta=0
                counter = 0
                best_loss = None
                early_stop = False
                if best_loss == None:
                        best_loss = J.cpu().item()
                elif best_loss - J.cpu().item() > min_delta:
                    best_loss = J.cpu().item()
                    # reset counter if validation loss improves
                    counter = 0
                elif best_loss - J.cpu().item() < min_delta:
                    counter += 1
                    print(f"INFO: Early stopping counter {counter} of {patience}")
                    if counter >= patience:
                        print('INFO: Early stopping')
                        early_stop = True
                
        wandb.log({
            "epoch_val_loss": torch.tensor(val_losses).mean(),
            "epoch_val_acc": torch.tensor(val_accuracies).mean()
        })
        
        torch.save(model.state_dict(), os.path.join('D:\Art DataBase\models',f"{CONFIG['model_conf']}_{epoch+7}.pth"))
        wandb.save(os.path.join('D:\Art DataBase\models', f"{CONFIG['model_conf']}_{epoch+7}.pth"))
        
        if early_stop:
            print("early")
            break
        
if __name__=="__main__":
    main()
