import csv
import os
import pickle
import statistics
from glob import glob
from zipfile import ZipFile

import albumentations
import cv2
from django import conf
import numpy as np
import pandas as pd
import timm
import torch
from albumentations import augmentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.functional import accuracy
from torchvision import models, transforms

import wandb
from ArtistDataset import ArtistData
from argparser import get_argparser


parser = get_argparser()
args = parser.parse_args()
CONFIG = vars(args)
print(CONFIG)

# Weight and Biases
torch.manual_seed(0)  # to fix the split result

wandb.init(project="my-test-project", entity="frem", save_code=True, config=CONFIG)
wandb.config = CONFIG
print("\tWANDB SET UP DONE")


# TRANSFORMS
train_transforms = albumentations.Compose(
    [
        A.Downscale(p=0.3),  # cuz overfit
        A.Blur(blur_limit=5, p=0.4),  # cuz overfit
        A.Flip(),
        A.geometric.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transforms = albumentations.Compose(
    [A.geometric.Resize(224, 224), A.Normalize(), ToTensorV2()]
)


data = ArtistData(
    base_path="D:\Art DataBase",
    metadata_csv=r"C:\Users\pcem\Desktop\Thesis\metadata_preprocessed.csv",
    val_perc=0.2,
    batch_size=CONFIG["batch_size"],
    sample=1,
    label_encode=False,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
)

# MODEL SET UP
gpu = torch.device("cuda:0")
model = timm.create_model(model_name=CONFIG["model_conf"], pretrained=True)
last_fc = nn.Linear(in_features=2048, out_features=2319)
model.fc = last_fc
params = model.parameters()
model = model.to(gpu)
print("\tMODEL SET UP DONE")

train_loader = data.train_loader()
val_loader = data.val_loader()
test_loader = data.test_loader()

# TRAINING SET UP
# optimizer = optim.SGD(params, lr = 1e-2)

optimizer = torch.optim.Adam(
    params,
    lr=CONFIG["lr_conf"],
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5,
    amsgrad=False,
)
loss = nn.CrossEntropyLoss()

# Training and Validation
print("\tSTARTING TRAINING AND VALIDATION LOOP")
n_epochs = 30
# model.load_state_dict(torch.load(r'D:\Art DataBase\models\part2\resnet50_13.pth'))
for epoch in range(n_epochs):
    # TRAIN LOOP
    model = model.train()  # because of dropout in train we want it
    train_losses = list()
    train_accuracies = list()
    wandb.watch(model)
    for i, data in enumerate(train_loader):
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

        wandb.log(
            {"train_batch_loss": train_batch_loss, "train_batch_acc": train_batch_acc}
        )

    wandb.log(
        {
            "epoch_train_loss": torch.tensor(train_losses).mean(),
            "epoch_train_acc": torch.tensor(train_accuracies).mean(),
        }
    )

    # VALIDATION LOOP
    model = model.eval()  # because of dropout in eval we dont
    val_losses = list()
    val_accuracies = list()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            labels = labels.long()
            inputs, labels = inputs.to(gpu), labels.to(gpu)
            outputs = model(inputs)

            # 2 compute objecttive function (howw well the network does)
            J = loss(outputs, labels)

            batch_val_loss = J.cpu().item()
            val_losses.append(batch_val_loss)

            batch_val_acc = (
                labels.eq(outputs.detach().argmax(dim=1)).float().mean().cpu()
            )
            val_accuracies.append(batch_val_acc)

    # EarlyStopping(losses_eval)
    patience = 5
    min_delta = 0
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
            print("INFO: Early stopping")
            early_stop = True

    wandb.log(
        {
            "epoch_val_loss": torch.tensor(val_losses).mean(),
            "epoch_val_acc": torch.tensor(val_accuracies).mean(),
        }
    )

    torch.save(
        model.state_dict(),
        os.path.join(
            "D:\Art DataBase\models", f"{CONFIG['model_conf']}_{epoch+13}.pth"
        ),
    )
    wandb.save(
        os.path.join("D:\Art DataBase\models", f"{CONFIG['model_conf']}_{epoch+13}.pth")
    )

    if early_stop:
        print("early")
        break
