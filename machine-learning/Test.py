import os
import wandb

import albumentations
import timm
import torch
from albumentations import augmentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn

from ArtistDataset import ArtistData
from argparser import get_argparser



parser = get_argparser()
args = parser.parse_args()
CONFIG = vars(args)
print(CONFIG)
# Weight and Biases
torch.manual_seed(0)  # to fix the split result

wandb.init(project="Testing", entity="frem", save_code=True, config=CONFIG)

wandb.config = CONFIG
print("\tWANDB SET UP DONE")

##
train_transforms = albumentations.Compose(
    [
        A.Downscale(p=0.3),  #1
        A.Blur(blur_limit=5, p=0.4),  # 2
        A.Flip(), #3
        A.geometric.Resize(224, 224),
        A.geometric.transforms.Affine(scale=(1, 0.9), translate_percent=(0, 0.2), rotate=(-10, 10), shear=(-10, 10), p=0.7),#4
        A.transforms.Emboss (alpha=(0.2, 0.9), strength=(0.2, 1.3), always_apply=False, p=0.5),#5
        A.Normalize(),
        ToTensorV2(),
    ]
)
val_transforms = albumentations.Compose(
    [A.geometric.Resize(224, 224), A.Normalize(), ToTensorV2()]
)

if CONFIG["artists_number"] == 200:
    csv_choice = metadata_choice = r"C:\Users\pcem\Desktop\Thesis\metadata_preprocessed_200.csv"
elif CONFIG["artists_number"] == 100:
    csv_choice = metadata_choice = r"C:\Users\pcem\Desktop\Thesis\metadata_preprocessed_100.csv"
elif CONFIG["artists_number"] == 50:
    csv_choice = metadata_choice = r"C:\Users\pcem\Desktop\Thesis\metadata_preprocessed_50.csv"
else:
    csv_choice = metadata_choice = r"C:\Users\pcem\Desktop\Thesis\metadata_preprocessed.csv"

data = ArtistData(
    base_path="D:\Art DataBase",
    metadata_csv=csv_choice,
    val_perc=0.2,
    batch_size=CONFIG["batch_size"],
    sample=CONFIG["sample"],
    label_encode=False,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
)

gpu = torch.device("cuda:0")

model = timm.create_model(model_name='resnet50d', pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=2319)
)   
params = model.parameters()
model.to(gpu)
wandb.watch(model)
print("\tMODEL SET UP DONE")

test_loader = data.test_loader()

optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss = nn.CrossEntropyLoss()

torch.cuda.is_available()
model.load_state_dict(torch.load(r'D:\Art DataBase\models\kinal\0\resnet50d_18.pth'))


model.eval()
correct5 = 0
correct3 = 0
correct = 0

total = 0
acc5 = 0
acc3 = 0
acc = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(gpu), labels.to(gpu)

        # calculate outputs by running images through the network
        outputs = model(images)
        #bale soft
        outputs = torch.softmax(outputs, dim=1)
        sorted_outs = torch.argsort(outputs, dim=1, descending=True)
        total += labels.size(0)
        for i, label in enumerate(labels):
            if label in sorted_outs[i][:5]:
                correct5 += 1
        for i, label in enumerate(labels):
            if label in sorted_outs[i][:3]:
                correct3 += 1    
        for i, label in enumerate(labels):
            if label in sorted_outs[i][:1]:
                correct += 1     
acc5 = 100 * correct5 / total #eixe 2 /
acc3 = 100 * correct3 / total
acc = 100 * correct / total
print(f'Accuracy of the network test images top 1 : {acc} %')
print(f'Accuracy of the network test images top 3 : {acc3} %')
print(f'Accuracy of the network test images top 5: {acc5} %')
wandb.log({
    "accuracy" : acc
})
wandb.log({
    "accuracy top 3" : acc3
})
wandb.log({
    "accuracy top 5" : acc5
})
"""model.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(gpu), labels.to(gpu)

        # calculate outputs by running images through the network
        outputs = model(images)
        #bale soft
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network test images: {100 * correct // total} %')"""
