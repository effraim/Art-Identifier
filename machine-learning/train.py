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

wandb.init(project="FinalRuns", entity="frem", save_code=True, config=CONFIG)

wandb.config = CONFIG
print("\tWANDB SET UP DONE")


# TRANSFORMS

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

# MODEL SET UP
gpu = torch.device("cuda:0")
model = timm.create_model(model_name=CONFIG["model_conf"], pretrained=True)
if (CONFIG["model_conf"] == 'efficientnet_b0'):      
    last_fc = nn.Linear(in_features=1280, out_features=2319) 
    model.classifier = last_fc
elif (CONFIG["model_conf"] == 'xception'):
    last_fc = nn.Linear(in_features=2048, out_features=2319)
    model.fc = last_fc
elif (CONFIG["model_conf"] == 'vit_base_patch16_224'):
    last_fc = nn.Linear(in_features=768, out_features=2319)
    model.head = last_fc
elif (CONFIG["model_conf"] == 'resnet34'):
    model.fc = nn.Sequential(
    #nn.Dropout(0.5),
    nn.Linear(in_features=512, out_features=2319)
    )
elif (CONFIG["model_conf"] == 'resnet50d'):
    model.fc = nn.Sequential(
    #nn.Dropout(0.5),
    nn.Linear(in_features=2048, out_features=2319)
    )   
    print("no dropout")

else:
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features=2048, out_features=2319)
    )
    print("dropout")

#model.load_state_dict(torch.load(r'D:\Art DataBase\models\junl\0\resnet50d_5.pth'))


# #for fine tunining freeze
# paranum = 0
# for para in model.parameters():
        
#         paranum = paranum + 1
# paranum = paranum - (0.7 * paranum)
# paranum = int(paranum)
# ct = 0
# for param in model.parameters():
#     ct = ct +1

#     if ct < paranum : param.requires_grad = False

params = model.parameters()
model.to(gpu)
wandb.watch(model)
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=0.1
)
criterion = nn.CrossEntropyLoss()
# Training and Validation
print("\tSTARTING TRAINING AND VALIDATION LOOP")
for epoch in range(CONFIG['epochs']):
    wandb.log({"epoch": epoch})
    
    # TRAIN LOOP     
    model = model.train()  # because of dropout in train we want it
    train_losses = list()
    train_accuracies = list()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(gpu), labels.to(gpu)

        labels = labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        outputs = torch.softmax(outputs, dim=1)

        loss.backward()
        
        optimizer.step()

        train_batch_loss = loss.cpu().item()
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
            loss = criterion(outputs, labels)

            outputs = torch.softmax(outputs, dim=1)

            batch_val_loss = loss.cpu().item()

            val_losses.append(batch_val_loss)

            batch_val_acc = (
                labels.eq(outputs.detach().argmax(dim=1)).float().mean().cpu()
            )
            val_accuracies.append(batch_val_acc)

    if (epoch > 12):
            scheduler.step()
    #test
    lr=optimizer.param_groups[0]["lr"]
    print(lr)

    wandb.log(
        {
            "epoch_val_loss": torch.tensor(val_losses).mean(),
            "epoch_val_acc": torch.tensor(val_accuracies).mean(),
        }
    )
    
    torch.save(
        model.state_dict(),
        os.path.join(
            "D:\Art DataBase\models\kinal", f"{CONFIG['folder']}\{CONFIG['model_conf']}_{epoch}.pth" 
        ),
    )
    wandb.save(
        os.path.join("D:\Art DataBase\models\kinal", f"{CONFIG['folder']}\,{CONFIG['model_conf']}_{epoch}.pth" )
    )
  
# DO TESTINGH ERE
"""model.eval()
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
"""