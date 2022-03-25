import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,random_split
import torch.nn.functional as F
from collections import OrderedDict
from cutout import Cutout
import csv
import pandas as pd
from project1_model import project1_model

data_statistics = ([0.4914, 0.4822, 0.4465],[0.2023,0.1994,0.2010])

test_transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*data_statistics, inplace=True)
])

init_dataset = torchvision.datasets.CIFAR10(root="data/", download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data/",download=True, train =False, transform=test_transform_cifar)
aug_dataset = init_dataset
aug_dataset.transform= transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*data_statistics, inplace=True),
    Cutout(n_holes=1, length=8)
])
dataset = torch.utils.data.ConcatDataset([aug_dataset,init_dataset])

val_ratio = 0.2 
train_dataset, val_dataset = random_split(dataset, [int((1-val_ratio)* len(dataset)), int(val_ratio * len(dataset))])
batch_size = 400
train_dl = DataLoader(train_dataset, batch_size, shuffle = True, pin_memory = True)
val_dl = DataLoader(val_dataset, batch_size, shuffle = True, pin_memory = True)
test_dl = DataLoader(test_dataset, batch_size, shuffle = True, pin_memory = True)

def denormalizer(images, means, std_devs):
    means = torch.tensor(means).reshape(1,3,1,1)
    std_devs = torch.tensor(std_devs).reshape(1,3,1,1)
    return images * std_devs + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize = (10,10))
        images = denormalizer(images, *data_statistics)
        fig.savefig('images.png')
        break
        
show_batch(train_dl)

def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(elem, device) for elem in entity]
    return entity.to(device, non_blocking = True)

class DeviceDataLoader():
    "wrapper around dataloaders to transfer batches to specified devices"
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
    
    
device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

model = project1_model()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if torch.cuda.is_available():
    model.cuda()

def accuracy(logits, labels):
    pred, predClassId = torch.max(logits, dim = 1) 
    return torch.tensor(torch.sum(predClassId == labels).item()/ len(logits))

def evaluate(model, dl, loss_func):
    model.eval()
    batch_losses, batch_accs = [], []                   
    for images, labels in train_dl:
        with torch.no_grad():
              logits = model(images)
        batch_losses.append(loss_func(logits, labels))
        batch_accs.append(accuracy(logits, labels))
    epoch_avg_loss = torch.stack(batch_losses).mean().item()
    epoch_avg_acc = torch.stack(batch_accs).mean()
    return epoch_avg_loss, epoch_avg_acc
    

def train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim):
    optimizer = optim(model.parameters(), max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs * len(train_dl))
    
    results = []
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_losses = []
        lrs = []
        for images, labels in train_dl:
            logits = model(images)
            loss = loss_func(logits, labels)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        epoch_train_loss = torch.stack(train_losses).mean().item()
                       
        model.eval()
        batch_losses, batch_accs = [], []               
        for images, labels in val_dl:
            with torch.no_grad():
                logits = model(images)
            batch_losses.append(loss_func(logits, labels))
            batch_accs.append(accuracy(logits, labels))
        epoch_avg_loss, epoch_avg_acc = evaluate(model, val_dl, loss_func)
        results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc, "avg_train_loss" : epoch_train_loss, "lrs" : lrs})
        print(f"Average loss: {epoch_avg_loss}, Average accuracy {epoch_avg_acc}, Training loss: {epoch_train_loss}")
    return results

epochs = 10
max_lr = 1e-2
loss_func = nn.functional.cross_entropy
optim = torch.optim.Adam 
results = train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim)

plt_x = ["Epoch", "Epoch", "Epoch"]
plt_y = ["Accuracy", "Loss", "Learning rates"]
def plot(results, pairs):
    plt_count = 0
    fig, axes = plt.subplots(len(pairs), figsize = (10,10))
    for i, pair in enumerate(pairs):
        for title, graphs in pair.items():
            axes[i].se_title = title
            axes[i].legend = graphs
            axes[i]
            for graph in graphs:
                axes[i].plot([result[graph] for result in results], '-x')
                axes[i].set_xlabel(plt_x[plt_count])
                axes[i].set_ylabel(plt_y[plt_count])
                axes[i].grid()
            fig.savefig(str(title)+'.png')
        plt_count +=1 
    
    
plot(results, [{"accuracy_vs_epochs": ["avg_valid_acc"]}, {"Losses_vs_epochs" : ["avg_valid_loss", "avg_train_loss"]}, {"learning_rates_vs_batches": ["lrs"]}])

_,test_acc=evaluate(model,test_dl,loss_func)
params = count_parameters(model)
print(f"Test accuracy is {test_acc*100} %")
print(f"Parameters are: {params}")

## save model
model_path = './project1_model.pt'
torch.save(model.state_dict(), model_path)


