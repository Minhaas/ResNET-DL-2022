
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
#from torchsummary import summary
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from cutout import Cutout
use_cuda = torch.cuda.is_available()

data_statistics = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transforms_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    Cutout(n_holes=1, length=8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010), inplace=True)
])

test_transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*data_statistics, inplace=True)
])

dataset = torchvision.datasets.CIFAR10(root="data/", download=True, transform=train_transforms_cifar)
test_dataset = torchvision.datasets.CIFAR10(root="data/", download=True, train=False, transform=test_transform_cifar)

val_ratio = 0.2
train_dataset, val_dataset = random_split(dataset, [int((1-val_ratio) * len(dataset)), int(val_ratio * len(dataset))])
batch_size = 64
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=True)


# def denormalizer(images, means, std_devs):
#     # (image*st_dev + mean
#     means = torch.tensor(means).reshape(1, 3, 1, 1)
#     std_devs = torch.tensor(std_devs).reshape(1, 3, 1, 1)
#     return images * std_devs + means
# # showing the images


# def show_batch(dl):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(10, 10))
#         images = denormalizer(images, *data_statistics)
#         # ax.imshow(make_grid(images,10).permute(1,2,0))
#         print(labels)
#         break


# show_batch(train_dl)


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")


def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(elem, device) for elem in entity]
    return entity.to(device, non_blocking=True)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18():
    layers = [2, 2, 2, 2]
    model = ResNet(BasicBlock, layers)
    return model


model = resnet18()

if torch.cuda.is_available():
    model.cuda()

def accuracy(logits, labels):
    pred, predClassId = torch.max(logits, dim=1)
    return torch.tensor(torch.sum(predClassId == labels).item() / len(logits))


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
        print(f"Finished {epoch}/{epochs} epochs.")
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
        results.append({'avg_valid_loss': epoch_avg_loss, "avg_valid_acc": epoch_avg_acc,
                        "avg_train_loss": epoch_train_loss, "lrs": lrs})
    return results

epochs = 10
max_lr = 1e-2
loss_func = nn.functional.cross_entropy
optim = torch.optim.Adam

results = train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim)
results

for result in results:
    print(result['avg_valid_acc'])

def plot(results, pairs):
    fig, axes = plt.subplots(len(pairs), figsize=(10, 10))
    for i, pair in enumerate(pairs):
        for title, graphs in pair.items():
            axes[i].se_title = title
            axes[i].legend = graphs
            axes[i]
            #for graph in graphs:
                # axes[i].plot([result[graph] for result in results], '-x')
                # fig.savefig(pairs+'.jpg')


#plot(results, [{"accuracy vs epochs": ["avg_valid_acc"]}, {"Losses vs epochs" : ["avg_valid_loss", "avg_train_loss"]}, {"learning rates vs batches": ["lrs"]}])
test_loss, test_acc = evaluate(model, test_dl, loss_func)
print(f"Test accuracy is: {test_acc}")
#summary(model, (3,32,32))
