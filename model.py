import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch import nn
import torch.optim as optim
import cv2
import json
import sys
from tqdm import tqdm


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def validation(model, loss_fn, val_loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    total_batch = 0

    with torch.no_grad(): 
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            preds = model(inputs)
            pred = preds.argmax(1)

            total += preds.size(0)
            correct += (pred == labels).sum().item()            
            
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
            total_batch += 1

    val_loss = total_loss / total_batch
    val_acc = correct / total
    print(f'Val Loss: {val_loss}, Val Accuracy: {100 * val_acc:0.4}%')
    return val_loss, val_acc


def train(model, loss_fn, optimizer, train_loader):
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    total_batch = 0

    for _, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        preds = model(inputs)
        pred = preds.argmax(1)

        total += preds.size(0)
        correct += (pred == labels).sum().item()
        
        loss = loss_fn(preds, labels)
        loss.backward() # Caculate gradients
        optimizer.zero_grad()
        optimizer.step() # Update weights
        total_loss += loss.item()
        total_batch += 1

    train_loss = total_loss / total_batch
    train_acc = correct / total
    print(f'Train Loss: {train_loss}, Train Accuracy: {100 * train_acc:0.4}%')
    return train_loss, train_acc

class Model(nn.Module):
    def __init__(self, test = False):
        super().__init__()
        if test:
            self.model = models.resnet50().to(device)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        else:
            self.model = models.resnet50(weights = ResNet50_Weights.DEFAULT).to(device)
            for param in self.model.parameters():
                param.requires_grad = True
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = MEAN,
            std = STD
        )]
    )

    train_dataset = torchvision.datasets.ImageFolder('../train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder('../val', transform=transform)

    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().to(device)


    for epoch in tqdm(range(10)):
        print(f"Epoch: {epoch+1}")
        train_loss, train_acc = train(model, loss_fn, optimizer, train_loader)
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'resnet_classifier': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f"resnet50_checkpoint/epoch{epoch+1}.pyt")
            del checkpoint

        val_loss_end, val_acc_end = validation(model, loss_fn, val_loader)

if __name__ == '__main__':
    main()