import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
from torchvision import models
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)

import cv2
import os
import numpy as np
from PIL import Image
import pickle
import model

# transform= transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
class Load_data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
dataset = Load_data(
    csv_file="new.csv",
    root_dir="train",
    transform=transforms.ToTensor()
)
train_loader = DataLoader(dataset=dataset)
vgg=model.vgg()
print(vgg)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(vgg.parameters(), lr=0.05)
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
print(len(train_loader))

# for image,label in enumerate(train_loader):
#     print(label)
out=vgg(images)
print(out)
for epoch in range(2):
    losses = []

    for _, tensor in enumerate(train_loader):
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)

        # forward
        print(tensor)
        data=tensor[0]
        targets=tensor[1]
        opt.zero_grad()
        print(data.shape)
        scores = vgg(data)
        loss = loss_fn(scores, targets)

        losses.append(loss.item())

        # backward

        loss.backward()

        # gradient descent or adam step
        opt.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")
#vgg.state_dict('./params')
# for i in range(3):
#  	print('running epoch ',i)
#  	opt.zero_grad()
#  	outputs = vgg(x_train)
#  	loss = loss_fn(outputs,y_labels)
#  	loss.backward()
#  	opt.step()
torch.save(vgg.state_dict(),'./params/params2.pth')
#
# #print(y_labels)
# #print(x_train)
#
# with open("./face-labels.pickle", 'wb') as f:
# 	pickle.dump(label_ids, f)
# vgg.classifier[6] = nn.Linear(final_in_features,len(label_ids))
# loss_fn = nn.CrossEntropyLoss()
# opt = optim.SGD(vgg.parameters(), lr=0.05)
# print(x_train)
# print(type(y_labels))
# x_train=np.array(x_train)
# y_labels=np.array(y_labels)
# print(x_train)
# print(type(y_labels))
# x_train=torch.FloatTensor(x_train)
# y_labels=torch.from_numpy(y_labels)
# print(type(x_train))
# print(type(x_train[1]))
# print(type(y_labels))
# for i in range(3):
# 	print('running epoch ',i)
# 	opt.zero_grad()
# 	outputs = vgg(x_train)
# 	loss = loss_fn(outputs,y_labels)
# 	loss.backward()
# 	opt.step()
# torch.save(vgg.state_dict(),'./params')
# # recognizer.train(x_train, np.array(y_labels))
# # recognizer.save("recognizers/face-trainner.yml")