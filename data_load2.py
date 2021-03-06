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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# vgg=models.vgg16()
# for param in vgg.parameters():
#     param.requires_grad = False
# final_in_features = vgg.classifier[6].in_features


current_id = 1
label_ids = {}
y_labels = []
x_train=[]
data=[]
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label not in label_ids :
                label_ids[label] = current_id
                current_id += 1
                number=0
            id_ = label_ids[label]
            img=cv2.imread(path)
            name=str(label)+'.'+str(current_id-1)+'.'+str(number)+'.jpg'
            number+=1
            print(name,id_)
            cv2.imwrite(name,img)

# df_final = pd.DataFrame(data=data, columns=["filename", "label"])
# #df_final.tail()
# df_final.to_csv('imagedata.csv')
