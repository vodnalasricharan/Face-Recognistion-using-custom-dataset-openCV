from torchvision import models
import torch.nn as nn
def vgg():
    vgg = models.vgg16()
    for param in vgg.parameters():
        param.requires_grad = False
    final_in_features = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(final_in_features, 5)

    return vgg
