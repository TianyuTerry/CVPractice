import torch.nn as nn
from torchvision import transforms, models

from models import WSDAN

def init_model(num_classes, use_pretrained):

    model = WSDAN(num_classes=num_classes, pretrained=use_pretrained)
    return model, 224
