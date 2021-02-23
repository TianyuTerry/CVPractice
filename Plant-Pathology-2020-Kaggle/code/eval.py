import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models import WSDAN

from PIL import Image
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, get_transform


ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


def predict(image_path, model_param_path, save_path, img_save_name, resize=(224,224), gen_hm=False):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
            # transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.Resize(size=(int(resize[0]), int(resize[1]))),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    image = transform(image)
    image = image.unsqueeze(0)

    net = WSDAN(num_classes=4)
    net.load_state_dict(torch.load(model_param_path))
    net.eval()

    if 'gpu' in model_param_path:
        print("please make sure your computer has a GPU")
        device = torch.device("cuda")
        try:
            net.to(device)
        except:
            print("No GPU in the environment")
    else:
        device = torch.device("cpu")

    X = image
    X = X.to(device)

    # WS-DAN
    y_pred_raw, _, attention_maps = net(X)
    attention_maps = torch.mean(attention_maps, dim=1, keepdim=True)

    # Augmentation with crop_mask
    crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

    y_pred_crop, _, _ = net(crop_image)
    y_pred = (y_pred_raw + y_pred_crop) / 2.
    y_pred = F.softmax(y_pred)

    if gen_hm:

        attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
        attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

        # get heat attention maps
        heat_attention_maps = generate_heatmap(attention_maps)

        # raw_image, heat_attention, raw_attention
        raw_image = X.cpu() * STD + MEAN
        heat_attention_image = raw_image * 0.4 + heat_attention_maps * 0.6
        raw_attention_image = raw_image * attention_maps

        for batch_idx in range(X.size(0)):
            rimg = ToPILImage(raw_image[batch_idx])
            raimg = ToPILImage(raw_attention_image[batch_idx])
            haimg = ToPILImage(heat_attention_image[batch_idx])
            rimg.save(os.path.join(save_path, '{}_raw.jpg'.format(img_save_name)))
            raimg.save(os.path.join(save_path, '{}_raw_atten.jpg'.format(img_save_name)))
            haimg.save(os.path.join(save_path, '{}_heat_atten.jpg'.format(img_save_name)))

    df = pd.read_csv("../data/train.csv")
    for i in range(len(df)):
        # if df.loc[i, 'image_id'] in image_path:
        head, tail = os.path.split(image_path)
        if df.loc[i, 'image_id'] == tail[:-4]:
            label = torch.tensor(df.loc[i, ['healthy', 'multiple_diseases', 'rust', 'scab']])
            break
    return y_pred, label

if __name__ == "__main__":
    image_path = "/Users/wangtianduo/Desktop/Term7/50.039/big proj/data/images/Train_15.jpg"
    model_param_path = "/Users/wangtianduo/Desktop/Term7/50.039/big proj/output/lala[cpu].pkl"
    print(predict(image_path, model_param_path, ".", "lala", resize=(224,224), gen_hm=True))
