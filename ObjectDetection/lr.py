import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import log_loss
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from shutil import copyfile


from torchvision import transforms, models
import torchvision
from PIL import Image
import xml.etree.cElementTree as ET
import os, sys, collections, random
import json, operator
from sklearn.metrics import precision_score
from torch.utils import model_zoo
from sklearn.metrics import average_precision_score, precision_score, accuracy_score


def to_binary(label_list):
    labels = torch.zeros(20)
    for key in label_list:
        labels[key] = 1
    return labels


class ImgDataset(Dataset):
    def __init__(self, root, transform, dataset='train', samplesize=-1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.dataset = dataset
        self.load()
        if samplesize != -1:
            self.images = self.images[:samplesize]
            self.annotations = self.annotations[:samplesize]

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return {'aeroplane':0 , 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
              'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
              'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
              'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}
        
    def load(self):
        image_dir = os.path.join(self.root, 'JPEGImages')
        annotation_dir = os.path.join(self.root, 'Annotations')

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it')

        set_annotation_dir = os.path.join(self.root, 'ImageSets/Main')
        set_annotation_path = os.path.join(set_annotation_dir, self.dataset.rstrip('\n') + '.txt')

        if not os.path.exists(set_annotation_path):
            raise ValueError(
                'Wrong dataset entered! Please use dataset="train" '
                'or dataset="trainval" or dataset="val" or a valid'
                'dataset from the VOC ImageSets/Main folder.')

        with open(os.path.join(set_annotation_path), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]        
        
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_xml(ET.parse(self.annotations[index]).getroot())

        ind_list = target['annotation']['object']
        label = []
        if (type(ind_list) is dict):
            label.append(self.list_image_sets()[ind_list['name']])
        else:
            for e in ind_list:
                label.append(self.list_image_sets()[e['name']])
            
        img = self.transform(img)
        return img, to_binary(label), self.images[index]

    def __len__(self):
        return len(self.images)

    def parse_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

def train(model, device, optimizer, train_loader, criterion):
    model.train()
    count = 0

    loss = []
    for idx, data in enumerate(train_loader):
        img, target, name = data
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        batch_loss = criterion(output, target)
        batch_loss.backward()
        optimizer.step()
        loss.append(batch_loss.item())
        count += img.shape[0]
        print('{} samples done'.format(count))

    avg_train_loss = sum(loss)/len(loss)
    print("training loss of batch: ", sum(loss)/len(loss))

    return avg_train_loss

def val(model, device, val_loader, criterion, thres):
    model.eval()

    loss = []
    outputs = []
    predictions = []
    targets = []

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, target, name = data
            targets.append(target)
            img, target = img.to(device), target.to(device)
            prediction = model(img)
            batch_loss = criterion(prediction, target)
            loss.append(batch_loss)
            output = sigmoid(prediction)
            filtered_output = (output > thres)
            outputs.append(output)
            predictions.append(filtered_output)
            
    avg_val_loss = sum(loss)/len(loss)
    output_f = torch.cat(outputs, 0).cpu().detach().numpy().reshape(-1, 20)
    targets_f = torch.cat(targets,0).cpu().detach().numpy().reshape(-1, 20)
    predictions_f = torch.cat(predictions,0).cpu().detach().numpy().reshape(-1, 20)
    avg_precision = average_precision_score(targets_f, output_f)

    precisions = [average_precision_score(targets_f[:,i], output_f[:,i]) for i in range(20)]
    return avg_val_loss, avg_precision, precisions


def run(device, train_loader, val_loader, num_epochs, lr1, lr2):

    train_losses1 = []
    train_losses2 = []
    val_losses1 = []
    val_losses2 = []
    class_wise_precisions1 = []
    class_wise_precisions2 = []

    model1 = models.resnet18(pretrained=True)
    model1.fc = nn.Linear(512, 20)
    model1.to(device)
    model2 = models.resnet18(pretrained=True)
    model2.fc = nn.Linear(512, 20)
    model2.to(device)

    criterion = BCEWithLogitsLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr = lr1, momentum=0.9)
    optimizer2 = optim.SGD(model2.parameters(), lr = lr2, momentum=0.9)

    for epoch in range(num_epochs):
        train_loss1 = train(model1, device, optimizer1, train_loader, criterion)
        train_losses1.append(train_loss1)
        avg_val_loss1, avg_precision1, cprecisions1 = val(model1, device, val_loader, criterion, 0.5)
        val_losses1.append(avg_val_loss1.item())
        cavg_p1 = sum(cprecisions1)/len(cprecisions1)
        class_wise_precisions1.append(cavg_p1)

        train_loss2 = train(model2, device, optimizer2, train_loader, criterion)
        train_losses2.append(train_loss2)
        avg_val_loss2, avg_precision2, cprecisions2 = val(model2, device, val_loader, criterion, 0.5)
        val_losses2.append(avg_val_loss2.item())
        cavg_p2 = sum(cprecisions2)/len(cprecisions2)
        class_wise_precisions2.append(cavg_p2)

    avg_train_loss1 = sum(train_losses1)/len(train_losses1)
    avg_val_loss1 = sum(val_losses1)/len(val_losses1)
    avg_val_precision1 = sum(class_wise_precisions1)/len(class_wise_precisions1)

    avg_train_loss2 = sum(train_losses2)/len(train_losses2)
    avg_val_loss2 = sum(val_losses2)/len(val_losses2)
    avg_val_precision2 = sum(class_wise_precisions2)/len(class_wise_precisions2)

    plot_accuracy(num_epochs, class_wise_precisions1, class_wise_precisions2, lr1, lr2)
    plot_losses(num_epochs, train_losses1, val_losses1, train_losses2, val_losses2, lr1, lr2)


def plot_accuracy(numepochs, accuracy1, accuracy2, lr1, lr2):
    
    x=range(1, numepochs+1)
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision Score')
    plt.plot(x, accuracy1, label="val_accuracy_lr{}".format(lr1))
    plt.plot(x, accuracy2, label="val_accuracy_lr{}".format(lr2))

    plt.legend(loc='upper right')
    plt.title('lr_accuracy')
    plt.savefig('lr_accuracy.png')
    plt.show()


def plot_losses(numepochs, train_losses1, val_losses1, train_losses2, val_losses2, lr1, lr2):
    
    x=range(1, numepochs+1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, train_losses1, label="train_loss_lr{}".format(lr1))
    plt.plot(x, val_losses1, label="validation_loss_lr{}".format(lr1))
    plt.plot(x, train_losses2, label="train_loss_lr{}".format(lr2))
    plt.plot(x, val_losses2, label="validation_loss_lr{}".format(lr2))

    plt.legend(loc='upper right')
    plt.title('lr_loss')
    plt.savefig('lr_loss.png')
    plt.show()
    


def main(num_epochs, batch_size_train, batch_size_val, lr1, lr2, device):
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                             ])
    transform_with_crops = transforms.Compose([transforms.RandomResizedCrop(224),
                               transforms.RandomRotation(20),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


    train_set = ImgDataset('./VOCdevkit/VOC2012', transform, dataset='train')
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=False)
    val_set = ImgDataset('./VOCdevkit/VOC2012', transform_with_crops, dataset='val')
    val_loader = DataLoader(val_set, batch_size=batch_size_val, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    run(device, train_loader, val_loader, num_epochs, lr1, lr2)

    
if __name__ == "__main__":
    num_epochs = 10
    batch_size_train = 32
    batch_size_val = 32
    lr1 = 0.01
    lr2 = 0.001
    device = torch.device("cuda")
    logsigmoid = nn.LogSigmoid()
    sigmoid = nn.Sigmoid()

    main(num_epochs, batch_size_train, batch_size_val, lr1, lr2, device)