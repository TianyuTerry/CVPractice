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


def run(device, train_loader, val_loader, num_epochs, lr):
    label_names = {0: 'aeroplane', 1: 'bicycle',
              2: 'bird', 3: 'boat', 4: 'bottle',
              5: 'bus', 6: 'car', 7: 'cat',
              8: 'chair', 9: 'cow', 10: 'diningtable',
              11: 'dog', 12: 'horse', 13: 'motorbike',
              14: 'person', 15: 'pottedplant', 16: 'sheep',
              17: 'sofa', 18: 'train', 19: 'tvmonitor'}

    best_val_loss = 2
    best_val_precision = -1
    train_losses = []
    val_losses = []
    # val_precision = []
    class_wise_precisions = []

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 20)
    model.to(device)

    criterion = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

    best_weights = {}
    for epoch in range(num_epochs):
        train_loss = train(model, device, optimizer, train_loader, criterion)
        train_losses.append(train_loss)
        avg_val_loss, avg_precision, cprecisions = val(model, device, val_loader, criterion, 0.5)
        # val_accuracies.append(avg_precision)
        val_losses.append(avg_val_loss.item())
        cavg_p = sum(cprecisions)/len(cprecisions)
        class_wise_precisions.append(cavg_p)


        
        # select best model by validation precision
        if cavg_p > best_val_precision:
            best_weights = model.state_dict()
            torch.save(best_weights, 'best_weight.pkl')
            best_val_precision = cavg_p
            best_val_loss = avg_val_loss
        print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Validation Loss: {:.3f}..".format(avg_val_loss),
                  "Validation precision: {:.3f}..".format(cavg_p))
        for i in range(len(cprecisions)):
            print("the classwise precision for ", label_names[i]," class is: ", cprecisions[i])
    
    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_val_loss = sum(val_losses)/len(val_losses)
    avg_val_precision = sum(class_wise_precisions)/len(class_wise_precisions)

    plot(range(1, num_epochs+1), train_losses, "training loss", xlabel = "epoch", ylabel="training loss")
    plot(range(1, num_epochs+1), val_losses, "validation loss", xlabel = "epoch", ylabel="validation loss")
    plot(range(1, num_epochs+1), class_wise_precisions, "average precisions", xlabel = "epoch", ylabel="average precisions")

# get precision vs threshold (pretty useless)
    model.load_state_dict(torch.load('best_weight.pkl'))
    model.eval()
    t_s = np.linspace(0.1, 0.9, 9, endpoint=True)
    avg_precisions = []
    outputs = []
    labels = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, label, name = data
            img, label = img.to(device), label.to(device)
            output = model(img)
            output_sig = sigmoid(output)
            outputs.append(output_sig)
            labels.append(label)

    output_f = torch.cat(outputs, 0).cpu().detach().numpy().reshape(-1, 20)
    targets_f = torch.cat(labels,0).cpu().detach().numpy().reshape(-1, 20)
    for t in t_s:
        pred = (output_f > t)
        avg_precision = precision_score(targets_f, pred, average="micro")
        avg_precisions.append(avg_precision)
    
    plot(t_s, avg_precisions, "average precision scores with different threshold values", xlabel="threshold", ylabel="micro averaged precision")
            
    return avg_train_loss, avg_val_loss, avg_val_precision, class_wise_precisions


def plot(t_values, tail_acc, name, xlabel = 'threshold value', ylabel='Tail Accuracy'):
    
    plt.plot(t_values, tail_acc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(name+'.png')
    plt.show()


def class_wise_tailacc(pred, target, t):
    """
        params: pred: the sigmoid predictied score of samples (ranging from 0 to 1)
                target: the ground truth label for samples of a specific class
                t: tail threshold
    """
    count = 0
    correct = 0
    for i in range(len(pred)):
        if pred[i] > t:
            count += 1
            if target[i] == 1:
                correct += 1
    if count == 0:
        accuracy = 0
    else:
        accuracy = correct/count
    return accuracy


def ranking(path, device, transform, val_loader):
    label_names = {0: 'aeroplane', 1: 'bicycle',
              2: 'bird', 3: 'boat', 4: 'bottle',
              5: 'bus', 6: 'car', 7: 'cat',
              8: 'chair', 9: 'cow', 10: 'diningtable',
              11: 'dog', 12: 'horse', 13: 'motorbike',
              14: 'person', 15: 'pottedplant', 16: 'sheep',
              17: 'sofa', 18: 'train', 19: 'tvmonitor'}
    dic = {}
    rank_d = {}
    for i in range(20):
        dic[i] = dict()
        rank_d[i] = dict()


    with torch.no_grad():
        model = models.resnet18(pretrained = True)
        model.fc = nn.Linear(512, 20)
        model.load_state_dict(torch.load('best_weight.pkl'))
        model.eval()
        model.to(device)
        
        outputs = []
        labels = []
        filenames = []
        
        for idx, data in enumerate(val_loader):
                img, label, name = data
                img, label = img.to(device), label.to(device)
                output = model(img)
                outputs.append(output)
                labels.append(label)
                filenames+=name  


    ranks = {}
    filenames_f = np.array(filenames).reshape(-1)
    outputs_f = sigmoid(torch.cat(outputs, 0)).cpu().detach().numpy().reshape(-1, 20)
    labels_f = torch.cat(labels, 0).cpu().detach().numpy().reshape(-1, 20)


    tailaccs = []
    max_score = max(outputs_f.copy().reshape(-1))
    ts = np.linspace(0.5, max_score, 20, endpoint=False)
    for i in range(20):
        classwise_acc = []

        c = label_names[i]
        sortedArr = np.array(outputs_f[:,i].argsort()).reshape(-1)

        names = filenames_f.copy()
        names = [names[n] for n in sortedArr]
        ranks[c] = names
        print("The 50 lowest score image in class ", c," is : ", ranks[c][0:50])
        print("The 50 highest score image in class ", c," is : ", ranks[c][len(ranks[c])-50:])
        
        output_c = outputs_f[:,i].reshape(-1)
        label_c = labels_f[:,i].reshape(-1)
        
        top_output = [output_c[n] for n in sortedArr]
        top_label = [label_c[n] for n in sortedArr]
    #     get highest score 
        
        for t in ts:
            accuracy = class_wise_tailacc(top_output, top_label, t)
            classwise_acc.append(accuracy)
        plot(ts, classwise_acc, 'classwise tail accuracy of class: '+c)
        tailaccs.append(classwise_acc)

    # copy the five top score image to folder:
        if not os.path.isdir("./TOP/"+c+"-top"):
            os.mkdir("./TOP/"+c+"-top")
        if not os.path.isdir("./TOP/"+c+"-bottom"):
            os.mkdir("./TOP/"+c+"-bottom")
        for j in range(5):
            src_bottom = ranks[c][j]
            src_top = ranks[c][len(ranks[c])-j-1]
            dst_top = os.path.join("./TOP/"+c+"-top", 'top'+str(j)+'.jpg')
            dst_bottom = os.path.join("./TOP/"+c+"-bottom", 'bottom'+str(j)+'.jpg')

            copyfile(src_top, dst_top)
            copyfile(src_bottom, dst_bottom)  

    tail_accuracies = np.array(tailaccs)
    avg_tail_acc = np.mean(tail_accuracies, 0)
    plot(ts, avg_tail_acc, "average tail accuracies")




def main(num_epochs, batch_size_train, batch_size_val, learning_rate, device):
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

    run(device, train_loader, val_loader, num_epochs, learning_rate)

    print("----------------- GET RANKING FOR EACH CLASS -------------------")
    ranking('best_weight.pkl', device, transform, val_loader)

    
if __name__ == "__main__":
    num_epochs = 5
    batch_size_train = 32
    batch_size_val = 32
    lr = 0.01
    device = torch.device("cuda")
    logsigmoid = nn.LogSigmoid()
    sigmoid = nn.Sigmoid()

    main(num_epochs, batch_size_train, batch_size_val, lr, device)