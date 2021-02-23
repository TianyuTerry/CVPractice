import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import init_model
from loss import get_loss_fn
from dataset import ppDataset
from parser_util import get_parser
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, get_transform

import matplotlib.pyplot as plt

cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def train(model, tr_dataloader, criterion, optimizer, epoch, options=None,feature_center=None):
    since = time.time()
    device = get_device()
    model.train()

    model.to(device)

    running_loss = 0.
    running_corrects = 0.
    for idx, (inputs, labels, _) in enumerate(tr_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #labels = labels.squeeze(-1)
        optimizer.zero_grad()
        model.zero_grad()

        y_pred_raw, feature_matrix, attention_map = model(inputs)
            
        # if len(labels.shape
        feature_center_batch = F.normalize(feature_center[labels], dim=-1)
        feature_center[labels] += 0.05 * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(inputs, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

        # crop images forward
        y_pred_crop, _, _ = model(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(inputs, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

        # drop images forward
        y_pred_drop, _, _ = model(drop_images)
        outputs = (y_pred_raw + y_pred_crop + y_pred_drop)/3.
        # loss
        loss = cross_entropy_loss(y_pred_raw, labels) / 3. + \
                    cross_entropy_loss(y_pred_crop, labels) / 3. + \
                    cross_entropy_loss(y_pred_drop, labels) / 3. + \
                    center_loss(feature_matrix, feature_center_batch)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item() * inputs.size(0)
        batch_corrects = torch.sum(labels == outputs.argmax(dim=1))
                
        running_loss += batch_loss
        running_corrects += batch_corrects

        if idx % 9 == 1:
            print('[Train]Epoch: {}, idx: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, idx, batch_loss/len(inputs), batch_corrects.float()/len(inputs)))

    epoch_loss = running_loss / len(tr_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(tr_dataloader.dataset)

    print('[Train]Epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

    return epoch_acc, epoch_loss


def validation(model, val_dataloader, criterion, epoch, options=None):
    device = get_device()
    model.to(device)
    model.eval()

    running_loss = 0.
    running_corrects = 0.
    with torch.no_grad():
        for inputs, labels, _ in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels = labels.squeeze(-1)
                    
            model.zero_grad()

            y_pred_raw, _, attention_map = model(inputs)

            crop_images = batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = model(crop_images)

            y_pred = (y_pred_raw + y_pred_crop) / 2.
            outputs = y_pred
            if len(labels.shape) == 0:
                print("Error")
                loss = torch.tensor(0)
            else:
                loss = cross_entropy_loss(y_pred, labels)


                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(labels == outputs.argmax(dim=1))

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

    print('[Validation]Epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss


def write_csv(model, te_dataset, submission_df_path, options=None):
    print("Generating prediction...")
    device = get_device()
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)
    submission_df = pd.read_csv(submission_df_path)

    test_pred = None
    model.eval()
    with torch.no_grad():
        for inputs in te_dataloader:
            inputs = inputs.to(device)

            if options is not None and options.model==4:

                y_pred_raw, _, attention_map = model(inputs)

                crop_images = batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
                y_pred_crop, _, _ = model(crop_images)

                y_pred = (y_pred_raw + y_pred_crop) / 2.
                outputs = y_pred
            else:

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            if test_pred is None:
                test_pred = outputs.data.cpu()
            else:
                test_pred = torch.cat((test_pred, outputs.data.cpu()), dim=0)

    test_pred = torch.softmax(test_pred, dim=1, dtype=float)
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_pred

    submission_df.to_csv(options.output_root+options.output_name+'.csv', index=False)



if __name__ == "__main__":
    options = get_parser().parse_args()
    # model_idx = options.model
    batch_size = options.batch_size
    num_epoch = options.epochs
    data_root = options.data_root
    input_size = options.input_size

    train_csv_path = data_root + "train.csv"
    test_csv_path = data_root + "test.csv"
    images_dir = data_root + "images/"
    submission_df_path = data_root + "sample_submission.csv"
    
    num_classes = 4
    num_cv_folds = 5

    device = get_device()
    model, _ = init_model(num_classes, use_pretrained=options.pre_train)

    feature_center = torch.zeros(4, 32 * model.num_features).to(device)
    criterion = get_loss_fn()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    tr_df_all = pd.read_csv(train_csv_path)
    tr_df, val_df = train_test_split(tr_df_all, test_size = 0.2)
    val_df = val_df.reset_index(drop=True)
    tr_df = tr_df.reset_index(drop=True)
    te_df = pd.read_csv(test_csv_path)
    
    tr_dataset = ppDataset(tr_df, images_dir, return_labels = True, transforms = get_transform((input_size,input_size), "train"))
    val_dataset = ppDataset(val_df, images_dir, return_labels = True, transforms = get_transform((input_size,input_size), "val"))
    te_dataset = ppDataset(te_df, images_dir, return_labels = False, transforms = get_transform((input_size,input_size), "test"))
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    train_loss_ls = []
    valid_loss_ls = []
    train_accu_ls = []
    valid_accu_ls = []

    for i in range(1, num_epoch+1):
        train_acc, train_loss = train(model, tr_dataloader, criterion, optimizer, i, options, feature_center)
        val_acc, val_loss = validation(model, val_dataloader, criterion, i, options)

        train_loss_ls.append(train_loss)
        train_accu_ls.append(train_acc)
        valid_loss_ls.append(val_loss)
        valid_accu_ls.append(val_acc)
        scheduler.step()

    for i in range(5):
        train(model, val_dataloader, criterion, optimizer, i, options, feature_center)
    out_root = options.output_root

    plt.figure(12)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_ls)
    plt.plot(valid_loss_ls)

    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accu_ls)
    plt.plot(valid_accu_ls)

    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    plt.savefig("train_figure.png")
    # plt.show()

    
    if options.model_addr == 'cpu':
        devic = torch.device("cpu")
        model.to(devic)
    torch.save(model.state_dict(), out_root+options.output_name+'['+options.model_addr+']'+'.pkl')



