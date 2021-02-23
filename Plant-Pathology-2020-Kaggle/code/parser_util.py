# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',
                        default=16)

    parser.add_argument('--data_root',
                        type=str,
                        help='path to data folder',
                        default='../data/')

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=5)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('--output_root',
                        type=str,
                        help='output file holder for submission csv',
                        default="../output/")

    parser.add_argument('--input_size',
                        type=int,
                        help='input image size',
                        default=224)

    parser.add_argument('--output_name',
                        type=str,
                        help='output name for both submission and model',
                        default="lala")

    parser.add_argument('--model_addr',
                        type=str,
                        help='gpu or cpu',
                        default='cpu')

    parser.add_argument('--pre_train',
                        type=bool,
                        help='',
                        default=True)

    return parser
