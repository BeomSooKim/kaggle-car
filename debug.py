#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np 
from utils import *
import os, sys, datetime
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%%

# define argument
epoch = 100
batch_size = 32
base_lr = 0.0001
save_dir = './models/'
seed = 1989
root_dir = 'D:/dataset/kaggle_car/'
#img_size = args.img_size
#crop_size = args.crop_size

print('epochs : {}'.format(epoch))
print('batch_size : {}'.format(batch_size))
print('base_lr : {}'.format(base_lr))
print('save model to : {}'.format(save_dir))
print('load data from : {}'.format(root_dir))
#print('load image size : {}'.format(img_size))
#print('crop image size : {}'.format(crop_size))

save_cur_dir = save_dir + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load meta file
meta = pd.read_csv('D:/dataset/2019-3rd-ml-month-with-kakr/train.csv')
meta = meta[['img_file','class']]
meta['class'].replace(196, 0, inplace = True) # class index starts from 0
meta['img_file'] = meta['img_file'].str.replace('.jpg','.png')
n_class = meta['class'].nunique()

# split train, validation dataset
fold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = seed)
for i, (train_idx, val_idx) in enumerate(fold.split(meta['img_file'], meta['class'])):
    # fold{x} path
    cv_dir = os.path.sep.join([save_cur_dir, 'fold{}'.format(i+1)])
    os.makedirs(cv_dir)
    # split train, validation set
    trainX = list(meta['img_file'][train_idx])
    valX = list(meta['img_file'][val_idx])
    trainY = list(meta['class'][train_idx])
    valY = list(meta['class'][val_idx])
    print(trainX[:3], valX[:3], trainY[:3], valY[:3])
    # define dataloader
    trainData = myDataset(trainX, trainY, os.path.sep.join([root_dir, 'train']), transform = train_aug)
    valData = myDataset(valX, valY, os.path.sep.join([root_dir, 'train']), transform = val_aug)
    
    trainLoader = torch.utils.data.DataLoader(
    dataset = trainData,
    batch_size = batch_size,
    shuffle = True
    )

    valLoader = torch.utils.data.DataLoader(
        dataset = valData,
        batch_size = batch_size,
        shuffle = False
    )

    # define resnet model
    resnet = torchvision.models.resnet50(pretrained = True)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
    resnet.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr = base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5, verbose = True, threshold = 1e-8)
    # train model
    train_steps = len(trainLoader)
    val_steps = len(valLoader)
    best_acc = 0.
    train_losses, val_losses = [],[]
    for e in range(epoch):
        start = datetime.datetime.now()
        for i, (image, label) in enumerate(trainLoader):
            print(i)