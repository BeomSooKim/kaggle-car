import torch
import torch.nn as nn
import torchvision as vision
import torchvision.transforms as transforms

import numpy as np 
from utils import *
import os, sys, datetime
import argparse
from glob import glob
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-s','--seed', required = False, default = 1989, type = int, help = 'seed number')
parser.add_argument('-f','--fold_num', required = False, default = 1, type = int, help = 'fold number')
parser.add_argument('-ld','--load_dir', required = False, default = '20190822-072518', type = str, help = 'directory for saving model')
parser.add_argument('-i','--img_size', required = False, default = 256, type = int, help = 'load image size')
parser.add_argument('-b','--batch_size', required = False, default = 128, type = int, help = 'test batch size')
parser.add_argument('-cr','--crop_size', required = False, default = 224, type = int, help = 'crop size')
parser.add_argument('-t','--tta', required = False, default = 1, type = int, help = 'test time augmentation step')
args = parser.parse_args()

def test(args):
    fold = args.fold_num
    load_dir = args.load_dir
    img_size = args.img_size
    batch_size = args.batch_size
    crop_size = args.crop_size
    tta = args.tta
    print('load model from : {}'.format(load_dir))
    print('fold number : {}'.format(fold))
    print('batch size : {}'.format(batch_size))
    print('load image size : {}'.format(img_size))
    print('crop image size : {}'.format(crop_size))
    print('test time augmentation : {}'.format(tta))

    n_class = 196
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test image transforms
    test_trn = transforms.Compose([
        #transforms.Resize(size = 256),
        #transforms.CenterCrop(size = 224),
        transforms.RandomResizedCrop(size = 224, scale = (0.95, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.106], std = [0.229, 0.224, 0.225])
    ])

    # load test file path
    test_list = glob('D:/dataset/kaggle_car/test/*.png')
    test_list = [x.split('\\')[-1] for x in test_list]
    test_data = myDataset(test_list, None, 'D:/dataset/kaggle_car/test', test_trn)
    print(len(test_data))
    test_loader = torch.utils.data.DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        shuffle = False
    )

    # load best model
    pred_ensemble = []
    model_list = glob(os.path.sep.join(['models', load_dir, 'fold{}'.format(fold), '*.pt']))
    model_selected = sorted(model_list, key = lambda x: float(x.split('_')[-1].rstrip('.pt')))[-1]
    print('best model for fold{} : {}'.format(fold, model_selected))

    # load model
    resnet = vision.models.resnet50(pretrained = False)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
    resnet.to(device)
    resnet.load_state_dict(torch.load(model_selected))

    print('number of test image : {}'.format(len(test_list)))
    resnet.eval()
    for n in range(tta):
        predict_list = []
        print('{}/{} tta'.format(n+1, tta))
        with torch.no_grad():
            for idx, img in enumerate(test_loader):
                img = img.to(device)
                predict = resnet(img).cpu().numpy()
                predict_list.append(np.exp(predict))

                print('{}/{}'.format(idx+1, np.ceil(len(test_list) // 128)), end = '\r')
            predict_list = np.concatenate(predict_list, axis = 0)
            pred_ensemble.append(predict_list)
            print()

    pred_ensemble = np.dstack(pred_ensemble)
    print('ensemble shape : {}'.format(pred_ensemble.shape))
    pred_ensemble = pred_ensemble.mean(axis = 2)
    output = pred_ensemble.argmax(axis = 1)
    result = pd.DataFrame({'img_file':test_list, 'class':output})
    result['class'].replace(0, 196, inplace = True)
    result['img_file'] = result['img_file'].str.replace(".png",".jpg")
    result.to_csv(os.path.sep.join(['models',load_dir,'fold{}'.format(fold), 'submission.csv']), index = False)


if __name__ == '__main__':
    set_seed(args.seed)
    test(args)