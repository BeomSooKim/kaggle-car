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

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epoch', required = False, default = 100, type = int, help = 'training epochs')
parser.add_argument('-b','--batch_size', required = False, default = 32, type = int, help = 'training batch_size')
parser.add_argument('-s','--seed', required = False, default = 1989, type = int, help = 'seed number')
parser.add_argument('-l','--base_lr', required = False, default = 0.0001, type = float, help = 'base learning rate')
#parser.add_argument('-c','--cuda', required = False, default = True, type = str2bool, help = 'flag for using GPU')
parser.add_argument('-sd','--save_dir', required = False, default = './models/', type = str, help = 'directory for saving model')
parser.add_argument('-rd','--root_dir', required = False, type = str, default = 'D:/dataset/kaggle_car/', help = 'root dataset')
#parser.add_argument('-i','--img_size', required = False, default = 256, type = int, help = 'load image size')
#parser.add_argument('-cr','--crop_size', required = False, default = 224, type = int, help = 'crop size')

args = parser.parse_args()

def train(args):
	# define argument
	epoch = args.epoch
	batch_size = args.batch_size
	base_lr = args.base_lr
	save_dir = args.save_dir
	seed = args.seed
	root_dir = args.root_dir
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
				print("train step : {} / {}".format(i+1, train_steps), end = '\r')
				image = image.to(device)
				label = label.to(device)

				output = resnet(image)
				loss = loss_fn(output, label)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print()
			train_losses.append(loss.item())
			# evaluation
			resnet.eval()
			with torch.no_grad():
				correct = 0
				total = 0
				for k, (image, label) in enumerate(valLoader):
					print("val step : {} / {}".format(k+1, val_steps), end = '\r')
					image = image.to(device)
					label = label.to(device)

					output = resnet(image)
					_, predict = torch.max(output, 1)
					total += label.size(0)
					correct += (predict == label).sum().item()
					val_loss = loss_fn(output, label)
				print()
			val_acc = correct / total
			val_losses.append(val_loss.item())
			scheduler.step(val_loss.item())
			running_time = datetime.datetime.now() - start
			print("epoch : {}({} sec) / train loss : {:.4f} / val loss : {:.4f} / val_acc : {:.4f}".format(e+1, running_time.seconds, loss.item(), val_loss.item(), val_acc))
			if val_acc > best_acc:
				torch.save(resnet.state_dict(), os.path.sep.join([cv_dir, 'model_{:03d}_{:.4f}_{:.4f}.pt'.format(e+1, val_loss.item(), val_acc)]))
				best_acc = val_acc
			
			plt.figure(figsize = (8, 4))
			plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label = 'train_loss')
			plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label = 'val_loss')
			plt.legend()
			plt.savefig(os.path.sep.join([cv_dir, 'loss_curve.png']))
			plt.close()

			resnet.train()

if __name__ == '__main__':
	set_seed(args.seed)
	train(args)