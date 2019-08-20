import torch
import torch.nn as nn 
import torchvision as vision
import torchsummary
from PIL import Image
import numpy as np
import os, random

# cut-out augmentation module
class RandomErase(object):
    '''
    ToTensor -> Erase -> Normalize
    a_ratio : range of rectangle area ratio.  
    raw image area * a_ratio(randomly selected) is the area of patch  
    h_ratio : ratio between height and width of patch.  
    0.5 means that width is 2 times of height.  
    '''
    def __init__(self, a_ratio, h_ratio):
        if isinstance(a_ratio, (float, int)):
            self.a_ratio = (a_ratio, a_ratio)
        else:
            self.a_ratio = a_ratio
        if isinstance(h_ratio, (int, float)):
            self.h_ratio = (h_ratio, h_ratio)
        else:
            self.h_ratio = h_ratio

    def __call__(self, img):
        a_r = np.random.uniform(self.a_ratio[0], self.a_ratio[1], size = 1)[0]
        h_r = np.random.uniform(self.h_ratio[0], self.h_ratio[1], size = 1)[0] 
        H, W = img.shape[1:]
        patch_area = W * H * a_r
        new_W = int(np.sqrt(patch_area / h_r))
        new_H = int(h_r * new_W)
        h_point = np.random.randint(H - new_H, size = 1)[0]
        w_point = np.random.randint(W - new_W, size = 1)[0]
        print(h_point, new_H, w_point, new_W)
        img[:,h_point:h_point+new_H,w_point:w_point + new_W] = 0

        return img

# define custom data loader and image transformer
class Dataloader(torch.utils.data.Dataset):
	def __init__(self, image_path, labels, root_dir, transform = None):
		self.image_path = image_path
		self.labels = labels
		self.root_dir = root_dir # image foloder
		self.transform = transform
		
	def __len__(self):
		return len(self.image_path)

	def __getitem__(self, index):
		image = Image.open(os.path.sep.join([self.root_dir, self.image_path[index]])).convert("RGB")
		label = self.labels[index]
		
		image = self.transform(image)
		
		return image, label

# fix random seed function
def set_seed(seed):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# mix-up augmentation
def mixup(x, y, alpha, device):
    batch_size = x.shape[0]
    
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = np.ones((batch_size))
    #lambda_ = torch.tensor(lambda_)
    suffled_idx = np.arange(batch_size)
    shuffled_idx = np.random.shuffle(batch_size)

    mixed_x = lambda_*x + (1-lambda_) * x[shuffled_idx,:,:,:]
    mixed_y = lambda_*y + (1-lambda_) * y[shuffled_idx]

    return mixed_x, mixed_y

# transformer for training data
train_aug = vision.transforms.Compose([
    vision.transforms.Resize(256),
    vision.transforms.RandomCrop(224),
    vision.transforms.RandomAffine(
        degrees = (-20,20),
        translate = (0.1, 0.1),
        scale = (0.8, 1.2),
        shear = 0.1
    ),
    vision.transforms.RandomHorizontalFlip(0.5),
    vision.transforms.ToTensor(),
    RandomErase((0.05, 0.1), (0.5, 2.0)),
    vision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# transformer for validation data
val_aug = vision.transforms.Compose([
    vision.transforms.Resize(256),
    vision.transforms.CenterCrop(224),
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# transformer for test data    
test_aug = vision.transforms.Compose([
    vision.transforms.Resize(256),
    vision.transforms.CenterCrop(224),
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# string argument to bool
def str2bool(arg):
    return str(arg).lower() in ['t','1','true']