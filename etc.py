#%%

img_dir = 'D:/dataset/3rd-ml-month-car-image-cropping-dataset/train_crop'
img_paths = glob(img_dir + '/*.jpg')
img = Image.open(img_paths[0])
#%%
class RandomErase(object):
    '''
    ToTensor -> Erase -> Normalize
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
#%%
#model = vision.models.shufflenet_v2_x1_0().cuda()
#model.fc = nn.Linear(1024, 196, bias = True).cuda()
#torchsummary.summary(model, input_size = (3, 299, 299))

#model = vision.models.squeezenet1_1(num_classes = 196).cuda()
#torchsummary.summary(model, input_size = (3, 224, 224))

#model = vision.models.densenet121().cuda()
#model.classifier = nn.Linear(1024, 196).cuda()
#torchsummary.summary(model, input_size = (3, 224, 224))

model = vision.models.inception_v3().cuda()
torchsummary.summary(model, input_size = (3, 299,299))


