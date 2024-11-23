from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import stack,manual_seed,float32,Tensor,from_numpy,argmax,int64,where,argmax
import os
from PIL import Image
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
import random
import numpy as np

class PolypDataset(Dataset):
    def __init__(self, image_dir, transform = None, test_transform = None, random_transforms = None):
        self.image_dir = image_dir
        self.train_dir = os.path.join(image_dir,"train/train")
        self.gt_dir = os.path.join(image_dir,"train_gt/train_gt")
        self.test_dir = os.path.join(image_dir,"test/test")
        self.transform = transform
        self.test_transform = test_transform
        self.random_transforms = random_transforms
        self.img_name = [f for f in os.listdir(self.train_dir) if os.path.isfile(os.path.join(self.train_dir, f))]
        self.test_img_name = [f for f in os.listdir(self.test_dir) if os.path.isfile(os.path.join(self.test_dir, f))]

    def __len__(self):
        return len(self.img_name)
    
    def split_train_test(self,test_size,random_seed=None):
        if random_seed == None:
            random_seed = random.randint(0,10000000)
        self.img_name, self.valid_img_name = train_test_split(self.img_name, test_size=test_size, random_state=random_seed)
        return CustomDataset(self.img_name,self.train_dir,self.gt_dir,self.transform, self.random_transforms, self.test_transform),CustomDataset(self.valid_img_name,self.train_dir,self.gt_dir,self.transform, self.random_transforms, self.test_transform)
        
class CustomDataset(Dataset):
    def __init__(self, img_name, train_dir, gt_dir, transform = None, random_transforms = None, test_transform = None):
        self.img_name = img_name
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.random_transforms = random_transforms
        self.test_transform = test_transform
        self.gt_to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.train_dir, self.img_name[idx])
        gt_name = os.path.join(self.gt_dir, self.img_name[idx])
        image = Image.open(img_name)
        gt = Image.open(gt_name)
        if self.random_transforms:
            image = self.random_transforms(image)
        if self.transform:
            rands = random.randint(0,10000000)
            random.seed(rands)
            manual_seed(rands)
            image = self.transform(image)
            random.seed(rands)
            manual_seed(rands)
            gt = self.transform(gt)
        if self.test_transform:
            image = self.test_transform(image)
        mask = self.prepare_gt(gt)
        return image,mask
    
    def prepare_gt(self,gt):
        gt = self.gt_to_tensor(gt)
        gt = where(gt > 0.65, 1.0, 0.0)
        gt[2,:,:] = 0.0001
        mask = argmax(gt,dim=0).type(int64)
        return mask
        