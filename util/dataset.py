import numpy as np
from PIL import Image
import pandas as pd
import os
import torchvision
import torch
import torch.utils.data as data
import _pickle as cPickle
import random
NO_LABEL = -1


def get_cifar10(root, n_labeled,transform_train=None, transform_val=None,
                download=True):
    def train_val_split(labels, n_labeled_per_class):
        labels = np.array(labels)
        train_labeled_idxs = []
        train_unlabeled_idxs = []
        val_idxs = []

        for i in range(10):
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
            val_idxs.extend(idxs[-500:])
        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)
        np.random.shuffle(val_idxs)

        return train_labeled_idxs, train_unlabeled_idxs, val_idxs


    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled / 10))
    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True,transform=transform_train)

    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} #Val: {len(val_dataset)}")
    return train_labeled_dataset,train_unlabeled_dataset, val_dataset, test_dataset

    """
    split data for labeled train, unlabeled train, val dataset
    """





cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        return x


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for _ in range(len(self.targets))])




class TCGA_DATASET(data.Dataset):
    def __init__(self,root,train=True,transform=None, target_transform=None,):
        self.root = root
        self.transform=transform
        self.target_transform=target_transform
        self.data =[]
        self.targets=[]
        if train:
            df = pd.read_csv(root+'/train.csv')
        else:
            df = pd.read_csv(root+'/test.csv')
#        df = pd.read_csv(root+'/dlbc.csv')
        
            
        self.data = np.array(df.iloc[:, 1:])
        self.targets=np.array(df.iloc[:, 0]-1)

    def __getitem__(self, index):

        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, int(target)

    def __len__(self):
        return  len(self.targets)

    def save_dir(self):
        f = open('label_dict', 'wb')
        cPickle.dump(self.label_dict, f)

class TCGA_labeled(TCGA_DATASET):
    def __init__(self, tcga_dataset, indexs=None, transform=None, target_transform=None, ):
        self.data = tcga_dataset.data
        self.targets = tcga_dataset.targets
        self.transform=transform
        self.target_transform=target_transform
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]


class TCGA_unlabeled(TCGA_labeled):
    def __init__(self, tcga_dataset,indexs=None, transform=None, target_transform=None, ):
        super(TCGA_unlabeled, self).__init__( tcga_dataset, indexs,transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for _ in range(len(self.targets))])


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, erasing_value=0):
        self.probability = probability
        self.erasing_value = erasing_value
        self.sl = sl
        self.sh = sh
    def __call__(self, data):
        if random.uniform(0, 1) > self.probability:
            return data
        length = len(data)
        for _ in range(100):
            erasing_length = int(random.uniform(self.sl, self.sh) * length)
            if erasing_length < length:
                x = random.randint(0, length - erasing_length)

                data[x:x+erasing_length] = self.erasing_value
                return data
        return data

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __init__(self,probability=0.5):
        self.probability = probability
    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        length = len(x)
        x += np.random.randn(length) * 0.15
        return x

def get_tcga(root,n_labeled,transform_train=None,transform_val=None):
    def train_val_split_random(labels, n_labeled, randomtype='all'):

        train_labeled_idxs = []
        train_unlabeled_idxs = []
        if randomtype == 'type':
            for i in range(33):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_labeled_idxs.extend(idxs[:n_labeled // 33])
                train_unlabeled_idxs.extend(idxs[n_labeled // 33:])
        else:
            for i in range(33):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_labeled_idxs.extend(idxs[:3])

            length = len(labels)
            idxs = np.array(list(set([i for i in range(length)])-set(train_labeled_idxs))) 
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled-(33*3)])
            train_unlabeled_idxs.extend(idxs[n_labeled-(33*3):])
        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)

        return train_labeled_idxs, train_unlabeled_idxs


    base_dataset = TCGA_DATASET(root)
    train_labeled_idxs, train_unlabeled_idxs = train_val_split_random(base_dataset.targets,n_labeled)
    train_labeled_dataset = TCGA_labeled(base_dataset, train_labeled_idxs,  transform=transform_train)
    train_unlabeled_dataset = TCGA_unlabeled(base_dataset, train_unlabeled_idxs,  transform=transform_train)

    val_dataset = TCGA_DATASET( root , train=False,transform=transform_val)


    print(f"#Labeled: {len(train_labeled_dataset)}  #Val: {len(val_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

if __name__ == '__main__':
    # dataset = TCGA_DATASET('./data')
    print(get_tcga('./data',1000))



