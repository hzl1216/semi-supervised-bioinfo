import numpy as np
from PIL import Image

import torchvision
import torch

NO_LABEL = -1


def get_cifar10(root, n_labeled,train_repeated=1,transform_train=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled / 10))
    train_dataset = CIFAR10_labeled(root, train_labeled_idxs,train_unlabeled_idxs, train_repeated=train_repeated, train=True, transform=transform_train)


    val_dataset = CIFAR10_labeled(root, labeled_idxs=val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_dataset)}")
    return train_dataset, val_dataset, test_dataset

    """
    split data for labeled train, unlabeled train, val dataset
    """


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

    return train_labeled_idxs, train_unlabeled_idxs,val_idxs


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root,  labeled_idxs=None, unlabeled_idxs=None,train_repeated=1,train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if train:
            data_list=[]
            targets_list=[]
            for _ in range(train_repeated):

                if labeled_idxs is not None:
                    data_list.extend(self.data[labeled_idxs].tolist())
                    targets_list.extend(np.array(self.targets)[labeled_idxs].tolist())
            if unlabeled_idxs is not None:
                data_list.extend(self.data[unlabeled_idxs].tolist())
                targets_list.extend([-1 for _ in range(len(unlabeled_idxs))])

            self.data=np.array(data_list)
            self.targets=np.array(targets_list)
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





