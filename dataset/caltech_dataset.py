from torchvision.datasets import VisionDataset

from PIL import Image

import cv2
import os
import torch 
import numpy as np 
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train',split_size=0.5, transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.root = root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.split_size = split_size    

        file = open(root+'train.txt',"r")
        image_paths=file.readlines()

        data = []
        labels = []
        for image_path in image_paths:
            label = image_path.split('/')[0]
            if label == 'BACKGROUND_Google':
                continue

            image = cv2.imread(self.root+'101_ObjectCategories\\'+image_path.replace("/","\\")[0:-1])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data.append(image)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)

        # one hot encode
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        print(f"Total number of classes: {len(lb.classes_)}")
        
        if self.split == 'train':
            # divide the data into train, validation, and test set
            (x_train, x_val , y_train, y_val) = train_test_split(data, labels, test_size=self.split_size, stratify=labels, random_state=42)
#             print(f"x_train examples: {x_train.shape}\nx_val examples: {x_val.shape}")

            self.images=x_train
            self.labels=y_train
            
            
        if self.split == 'validation':
            # divide the data into train, validation, and test set
            (x_train, x_val , y_train, y_val) = train_test_split(data, labels, test_size=self.split_size, stratify=labels, random_state=42)
#             print(f"x_train examples: {x_train.shape}\nx_val examples: {x_val.shape}")

            self.images=x_val
            self.labels=y_val

            
        if self.split == 'test':
            
            self.images=data
            self.labels=labels
        
        '''
       The Custom Module "Caltech" is used for Custom DataLoading for all the Train,Validation and Test Datasets. it is calibrated to work with the Caltech101 Dataset and its input parameters are as follows:
       root : pass the directory your notebook and 101_ObjectCategories folder containing the dataset are present
       split : pass the dataset split that you are using for creating the Dataloader Obj. 'train' or 'validation' or 'test'
       split_size : pass the size you find adequate for spliting the train-Validation split
       transform : pass the transformation you want to apply to the images
        '''
        
        
        
    def __getitem__(self, index):
        dataset = self.images[index][:]
        
        if self.transforms:
            dataset = self.transform(dataset)
            
        if self.labels is not None:
            return (dataset, self.labels[index])
        else:
            return dataset
        
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''


    def __len__(self):
        return (len(self.images))
    
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        