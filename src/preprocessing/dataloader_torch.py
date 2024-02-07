import sys
import os
import math
import random
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, "../"))

from config import config
from utils.logger import get_logger

APP_NAME = 'DATALOADER_PYTORCH'
LOGGER = get_logger(APP_NAME)

class WeightDataset(Dataset):
    """
    Weight dataset class
    """
    def __init__(self, train=True):
        self.cropped_imgs_dir = config.CROPPED_IMGS_DIR
        self.cropped_imgs_info_file = config.CROPPED_IMGS_INFO_FILE
        self.annotation = None
        self.transform = transforms.Compose([
            transforms.Resize((config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH)),
            #transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            #transforms.RandomRotation(degrees=180),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.5),
            #transforms.RandomSolarize(threshold=128, p=0.25),
        ])
        self.train = train
    
    def __len__(self):
        """
        Get length of dataset
        Returns:
            shape_0: length of dataset
        """
        self.compute_train_valid_test_annotation()
        return self.annotation.shape[0]
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        Arguments:
            idx: index of the item
        Returns:
            sample: dictionary with image and weight
        """
        self.compute_train_valid_test_annotation()
        img_name = os.path.join(self.cropped_imgs_dir, self.annotation.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        image = np.array(image)
        weight = self.annotation.iloc[idx, 1]
        sample = {'image': image, 'weight': weight}
        sample = (image, weight)

        return sample

    def compute_train_valid_test_annotation(self):
        """
        Compute train, valid and test annotation
        Arguments:
            None
        Returns:
            None
        """
        df = pd.read_csv(self.cropped_imgs_info_file, sep=';')
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle dataframe
        total_size = df.shape[0]
        train_size = int(total_size * config.TRAIN_SIZE)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        valid_size = int(df_train.shape[0] * (1-config.VALID_SIZE))
        df_train, df_valid = df_train.iloc[:valid_size], df_train.iloc[valid_size:]
        self.annotation = df_train if self.train else df_valid

    @staticmethod
    def plot_images_batch(sample_batched):
        """
        Plot images from batch
        Arguments:
            sample_batched: batch of images
        Returns:
            None
        """
        # images batch will be of shape (batch_size, 224, 224, 3)
        images_batch, labels_batch = sample_batched['image'], sample_batched['weight']
        # we reshape it to (batch_size, 3, 224, 224)
        images_batch = images_batch.permute(0, 3, 1, 2)
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2
        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')

def main():
    weight_train_dataset = WeightDataset()
    weight_valid_dataset = WeightDataset(train=False)
    train_dataloader = DataLoader(weight_train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(weight_valid_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['weight'])
        if i_batch == 3:
            plt.figure()
            WeightDataset.plot_images_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    
if __name__ == '__main__':
    main()
