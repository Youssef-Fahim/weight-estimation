import sys
import os
import math
import Augmentor
import random
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, 'utils'))

import config as config
from logger import get_logger

APP_NAME = 'DATASET_GENERATOR'
LOGGER = get_logger(APP_NAME)

class DatasetGenerator():
    def __init__(self) -> None:
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.train_generator = None
        self.valid_generator = None
        self.plot_images = False

    @staticmethod
    def augment_image(np_img):
        """
        Augment image using Augmentor library
        Arguments:
            np_img: image to augment
        Returns:
            np_img: augmented image
        """
        p = Augmentor.Pipeline()
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.random_distortion(probability=0.25, grid_width=2, grid_height=2, magnitude=8)
        p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
        p.random_contrast(probability=.5, min_factor=0.8, max_factor=1.2)
        p.random_brightness(probability=1, min_factor=0.5, max_factor=1.5)

        image = [Image.fromarray(np_img.astype('uint8'))]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        image = [np.array(i).astype('float64') for i in image]
        return image[0]
    
    @staticmethod
    def plot_imgs_from_generator(generator, number_imgs_to_show):
        """
        Plot images from generator
        Arguments:
            generator: generator to plot images from
            number_imgs_to_show: number of images to plot
        Returns:
            None
        """
        LOGGER.info('Plotting images...')
        n_rows_cols = int(math.ceil(math.sqrt(number_imgs_to_show)))
        plot_index = 1
        x_batch, _ = next(generator)
        while plot_index <= number_imgs_to_show:
            plt.subplot(n_rows_cols, n_rows_cols, plot_index)
            plt.imshow(x_batch[plot_index-1])
            plot_index += 1
        plt.show()

    def compute_train_valid_test_split(self):
        """
        Compute train, valid and test dataframes
        Arguments:
            None
        Returns:
            None
        """
        df = pd.read_csv(config.CROPPED_IMGS_INFO_FILE, sep=';')
        total_size = df.shape[0]
        train_size = int(total_size * config.TRAIN_SIZE)
        df_train = df.iloc[:train_size]
        self.df_test = df.iloc[train_size:]

        valid_size = int(df_train.shape[0] * config.VALID_SIZE)
        self.df_train, self.df_valid = df_train.iloc[:valid_size], df_train.iloc[valid_size:]


    def get_dataset_generator(self, df, directory, target_width, batch_size, 
                              class_mode='raw', color_mode='rgb', rescale=1./255,
                              samplewise_center=True, samplewise_std_normalization=True):
        """
        Get TensorFlow dataset generator
        Arguments:
            df: dataframe to get generator from
            directory: directory where images are located
            target_width: width of images
            batch_size: batch size
            class_mode: class mode
            color_mode: color mode
            rescale: rescale factor
            samplewise_center: samplewise center
            samplewise_std_normalization: samplewise standard normalization
        Returns:
            dataset_generator: dataset generator
        """
        
        image_processor = ImageDataGenerator(
                                rescale=rescale,
                                samplewise_center=samplewise_center, 
                                samplewise_std_normalization=samplewise_std_normalization,
                                preprocessing_function=DatasetGenerator.augment_image
                            )
        
        dataset_generator = image_processor.flow_from_dataframe(
                            dataframe=df,
                            directory=directory,
                            x_col=df.columns[0],
                            y_col=df.columns[1],
                            class_mode=class_mode,
                            color_mode=color_mode,
                            target_size=(target_width, target_width),
                            batch_size=batch_size)
        
        return dataset_generator

    def compute_train_valid_generators(self):
        """
        Compute train and valid generators
        Arguments:
            None
        Returns:
            train_generator: train generator
            valid_generator: valid generator
        """

        self.compute_train_valid_test_split()

        self.train_generator = self.get_dataset_generator(self.df_train, config.CROPPED_IMGS_DIR,
                                                          config.RESNET50_DEFAULT_IMG_WIDTH, 
                                                          config.TRAIN_BATCH_SIZE)
        self.valid_generator = self.get_dataset_generator(self.df_valid, config.CROPPED_IMGS_DIR,
                                                          config.RESNET50_DEFAULT_IMG_WIDTH, 
                                                          config.VALID_BATCH_SIZE)
        if self.plot_images:
            DatasetGenerator.plot_imgs_from_generator(self.train_generator, 
                                                      config.TRAIN_BATCH_SIZE)
            DatasetGenerator.plot_imgs_from_generator(self.valid_generator, 
                                                      config.TRAIN_BATCH_SIZE)

        return self.train_generator, self.valid_generator


if __name__ == '__main__':
    generator = DatasetGenerator()
    train_generator, valid_generator =  generator.compute_train_valid_generators()
                         