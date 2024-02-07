import sys
import os
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import Callback

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, "../"))

from config import config
from utils.logger import get_logger
from preprocessing.train_generator_tf import DatasetGenerator

APP_NAME = 'MAE_CALLBACK'
LOGGER = get_logger(APP_NAME)


class MAECallback(Callback):

    @staticmethod
    def get_mae(actual, predicted):
        n_samples = predicted.shape[0]
        diff_sum = 0.00
        for i in range(n_samples):
            p = predicted[i][0]
            a = actual[i]
            d = abs(p - a)
            diff_sum += d
        return diff_sum / n_samples

    def on_train_begin(self, logs={}):
        self._data = []


    def on_epoch_end(self, batch, logs={}):
        with open(config.CROPPED_IMGS_INFO_FILE, 'r') as f:
            test_images_info = f.read().splitlines()[-config.VALIDATION_SIZE:]
        test_x = []
        test_y = []
        for info in test_images_info:
            weight = float(info.split(';')[1])
            test_y.append(weight)
            file_name = info.split(';')[0]
            file_path = f'{config.CROPPED_IMGS_DIR}/{file_name}'
            print(file_path)
            img = cv2.imread(file_path)
            img = np.resize(img, (config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH,3))
            test_x.append(img.__div__(255.00))
        
        generator = DatasetGenerator()
        train_generator, valid_generator =  generator.compute_train_valid_generators()
        
        train_batch = next(train_generator)
        val_batch = next(valid_generator)
    
        X_train = train_batch[0]
        y_train = train_batch[1]
        y_train_pred = np.asarray(self.model.predict(X_train))

        X_val= val_batch[0]
        y_val = val_batch[1]
        y_val_pred = np.asarray(self.model.predict(X_val))
        
        a=[]
        for i in range(len(y_train_pred)):
            a.append(y_train_pred[i][0])

        b=[]
        for i in range(len(y_val_pred)):
            b.append(y_val_pred[i][0])    
        
        val_mae = MAECallback.get_mae(y_val, y_val_pred)
        logs['val_mae'] = val_mae
        self._data.append({
            'val_mae': val_mae,
        })
        return


    def get_data(self):
        return self._data