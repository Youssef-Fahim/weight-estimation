import os
import sys
import cv2
import dlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, "../"))

from config import config
from utils.logger import get_logger

APP_NAME = 'CROPPING_PROCESS'
LOGGER = get_logger(APP_NAME)

detector = dlib.get_frontal_face_detector()

class CroppingProcess():
    def __init__(self) -> None:
        self.plot_images = True
        self.max_images_to_plot = 5
        self.bad_crop_count = 0
        self.img_index = 0
        self.plot_index = 1
        self.plot_n_cols = 3
        self.good_cropped_images = []
        self.good_cropped_img_file_names = []
        self.detected_cropped_images = []
        self.original_images_detected = []

    @staticmethod
    def crop_image(img, x1, y1, x2, y2):
        """
        Crop image to fit the bounding box
        (Credits) Image cropping method taken from:
        https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
        Arguments:
            img: image to crop
            x1: x coordinate of the top left corner
            y1: y coordinate of the top left corner
            x2: x coordinate of the bottom right corner
            y2: y coordinate of the bottom right corner
        Returns:
            img: cropped image
        """
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = CroppingProcess.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img

    @staticmethod
    def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
        """
        Pad image to fit the bounding box
        Arguments:
            img: image to pad
            x1: x coordinate of the top left corner
            y1: y coordinate of the top left corner
            x2: x coordinate of the bottom right corner
            y2: y coordinate of the bottom right corner
        Returns:
            img: padded image
            x1: x coordinate of the top left corner
            x2: x coordinate of the bottom right corner
            y1: y coordinate of the top left corner
            y2: y coordinate of the bottom right corner
        """
        img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2
    
    def plot_cropped_images(self) -> None:
        """
        Plot original images, detected cropped images and good cropped images
        Arguments:
            None
        Returns:
            None
        """
        plot_n_rows = len(self.original_images_detected) \
                        if len(self.original_images_detected) < self.max_images_to_plot \
                        else self.max_images_to_plot
        LOGGER.info(f'Plotting images with n_cols={self.plot_n_cols} and n_rows={plot_n_rows}')
        for row in range(plot_n_rows):
            plt.subplot(plot_n_rows, self.plot_n_cols, self.plot_index)
            plt.imshow(self.original_images_detected[self.img_index].astype('uint8'))
            self.plot_index += 1

            plt.subplot(plot_n_rows, self.plot_n_cols, self.plot_index)
            plt.imshow(self.detected_cropped_images[self.img_index])
            self.plot_index += 1

            plt.subplot(plot_n_rows, self.plot_n_cols, self.plot_index)
            plt.imshow(self.good_cropped_images[self.img_index])
            self.plot_index += 1

            self.img_index += 1
        plt.show()
    
    def crop_faces(self) -> list:
        """
        Crop faces from images and save them to a new directory
        Arguments:
            None
        Returns:
            good_cropped_images: list of cropped images
        """
        if not os.path.exists(config.CROPPED_IMGS_DIR):
            os.makedirs(config.CROPPED_IMGS_DIR)
        LOGGER.info(f'Cropping faces and saving to {config.CROPPED_IMGS_DIR}')

        for file_name in sorted(os.listdir(config.ORIGINAL_IMGS_DIR)):
            LOGGER.info(f'Cropping image {file_name}')
            np_img = cv2.imread(os.path.join(config.ORIGINAL_IMGS_DIR,file_name))
            detected = detector(np_img, 1)
            img_h, img_w, _ = np.shape(np_img)
            self.original_images_detected.append(np_img)

            if len(detected) != 1:
                self.bad_crop_count += 1
                continue

            d = detected[0]
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = int(x1 - config.MARGIN * w)
            yw1 = int(y1 - config.MARGIN * h)
            xw2 = int(x2 + config.MARGIN * w)
            yw2 = int(y2 + config.MARGIN * h)
            cropped_img = self.crop_image(np_img, xw1, yw1, xw2, yw2)
            self.detected_cropped_images.append(cropped_img)
            self.good_cropped_images.append(cropped_img)
            norm_file_path = f'{config.CROPPED_IMGS_DIR}/{file_name}'
            cv2.imwrite(norm_file_path, cropped_img)

            self.good_cropped_img_file_names.append(file_name)

        # save info of good cropped images
        df = pd.read_csv(config.ORIGINAL_IMGS_INFO_FILE, sep=";")
        df = df.loc[df['number'].isin(self.good_cropped_img_file_names)]
        df.to_csv(config.CROPPED_IMGS_INFO_FILE, sep=";", index=False)

        print(f'Cropped {len(self.original_images_detected)} images and saved in ' 
            f'{config.CROPPED_IMGS_DIR} - info in {config.CROPPED_IMGS_INFO_FILE}')
        print(f'Application was not able to detect face in {self.bad_crop_count} images'
              ' - info in Data/unnormalized.txt')

        if self.plot_images:
            self.plot_cropped_images()
        return self.good_cropped_images


if __name__ == '__main__':
    crop = CroppingProcess()
    crop.crop_faces()