import os
import sys
import time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from PIL import Image
from tempfile import TemporaryDirectory

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, 'utils'))

import config as config
from logger import get_logger
from dataloader_torch import WeightDataset
from model_torch import CustomResnet50

APP_NAME = 'MODEL_TRAINING'
LOGGER = get_logger(APP_NAME)

class ModelTraining():

    def __init__(self, verbose=False) -> None:
        self.model = CustomResnet50().to(config.DEVICE)
        self.model_weights_dir = config.MODEL_WEIGHTS_DIR
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = nn.L1Loss()
        self.running_loss = 0.0
        self.running_mae = 0.0
        self.best_mae = 30.0
        self.best_model_params_path = None
        self.best_model_wts = self.model.state_dict()
        self.train_loader, self.valid_loader = self.compute_dataloaders()
        self.dataloaders = {'train': self.train_loader, 'valid': self.valid_loader}
        self.dataset_sizes = {'train': len(self.train_loader.dataset), 
                              'valid': len(self.valid_loader.dataset)}
        self.verbose = verbose

    def compute_dataloaders(self):
        """
        Compute dataloaders for training and validation
        Returns:
            train_loader: training dataloader
            valid_loader: validation dataloader
        """
        train_dataset = WeightDataset(train=True)
        valid_dataset = WeightDataset(train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
                                                   shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, 
                                                   shuffle=False, num_workers=4)

        return train_loader, valid_loader


    def train_model(self, num_epochs=200):
        """
        Train the model
        Arguments:
            num_epochs: number of epochs to train the model
        Returns:
            model: trained model
        """
        start = time.time()

        for epoch in range(num_epochs):
            LOGGER.info(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train() # Set model to training mode
                else:
                    self.model.eval() # Set model to evaluate mode

                self.running_loss = 0.0
                self.running_mae = 0.0

                # Iterate over batches of data
                for idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if self.verbose:
                        LOGGER.info(f'Batch {idx+1}/{len(self.dataloaders[phase])} for {phase} phase in epoch num {epoch+1}')
                    inputs = inputs.to(config.DEVICE)
                    labels = labels.to(config.DEVICE)

                    # Zero the parameter gradients to avoid accumulation
                    # between the batches
                    self.optimizer.zero_grad()

                    # Forward pass to get the predictions
                    # Track history if only in train
                    with torch.set_grad_enabled(phase=='train'):
                        inputs = inputs.float()
                        outputs = self.model(inputs)
                        outputs = outputs.squeeze() # remove the extra dimension
                        loss = self.criterion(outputs, labels)

                        # Backward pass + optimize only if in training phase
                        if phase == 'train':
                            loss.backward() # compute the gradients
                            self.optimizer.step() # update the weights

                    # Compute the running loss and mae
                    self.running_loss += loss.item() * inputs.size(0)
                    self.running_mae += torch.sum(torch.abs(outputs - labels.data))

                if phase == 'train':
                    self.scheduler.step() # update the learning rate

                # Loss and mae for the epoch
                epoch_loss = self.running_loss / self.dataset_sizes[phase]
                epoch_mae = self.running_mae / self.dataset_sizes[phase]

                LOGGER.info(f'{phase} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}')

                # Deep copy the model
                if phase == 'valid' and epoch_mae < self.best_mae:
                    self.best_mae = epoch_mae
                    self.best_model_wts = self.model.state_dict()
                    file_name = f'best_model_params_MAE_{self.best_mae:.2f}_epoch_{epoch}.pth'
                    self.best_model_params_path = os.path.join(self.model_weights_dir, 
                                                               file_name)
                    torch.save(self.best_model_wts, self.best_model_params_path)
                    LOGGER.info(f'Saving best model params to {self.best_model_params_path}')

        time_elapsed = time.time() - start
        LOGGER.info(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        LOGGER.info(f'Best val MAE: {self.best_mae:.4f}')

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)

        return self.model


if __name__ == "__main__":
    training = ModelTraining(verbose=True)
    training.train_model()