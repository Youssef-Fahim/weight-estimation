import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, 'utils'))

import config as config
from logger import get_logger
from mae_callback import MAECallback
from mae_callback import get_mae
from model import get_model
from train_generator import train_generator, plot_imgs_from_generator, validation_generator

APP_NAME = 'MODEL_TRAINING'
LOGGER = get_logger(APP_NAME)

batches_per_epoch=train_generator.n //train_generator.batch_size

print(batches_per_epoch)

def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss

def train_top_layer(model):

    LOGGER.info('Training top layer...')

    for l in model.layers[:-30]:
        l.trainable = False

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    mae_callback = MAECallback()

    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        verbose=1,
        patience=5)

    model_checkpoint_callback = ModelCheckpoint(
        'saved_models/top_layer_trained_weights.L1_smooth{epoch:02d}-{val_mae:.2f}.h5',
        monitor='val_mae',
        mode='min',
        verbose=1,
        save_best_only=True
    )

    tensorboard_callback = TensorBoard(
        log_dir=config.TOP_LAYER_LOG_DIR,
        batch_size=train_generator.batch_size
    )

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=batches_per_epoch,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[
            mae_callback,
            early_stopping_callback,
            model_checkpoint_callback,
            tensorboard_callback
        ]
    )

    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train_all_layers(model):

    LOGGER.info('Training all layers...')

    for l in model.layers:
        l.trainable = True

    mae_callback = MAECallback()

    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        verbose=1,
        patience=10)

    model_checkpoint_callback = ModelCheckpoint(
        'saved_models/all_layers_trained_weights_checkMALES.{epoch:02d}-{val_mae:.2f}.h5',
        monitor='val_mae',
        mode='min',
        verbose=1,
        save_best_only=True)

    tensorboard_callback = TensorBoard(
        log_dir=config.ALL_LAYERS_LOG_DIR,
        batch_size=train_generator.batch_size
    )

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=batches_per_epoch,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[
            mae_callback,
            early_stopping_callback,
            model_checkpoint_callback,
            tensorboard_callback
        ]
    )

    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def test_model(model):

    def Sort(sub_li): 
  
        # reverse = None (Sorts in Ascending order) 
        # key is set to sort using second element of  
        # sublist lambda has been used 
        sub_li.sort(key = lambda x: x[1]) 
        return sub_li     
    
    batch = next(validation_generator)
    
    #print batch

    test_X = batch[0]
    test_y = batch[1]
    

    a=[]
    for i in range (len(test_X)):
        a.append([test_X[i],test_y[i]])
        
            
    a = Sort(a)
    
#
    
    for i in range (len(test_X)):
        test_X[i] = a[i][0]
        
    weights_file = 'saved_models/all_layers_trained_weights_NEWMALES.14-14.92.h5'
    model.load_weights(weights_file)
    predict = model.predict(test_X)

    
    
    b=[]
    c=[]
    for i in range(len(predict)):
        b.append(predict[i][0])
        c.append(a[i][1])
    
    
    print(c)
    print(b)
    
    
    mae = get_mae(test_y, predict)
    print('\nMAE:', mae)
    


if __name__ == '__main__':
    model = get_model()
    train_top_layer(model)
    train_all_layers(model)
    test_model(model)
#train_top_layer(get_model(ignore_age_weights=False))
#train_all_layers(get_model(ignore_age_weights=False))
#test_model(get_model(ignore_age_weights=False))
    
