import os
import sys
import tensorflow as tf

from keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from keras.models import load_model

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, "../"))

from config import config
#K.set_image_data_format('channels_last')

class CustomNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(CustomNN, self).__init__()
        self.input_layer = ResNet50(
                            include_top=False,
                            weights='imagenet',
                            input_shape=(input_dim, input_dim, 3),
                            pooling='avg'
                        )
        self.output_layer = tf.keras.layers.Dense(units=output_dim,
                                                kernel_initializer='he_normal',
                                                use_bias=False,
                                                activation='softmax',
                                                name='pred_age')
        
    def call(self, inputs):
        x = self.input_layer(inputs)
        return self.output_layer(x)

class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return Dense(units=101,
                     kernel_initializer='he_normal',
                     use_bias=False,
                     activation='softmax',
                     name='pred_age')(inputs)

def get_age_model():

    age_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH, 3),
        pooling='avg'
    )

    #age_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    #age_model.load_weights(config.AGE_TRAINED_WEIGHTS_FILE)
    age_model.summary()
    
    # prediction = Dense(units=101,
    #                    kernel_initializer='he_normal',
    #                    use_bias=False,
    #                    activation='softmax',
    #                    name='pred_age')(age_model.output)

    prediction = CustomLayer()(age_model.output)

    age_model = Model(inputs=age_model.input, outputs=prediction)
    return age_model


def get_model(ignore_age_weights=False):

    base_model = CustomNN(config.RESNET50_DEFAULT_IMG_WIDTH, 101)

    base_model.build((None, config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH, 3))

    #base_model = get_age_model()                                
    #base_model.summary()

    if not ignore_age_weights:
        base_model.load_weights(config.AGE_TRAINED_WEIGHTS_FILE)
    last_hidden_layer = base_model.get_layer(index=-2)

    base_model = Model(
        inputs=base_model.input,
        outputs=last_hidden_layer.output)
    prediction = Dense(1, kernel_initializer='normal')(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)
    return model

get_model(ignore_age_weights=True).summary()