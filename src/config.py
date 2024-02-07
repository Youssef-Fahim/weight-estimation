import torch

RESNET50_DEFAULT_IMG_WIDTH = 224
MARGIN = 0
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TRAIN_SIZE = 0.9
VALID_SIZE = 0.2

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
LEARNING_RATE = 0.01

MODEL_WEIGHTS_DIR = 'data/model_weights'
#ORIGINAL_IMGS_DIR = 'male_pictures'
ORIGINAL_IMGS_DIR = 'data/prisoners/images'
#ORIGINAL_IMGS_INFO_FILE = 'male_weights.csv'
ORIGINAL_IMGS_INFO_FILE = 'data/prisoners/annotation.csv'
# AGE_TRAINED_WEIGHTS_FILE = 'age_only_resnet50_weights.061-3.300-4.410.h5'
AGE_TRAINED_WEIGHTS_FILE = 'data/model_weights/age_only_resnet50_weights.061-3.300-4.410.hdf5'
# CROPPED_IMGS_DIR = 'female_prisoners'
CROPPED_IMGS_DIR = 'data/prisoners/cropped_images'
# CROPPED_IMGS_INFO_FILE = 'female_prisoners_kg_data.csv'
CROPPED_IMGS_INFO_FILE = 'data/prisoners/cropped_annotation.csv'
TOP_LAYER_LOG_DIR = 'logs/top_layer'
ALL_LAYERS_LOG_DIR = 'logs/all_layers'