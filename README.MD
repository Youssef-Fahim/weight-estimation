# Weight Estimation from Facial Images

## Description

This project aims to estimate a person's weight based on their facial images using computer vision and deep learning techniques. The model is trained on a dataset of facial images and corresponding weights, and can be used to predict the weight of a person from a new image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

- Python 3.8 (used in the project)
To install the necessary dependencies for this project, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use this project, first run the training script to train the model:

```bash
python src/main.py
```

## Code structure

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── model_weights  <- Data from third party sources.
│   ├── prisoners      <- Intermediate data that has been transformed.
│   └── vip            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── main.py        <- Main script
│   │
│   ├── model          <- Scripts to build the CNNs
│   │   ├── __init__.py
│   │   ├── model_tf.py
│   │   └── model_torch.py
│   │
│   ├── preprocessing  <- Scripts to run preprocessing on images and create data loaders
│   │   ├── __init__.py
│   │   ├── crop_images.py
│   │   ├── dataloader_torch.py
│   │   └── train_generator.py
│   │
│   ├── training       <- Scripts to train models and then use trained models to make
│   │   │                 predictions (TensorFlow and PyTorch implementation)
│   │   ├── __init__.py
│   │   ├── mae_callback_tf.py
│   │   ├── train_tf.py
│   │   └── train_torch.py
│   │
│   ├── config          <- Folder containing config files
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   └── utils           <- Scripts to create helper functions
│       ├── __init__.py
│       └── logger.py
│
└── .gitignore           <- ignore files when pushing to GitHub
```



