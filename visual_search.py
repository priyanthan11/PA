"""
Buld the core AI engine for an intellignet product catalog system.
"""


import torch
import torch.nn as nn
import torchinfo
import math
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import os
import unittest
import random
from IPython.display import display
import copy


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


"""
1 - Build a Fasthon Item Classifier
 - 1 - Build a model that can accuratly categorize clothing items. This is the fundermental step for organizing the product catalog and enablig features like filtered search.
 
1 - The Fasion Dataset
* Cloathing-dataset-small (https://github.com/alexeygrigorev/clothing-dataset-small). It has "dress", "hat","longsleeve","pants","shoes", and "t-shirts".

2 - Preparing the Data Pipeline
Befor we train the model craft the pipeline. Transfom the images (resize, augmented, normalized) and how they are loaded in batches. We serperate transformation pipleines for training and validation.


"""


# Compose the trains formations for training: resize, augment, them preprocess.
train_transform = transforms.Compose([
    # Resize images to consistent squre size.
    transforms.Resize((64, 64)),
    # Apply Random horizontal flipping.
    transforms.RandomHorizontalFlip(),
    # Apply Random Rotation
    transforms.RandomRotation(10),
    # Convert PIL (Pillow) into PyTorch tensors
    transforms.ToTensor(),
    # Normalize tesnros values to range [-1,1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Validations: Only Resize and preprocess (no augmentation)
val_transform = transforms.Compose([
    # Resize images to the same consistent squire size
    transforms.Resize((64, 64)),
    # Convert PIL to PyTorch Tensors
    transforms.ToTensor(),
    # Normaloize tensor values to range [-1,1]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

])

# Data path
dataset_path = "./data/clothing_dataset_small"


# Load The datasets
def load_dataset(dataset_path, train_transform=None, val_transform=None, batch_size=32, num_worker=4, shuffle_train=True):
    """Load traning and validation dataset from a directory

    Args:
        dataset_path (str): _Path to the root dataset directory.
        train_transform: TorchVision transforms to apply to training images. Defaults to None.
        val_transform: TorchVision transforms to apply to validation images. Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_worker (int, optional): Subprocesses for data loading. Defaults to 4.
        shuffle_train (bool, optional): Whether to suffle the trainig set. Defaults to True.
    """

    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "validation")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_dataset = datasets.ImageFolder(
        root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    return train_dataset, val_dataset


# Load the training and validation datsets
train_dataset, validation_dataset = load_dataset(
    dataset_path, train_transform, val_transform)

# Get the list of class names automatically iferred from the folder structuer
classes = train_dataset.classes

# Get the total number of classes
num_classes = len(classes)

# # Print the discrocered class names
# print(f"Classes: {classes}")
# print(f"Number of Classess: {num_classes}")

# Display a grid of sample images from the traing dataset with their labels


def show_some_images(dataset, num_images=16, cols=4, figsize=(12, 12), random_sample=True, seed=None):
    # Set the seed
    if seed is not None:
        random.seed(seed)

    total = len(dataset)
    num_images = min(num_images, total)

    indicies = random.sample(
        range(total), num_images) if random_sample else list(range(num_images))

    rows = math.ceil(num_images/cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for ax, idx in zip(axes, indicies):
        image, label = dataset[idx]
        print(
            f"type: {type(image)}, dtype: {image.dtype}, min: {image.min():.3f}, max: {image.max():.3f}, shape: {image.shape}")

        # Convert tensor to numpy (C,H,W) to (H,W,C)
        if hasattr(image, "numpy"):
            img_np = image.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(image)

        # Undo ImageNet style normalization if values fall outsite [0,1]
        if img_np.min() < 0 or img_np.max() > 1:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std+mean)

        img_np = np.clip(img_np, 0, 1)

        class_name = dataset.classes[label] if hasattr(
            dataset, "classes") else str(label)
        ax.imshow(img_np)
        ax.set_title(class_name, fontsize=10, pad=4)
        ax.axis("off")
    # hide any unused subplot cells
    for ax in axes[num_images:]:
        ax.axis("off")
    plt.suptitle(
        f"Sample Trainng Images ({num_images} if {total})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# show_some_images(train_dataset)
