import os
import numpy as np
from sklearn import preprocessing
from config import settings
# For creating the train/val/test splits and converting to PyTorch tensors
from sklearn.model_selection import train_test_split
import torch

import pyprojroot
import sys
from pathlib import Path
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

def create_data_splits(data, labels, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Create a stratified 70-20-10 train-validation-test split from the features and labels.
    
    Parameters:
        features (ndarray): 2D array of shape (n_samples, n_features) or multi dimensional array of images with shape (n_samples, ...)
        labels (list or ndarray): List or array of labels corresponding to each sample.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple of PyTorch tensors:
          (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    os.makedirs(Path(settings.BASE,output_dir), exist_ok=True)
    
    # Convert string labels to numerical classes using LabelEncoder.
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # First, split off the training set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # The remaining data (X_temp, y_temp) is split into validation and test.
    # The fraction of validation samples in the temporary set is:
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=relative_val_ratio,
        stratify=y_temp,
        random_state=random_seed
    )
    
    # Convert data and labels to PyTorch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train,dtype=torch.long)  # you might convert labels to numerical classes if needed
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,dtype=torch.long)

    #save tensors
    torch.save(X_train_tensor, Path(settings.BASE,output_dir,'X_train_tensor.pt'))
    torch.save(y_train_tensor, Path(settings.BASE,output_dir,'y_train_tensor.pt'))
    torch.save(X_val_tensor, Path(settings.BASE,output_dir,'X_val_tensor.pt'))
    torch.save(y_val_tensor, Path(settings.BASE,output_dir,'y_val_tensor.pt'))
    torch.save(X_test_tensor, Path(settings.BASE,output_dir,'X_test_tensor.pt'))
    torch.save(y_test_tensor, Path(settings.BASE,output_dir,'y_test_tensor.pt'))
    
    return (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor)
