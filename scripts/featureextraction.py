import sys
import os
import logging
import click
import numpy as np
from pathlib import Path

import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

logger = logging.getLogger(__name__)

from config import settings
from src import feature_extractor, data_extractor
from utils import logger as logging

@click.command()
@click.option("--input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def feature_extractor_main(input_dir, output_dir):
    """Function to extract HOG features
    """
    if input_dir is None:
        input_dir = "dataset/processed"

    try:
        # Process the directory and compute features for each image.
        full_input_dir = Path(settings.BASE,input_dir)
        feature_matrix, labels = feature_extractor.process_directory(full_input_dir, settings.CATEGORIES, settings.orientations, settings.pixels_per_cell, settings.cells_per_block, settings.visualize)
        
        # Save the feature matrix and labels. The feature matrix is a 2D array where each row is a HOG descriptor.
        os.makedirs(Path(settings.BASE,output_dir), exist_ok=True)
        np.save(Path(settings.BASE,output_dir,"HOG_features.npy"), feature_matrix)
        np.save(Path(settings.BASE,output_dir,"labels.npy"), np.array(labels))
        
        logger.info("Feature extraction completed.")

        (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor) = data_extractor.create_data_splits(feature_matrix, labels,output_dir="dataset/train_splits_hog", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42)
        logger.info("Data splits created.")
        logger.info("Training set shape: %s", X_train_tensor.shape)
        logger.info("Validation set shape: %s", X_val_tensor.shape)
        logger.info("Test set shape: %s", X_test_tensor.shape)
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")

@click.command()
@click.option("--input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def overlay_extractor_main(input_dir, output_dir):
    """function to extract images directly
    """
    if input_dir is None:
        input_dir = "dataset/processed"

    try:
        # Process the directory and load overlayed images for each category.
        full_input_dir = Path(settings.BASE, input_dir)
        images_array, labels = feature_extractor.process_overlayed_images(
            full_input_dir, settings.CATEGORIES
        )
        
        # Save the image data and labels.
        os.makedirs(Path(settings.BASE, output_dir), exist_ok=True)
        np.save(Path(settings.BASE, output_dir, "overlayed_images.npy"), images_array)
        np.save(Path(settings.BASE, output_dir, "labels.npy"), np.array(labels))
        
        logger.info("Overlayed image extraction completed.")
        (X_train_tensor, y_train_tensor), (X_val_tensor, y_val_tensor), (X_test_tensor, y_test_tensor) = data_extractor.create_data_splits(images_array, labels,output_dir="dataset/train_splits", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42)
        logger.info("Data splits created.")
        logger.info("Training set shape: %s", X_train_tensor.shape)
        logger.info("Validation set shape: %s", X_val_tensor.shape)
        logger.info("Test set shape: %s", X_test_tensor.shape)
    except Exception as e:
        logger.error(f"Error during overlay extraction: {e}") 

    


if __name__ == "__main__":
    logging.setup_logging()
    if settings.use_hog_features: 
        feature_extractor_main()
    else:
        overlay_extractor_main()
        
 
    

