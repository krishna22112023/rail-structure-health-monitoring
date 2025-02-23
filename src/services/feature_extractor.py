# feature_extractor.py
import logging 
logger = logging.getLogger(__name__)

import os
from tqdm import tqdm
import glob
import numpy as np
from skimage.io import imread
from skimage.feature import hog

def compute_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False):
    """
    Compute HOG features for a given grayscale image (typically a skeletonized image).
    
    Parameters:
        image (ndarray): Grayscale image.
        orientations (int): Number of orientation bins.
        pixels_per_cell (tuple): Size (in pixels) of a cell.
        cells_per_block (tuple): Number of cells in each block.
        visualize (bool): If True, returns a hog image for visualization.
        
    Returns:
        features (1D ndarray): The HOG feature vector.
        hog_image (ndarray, optional): Visualization of the HOG (if visualize is True).
    """
    if visualize:
        features, hog_image = hog(image,
                                  orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  visualize=True,
                                  channel_axis=None)
        return features, hog_image
    else:
        features = hog(image,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       visualize=False,
                       channel_axis=None)
        return features

def process_directory(processed_dir,categories):
    """
    Process the skeletonized images from the given processed folder.
    
    Assumes the following directory structure:
      processed/
          Cracks/processed_skeleton/
          Flakings/processed_skeleton/
          Shellings/processed_skeleton/
          Spallings/processed_skeleton/
          Squats/processed_skeleton/
    
    For each image, HOG features are computed. The function returns:
      - feature_matrix: A 2D numpy array of shape (n_samples, n_features)
      - labels: A list of labels (fault types) corresponding to each sample.
    
    Parameters:
        processed_dir (str): Path to the top-level "processed" folder.
    
    Returns:
        feature_matrix (ndarray): Matrix of HOG features.
        labels (list): List of string labels for each sample.
    """
    # Define the fault categories (subdirectory names)
    categories = categories
    feature_list = []
    label_list = []
    
    for category in categories:
        skeleton_dir = os.path.join(processed_dir, category, "processed_skeleton")
        image_files = glob.glob(os.path.join(skeleton_dir, "*"))
        logger.info("Processing %d images from category: %s", len(image_files), category)
        
        for img_file in tqdm(image_files,desc=f"Processing"):
            # Read the image in grayscale mode.
            # The images are assumed to be skeletonized (binary or near-binary images).
            image = imread(img_file, as_gray=True)
            # Compute the HOG features
            features = compute_hog_features(image)
            feature_list.append(features)
            label_list.append(category)
    
    # Convert the list of features to a 2D numpy array.
    # Each row corresponds to one image's HOG feature vector.
    feature_matrix = np.array(feature_list)
    logger.info("Feature matrix shape: %s", feature_matrix.shape)
    return feature_matrix, label_list

def process_overlayed_images(base_dir, categories):
    """
    Loads the overlayed images (with green pixels) from the given directory structure.
    
    Assumes a directory layout like:
        base_dir/
            Cracks/overlayed_images/
            Flakings/overlayed_images/
            Shellings/overlayed_images/
            Spallings/overlayed_images/
            Squats/overlayed_images/
    
    Each subfolder corresponds to one category (fault type).
    
    Returns:
        images_array (ndarray): Array of shape (num_images, H, W, C) or (num_images, H, W) if grayscale.
        labels (list): List of string labels for each image.
    """
    images_list = []
    labels_list = []
    
    for category in categories:
        # For example, we look for a subfolder named 'overlayed_images'
        overlayed_dir = os.path.join(base_dir, category, "processed_overlay")
        
        # Grab all image files
        image_files = glob.glob(os.path.join(overlayed_dir, "*"))
        logger.info("Found %d overlayed images for category: %s", len(image_files), category)
        
        for img_file in tqdm(image_files, desc=f"Loading {category}"):
            # Read the image in color. skimage will return shape (H, W) for grayscale or (H, W, 3) for RGB
            image = imread(img_file)
             # Ensure images are in (C, H, W) format
            if image.ndim == 3:  # RGB Image
                image = np.transpose(image, (2, 0, 1))  # Change (H, W, C) -> (C, H, W)
            else:  # Grayscale image, add channel dimension
                image = np.expand_dims(image, axis=0)  # Change (H, W) -> (1, H, W)
            
            images_list.append(image)
            labels_list.append(category)
    
    # Convert list to a numpy array
    images_array = np.array(images_list)
    logger.info("Overlayed images array shape: %s", images_array.shape)
    return images_array, labels_list
