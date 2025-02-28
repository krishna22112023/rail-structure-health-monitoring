# Rail Surface Fault detection 

This is the official software implementation of paper titled "Structural health monitoring of railway tracks using IoT-based multi-robot system" [paper](https://link.springer.com/article/10.1007/s00521-020-05366-9)

## Overview 

The repo consists of three main components : 

1. Preprocessing pipeline : Uses RGB rail surface images [dataset] of five fault types namely : Cracks, Flakings, Shellings, Spallings, Squats to create processed images:\
a. Processed Skeleton: A thinned edge map generated through multi-scale Sobel edge detection and morphological skeletonization.\
b. Processed Overlay: An image where the fault skeleton is overlaid (green) on the cropped original image for clear visualization.
![alt text](<assets/Screenshot 2025-02-27 at 6.14.00 PM.png>)


2. Feature extraction : Uses [Histogram of oriented gradients (HoG)](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) to perform feature extraction on processed skeleton images. 

3. ML/DL classification 
a. Simple ML models (eg. Random forests, ANN) : Trained on the HoG features.\
b. DL models (eg. CNN) : Trained directly on the processed overlayed images.
c. (TO DO) PyTorch RPi optimized models : [PyTorch](https://pytorch.org/tutorials/intermediate/realtime_rpi.html) has out of the box support to run realtime object detection on Raspberry Pi 4 (4GB/2GB variants). 

## Setup

Create virtual environment and install dependencies
```bash
conda env create -f dependencies/env.yaml
```
For Linux/Windows, please install the CUDA version of PyTorch [here](https://pytorch.org/get-started/locally/)

## Dataset

The Railway track surface fault detection dataset was downloaded from [here](https://data.mendeley.com/datasets/8hxtgyyxrw/2)

## Configuration

## Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| **CATEGORIES** | List of defect categories: `["Cracks", "Flakings", "Shellings", "Spallings", "Squats"]`. Modify this list based on the defects you want to classify. |
| **crop_width** | Defines the width of the cropped image. Default: `200`. Ensure it captures enough relevant features. |
| **left_offset** | Proportion of the image to crop from the left. Default: `0.15`. Adjust to focus on key regions. |
| **right_offset** | Proportion of the image to crop from the right. Default: `0.6`. Modify based on image structure. |
| **diameter_neighbor_pixel** | Diameter of the pixel neighborhood for bilateral filtering. Default: `3`. Affects smoothing vs. edge preservation. |
| **sigma_color** | Filter strength based on color similarity. Default: `75`. Higher values preserve more color details. |
| **sigma_space** | Filter strength based on pixel proximity. Default: `75`. Higher values smooth over larger areas. |
| **clip_limit** | Contrast clipping for CLAHE. Default: `2`. Helps enhance local contrast. |
| **tile_grid_size** | Grid size for CLAHE. Default: `(8,8)`. Increase for finer contrast adjustments. |
| **weight** | Weight for Laplacian sharpening. Default: `1.75`. Higher values enhance sharpness. |
| **scales** | List of Sobel filter scales. Default: `[3, 5, 7]`. Affects edge detection sensitivity. |
| **morph_kernel** | Kernel size for morphological operations. Default: `(2,2)`. Adjust to refine object boundaries. |
| **model_name** | Chosen model: `"random_forest"`, `"grad_boost"`, `"ann"`, or `"cnn"`. Default: `"cnn"`. Pick based on complexity vs. performance. |
| **model_params** | Path to model config file. Default: `"config/model_configs/cnn.json"`. Ensure the file exists and matches the selected model. |
| **orientations** | Number of gradient orientations for HoG. Default: `9`. Reduce to `6` to shrink feature size. |
| **pixels_per_cell** | Size of each cell in pixels. Default: `(8,8)`. Increase to `(16,16)` to reduce feature size. |
| **cells_per_block** | Number of cells per block. Default: `(2,2)`. Consider `(3,3)` for more spatial context. |
| **visualize** | Whether to generate HoG visualizations. Default: `False`. Set `True` for debugging feature maps. |
| **use_hog_features** | Use HoG feature extraction. Default: `False`. Set to `False` if using CNN or deep learning models. |

### Tips & Tricks

#### Best Strategy to Reduce Feature Size using HoG:
- Reduce `orientations` (e.g., from `9` → `6`).
- Increase `pixels_per_cell` (e.g., from `(8,8)` → `(16,16)`).
- Optionally tweak `cells_per_block` (e.g., from `(2,2)` → `(3,3)`).

Modify the above parameters based on your dataset and experiment for optimal results.

## Scripts

To run the preprocess script, pass the input dir (eg. dataset/raw) and output dir (eg. dataset/processed) paths. 
```bash
python scripts/preprocess.py dataset/raw dataset/processed
```
Refer to the [preprocess.ipynb](notebooks/preprocess.ipynb) notebook for step by step output of preprocessing steps

To run feature extraction, pass the input dir (optional arg, by default : dataset/processed) and output dir (eg. dataset/output_HOG)
```bash
python scripts/featureextraction.py dataset/output_HOG
```
The above script should create a folder called ```train_splits``` or ```train_splits_hog``` containing the train (70%),test(10%),validation(20%) splits

To run the main training script, pass the model name (optional arg, by default : same as model name in settings.py), model_params (optional arg, by default : same as model params in settings.py) and path to train splits (eg. dataset/train_splits)
```bash
python scripts/train.py dataset/train_splits
```

## Results

| Model | Test Accuracy (Cracks) | Test Accuracy (Flakings) | Test Accuracy (Shellings) | Test Accuracy (Spallings) | Test Accuracy (Squats) | Overall Accuracy | Average Latency (ms) |
|---|---|---|---|---|---|---|---|
| ANN | 0.500 | 0.940 | 0.231 | 0.724 | 1.000 | 0.928 | 0.024 |
| CNN | 0.000 | 0.972 | 0.154 | 0.345 | 1.000 | 0.918 | 0.119 |

The above test was run on a Mac M4 chip with 64GB RAM.

Note that the the classes are highly unequally distributed. Hence future work could look into data balancing techniques or adding more samples possibly from [roboflow](https://universe.roboflow.com/dataset-aq7x0/rail-surface-defects-flrty).

Cite this article if you decided to use our work : 
```
Iyer, S., Velmurugan, T., Gandomi, A.H. et al. Structural health monitoring of railway tracks using IoT-based multi-robot system. Neural Comput & Applic 33, 5897–5915 (2021). https://doi.org/10.1007/s00521-020-05366-9
```
