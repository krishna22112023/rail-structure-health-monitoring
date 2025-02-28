from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import pyprojroot
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Dict,Literal, Optional


class PathInfo:
    """
    Base information class that defines core paths and environment settings.
    These paths are crucial for the application to locate resources and configurations.
    """

    HOME: Path = Path.home()
    BASE: Path = pyprojroot.find_root(pyprojroot.has_dir("config"))
    WORKSPACE: Path = BASE.parent.parent
    ENV = "dev"


# Load environment variables from .env file
load_dotenv(Path(PathInfo.BASE, ".env"))


class GeneralSettings(BaseSettings, PathInfo):

    model_config = SettingsConfigDict(case_sensitive=True)

    #categories should have same names as folder names in dataset/processed
    CATEGORIES : list = ["Cracks", "Flakings", "Shellings", "Spallings", "Squats"]

    #preprocess params 
    #cropping params
    crop_width : int = 200
    left_offset : float = 0.15
    right_offset : float = 0.6
    #bilateral filter params
    diameter_neighbor_pixel : int = 3
    sigma_color : float = 75
    sigma_space : float = 75
    #CLAHE params
    clip_limit : float = 2
    tile_grid_size : tuple = (8, 8)
    #laplacian sharpening params
    weight : float = 1.75
    #sobel edge detection params
    scales : list = [3, 5, 7]
    #morphological operations params
    morph_kernel : tuple = (2,2)

    #model params
    model_name : str = "cnn" #random_forest/grad_boost/ann/cnn
    model_params : str = "config/model_configs/cnn.json"

    #hog feature extraction params
    orientations : int = 9
    pixels_per_cell : tuple = (8, 8)
    cells_per_block : tuple = (2, 2)
    visualize : bool = False
    use_hog_features : bool = False #set to false if you are using CNN or other deep learning models

