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

    model_name : str = "cnn" #random_forest/grad_boost/ann/cnn

    model_params : str = "config/model_configs/cnn.json"

    use_hog_features : bool = False #set to false if you are using CNN or other deep learning models

    # Database configuration
    '''DATABASE_URL: str  # PostgreSQL connection string (e.g., postgresql://user:pass@localhost:5432/db)
    POSTGRES_URL: str

    # MinIO/S3 configuration
    AWS_ENDPOINT_URL: str  # MinIO server endpoint
    AWS_ACCESS_KEY_ID: str  # MinIO access key
    AWS_SECRET_ACCESS_KEY: str  # MinIO secret key
    AWS_BUCKET_NAME: str  # Target bucket name

    '''

