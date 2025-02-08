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

    # Database configuration
    DATABASE_URL: str  # PostgreSQL connection string (e.g., postgresql://user:pass@localhost:5432/db)
    POSTGRES_URL: str

    # MinIO/S3 configuration
    AWS_ENDPOINT_URL: str  # MinIO server endpoint
    AWS_ACCESS_KEY_ID: str  # MinIO access key
    AWS_SECRET_ACCESS_KEY: str  # MinIO secret key
    AWS_BUCKET_NAME: str  # Target bucket name

    # Camera settings
    CAMERA_RESOLUTION: tuple = (1024, 768)
    CAMERA_BRIGHTNESS: int = 60
    MODALITY: Literal["image", "video"] = "image"
    CAMERA_INTERVAL: Optional[int] = 5 #in seconds
    
    # GPIO settings (TO BE SET UP BY USER)
    GPIO_PINS: Dict[str, int] = {
        "IN1": 7,
        "IN2": 11,
        "IN3": 3,
        "IN4": 5
    }

    # GPS settings
    GPS_PORT: str = '/dev/ttyAMA0'
    GPS_BAUDRATE: int = 9600
    GPS_MAX_ATTEMPTS: int = 3

