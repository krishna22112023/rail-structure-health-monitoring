# src/schemas/fault.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class FaultType(str, Enum):
    """Enumeration of possible fault types"""
    RAIL_SURFACE = "Rail Surface"
    CONTACT_BAND = "Contact Band"
    SPALLING = "Spalling"
    CORRUGATION_GRINDING = "Corrugation Grinding"
    DARK_CONTACT_BAND = "Dark Contact Band"
    FASTENER = "Fastener"
    SPIKE_SCREW = "Spike Screw"
    SET_SCREW = "Set Screw"
    INDENTATION = "Indentation"
    CRACK = "Crack"

class RailFaultBase(BaseModel):
    """
    Base schema for railway fault detection metadata
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(
        ..., 
        description="Unique identifier composed of timestamp + bot_id + location_hash"
    )

class RailFaultCreate(RailFaultBase):
    pred_fault_type: FaultType = Field(
        ..., 
        description="Type of fault detected by the model"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score of the prediction",
        ge=0.0, 
        le=1.0
    )
    latitude: Optional[float] = Field(
        None, 
        description="Latitude of fault location",
        ge=-90.0, 
        le=90.0
    )
    longitude: Optional[float] = Field(
        None, 
        description="Longitude of fault location",
        ge=-180.0, 
        le=180.0
    )
    raw_image_path: str = Field(
        ..., 
        description="Path to original captured image"
    )
    processed_image_path: str = Field(
        ..., 
        description="Path to preprocessed image used for inference"
    )
    bot_id: str = Field(
        ..., 
        description="Identifier of the bot that detected the fault"
    )

class RailFaultResponse(RailFaultBase):
    pred_fault_type: FaultType = Field(
        ..., 
        description="Type of fault detected by the model"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score of the prediction",
        ge=0.0, 
        le=1.0
    )
    latitude: Optional[float] = Field(
        None, 
        description="Latitude of fault location",
        ge=-90.0, 
        le=90.0
    )
    longitude: Optional[float] = Field(
        None, 
        description="Longitude of fault location",
        ge=-180.0, 
        le=180.0
    )
    raw_image_path: str = Field(
        ..., 
        description="Path to original captured image"
    )
    processed_image_path: str = Field(
        ..., 
        description="Path to preprocessed image used for inference"
    )
    bot_id: str = Field(
        ..., 
        description="Identifier of the bot that detected the fault"
    )
    timestamp: datetime = Field(
        None, 
        description="Timestamp when the fault was first detected"
    )