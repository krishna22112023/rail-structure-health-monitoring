from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Fault(Base):
    __tablename__ = "rail_faults_metadata"
    
    id = Column(Integer, primary_key=True)
    pred_fault_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    raw_image_path = Column(String)
    processed_image_path = Column(String)
    bot_id = Column(String)
    timestamp = Column(DateTime)