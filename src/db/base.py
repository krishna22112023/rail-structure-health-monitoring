from sqlalchemy.orm import declarative_base

Base = declarative_base()

def import_models():
    # Import models here within the function to avoid circular imports
    from src.models.fault_metadata import Fault
