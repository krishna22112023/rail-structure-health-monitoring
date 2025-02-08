from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session

import logging
logger = logging.getLogger(__name__)

from src.models.fault_metadata import Fault
from src.schemas.fault_metadata import RailFaultCreate, RailFaultResponse


def create(db:Session, item: RailFaultCreate) -> RailFaultResponse:
    """
    Create a new static crawler job.

    Args:
    db (Session): SQLAlchemy session connected to the database.
    item (RailFaultCreate): Pydantic model for updating the fault data.

    Returns:
    RailFaultResponse : Raises ValidationError if the object could not be validated.
    """
    db_file = Fault(**item.model_dump())
    db_file.timestamp = datetime.now()
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    logger.info(f"entry created in sql : {item.id} with fault type {item.fault_type}")
    return RailFaultResponse.model_validate(db_file)  