from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float
)

from datetime import datetime

from database import Base


class Asset(Base):

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)

    asset_code = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    vendor = Column(String)

    supply = Column(String)

    warranty = Column(String)

    support = Column(String)

    location = Column(String)

    status = Column(
        String,
        default="active"
    )


class Inspection(Base):

    __tablename__ = "inspections"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    asset_code = Column(
        String,
        index=True,
        nullable=False
    )

    qr_status = Column(String)

    ai_prediction = Column(String)

    ai_confidence = Column(Float)

    inspector = Column(String)

    image_path = Column(String)

    notes = Column(Text)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow
    )
