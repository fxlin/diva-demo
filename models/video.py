"""
File for Video table
"""

from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship, backref
from models.common import Base
from models.frame import Frame


class Video(Base):
    """
    Model of video
    """
    __tablename__ = 'video'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    path = Column(String)
    frames = relationship("frame",
                          back_populates="video",
                          cascade="all, delete-orphan")

    def __init__(self, name: str, path: str, frames: 'list[Frame]'):
        self.name = name
        self.path = path
        if frames:
            for f in frames:
                self.frames.append(f)
