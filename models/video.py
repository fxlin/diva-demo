"""
File for Video table
"""

from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship, backref
from common import Base


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

    def __init__(self, name: str, path: str, frames=[]):
        self.name = name
        self.path = path
        self.frames = frames