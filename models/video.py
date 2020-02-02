"""
File for Video table
"""

from sqlalchemy import Column, String, Integer, DateTime, func
from sqlalchemy.orm import relationship
from models.common import Base


class Video(Base):
    """
    Model of video
    """
    __tablename__ = 'video'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    path = Column(String)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now())

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def add_frames(self, frames):
        if frames:
            for f in frames:
                self.frames.append(f)

    def __repr__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} path:{self.path} frames:{self.frames}>"

    def __str__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} path:{self.path} frames:{self.frames}>"