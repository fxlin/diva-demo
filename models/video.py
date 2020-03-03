"""
File for Video table
"""

from enum import IntEnum
from sqlalchemy import Column, String, Integer, DateTime, func, ForeignKey
from sqlalchemy.orm import backref, relationship
from models.common import Base
from models.camera import Camera


class VideoStatus(IntEnum):
    INITIALIZED = 1
    PROCESSING = 2
    COMPLETED = 3
    UNKOWN = 4
    REQUESTING = 5
    WILL_REQUEST = 6


class Video(Base):
    """
    Model of video
    """
    __tablename__ = 'video'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    status = Column(Integer, nullable=False)

    camera_id = Column(Integer, ForeignKey('camera.id'))
    camera = relationship(Camera,
                          backref=backref('videos',
                                          uselist=True,
                                          cascade="all,delete-orphan,delete"))
    name_on_camera = Column(String)
    offset = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now())

    def __init__(self,
                 name: str,
                 path: str,
                 camera_id: int,
                 camera,
                 name_on_camera: str,
                 offset: int,
                 status=VideoStatus.UNKOWN):
        self.name = name
        self.path = path
        self.status = status
        self.camera_id = camera_id
        self.camera = camera
        self.name_on_camera = name_on_camera
        self.offset = offset

    def add_frames(self, frames):
        if frames and not (self.frames is None):
            for f in frames:
                self.frames.append(f)

    def __repr__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} path:{self.path} frames:{self.frames}>"

    def __str__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} path:{self.path} frames:{self.frames}>"
