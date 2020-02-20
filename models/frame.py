from enum import IntEnum
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship, backref
from models.common import Base
from models.video import Video


class Status(IntEnum):
    Initialized = 1
    Processing = 2
    Finished = 3
    Failed = 4


class Frame(Base):
    """
    Model of video frame
    """
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    video_id = Column(Integer, ForeignKey('video.id'), nullable=False)
    video = relationship(Video,
                         backref=backref('frames',
                                         uselist=True,
                                         cascade="all,delete-orphan,delete"))
    processing_status = Column(Integer)
    # https://stackoverflow.com/questions/13370317/sqlalchemy-default-datetime/13370382
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now())

    def __init__(self,
                 name: str,
                 video_id: int,
                 video,
                 status=Status.Initialized):
        self.name = name
        self.video_id = video_id
        self.video = video
        self.processing_status = status

    def add_elements(self, elements):
        if elements:
            for e in elements:
                self.elements.append(e)

    def __repr__(self):
        msg = " ".join([
            "<", f"{self.__tablename__}", f"id:{self.id}", f"name:{self.name}",
            f"video_id:{self.video_id}", f"elements:{self.elements}", ">"
        ])
        return msg

    def __str__(self):
        msg = " ".join([
            "<", f"{self.__tablename__}", f"id:{self.id}", f"name:{self.name}",
            f"video_id:{self.video_id}", f"elements:{self.elements}", ">"
        ])
        return msg
