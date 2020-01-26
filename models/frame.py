from enum import IntEnum
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from models.common import Base


class Status(IntEnum):
    Initialized = 1
    Processing = 2
    Finished = 3


class Frame(Base):
    """
    Model of video frame
    """
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    video_id = Column(Integer, ForeignKey('video.id'), nullable=False)
    video = relationship("Video", back_populates="frames", order_by=id)
    elements = relationship("Element",
                            back_populates="frame",
                            cascade="all,delete-orphan")
    processing_status = Column(Integer)

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
                self.elements.append(f)

    def __repr__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} video_id:{self.video_id} elements:{self.elements}>"

    def __str__(self):
        return f"<{self.__tablename__} id:{self.id} name:{self.name} video_id:{self.video_id} elements:{self.elements}>"
