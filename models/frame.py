from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref
from common import Base
from video import Video


class Frame(Base):
    """
    Model of video frame
    """
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    video_id = Column(Integer, ForeignKey('video.id'))
    video = relationship("video", backref=backref("frames", cascade="all, delete-orphan"))

    def __init__(self, name: str, video_id: int, video: Video):
        self.name = name
        self.video_id = video_id
        self.video = video