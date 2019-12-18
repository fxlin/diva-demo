from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref
from models.common import Base


class Frame(Base):
    """
    Model of video frame
    """
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    video_id = Column(Integer, ForeignKey('video.id'), nullable=False)
    video = relationship("Video", back_populates="frames")
    elements = relationship("Element",
                            back_populates="frame",
                            cascade="all,delete-orphan")

    def __init__(self, name: str, video_id: int, video):
        self.name = name
        self.video_id = video_id
        self.video = video

    def add_elements(self, elements):
        if elements:
            for e in elements:
                self.elements.append(f)