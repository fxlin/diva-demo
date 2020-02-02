from typing import Tuple

from sqlalchemy import Column, String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship, backref
from models.common import Base
from models.frame import Frame


class Element(Base):
    __tablename__ = 'image_element'
    id = Column(Integer, primary_key=True)
    object_class = Column(String)
    box_coordinate = Column(String)
    frame_id = Column(Integer, ForeignKey('frame.id'), nullable=False)
    frame = relationship(Frame,
                         backref=backref('elements',
                                         uselist=True,
                                         cascade="all,delete-orphan,delete"))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now())

    def __init__(self, object_class: str, box_coordinate: str, frame_id: int):
        self.object_class = object_class
        self.box_coordinate = box_coordinate
        self.frame_id = frame_id

    @staticmethod
    def coordinate_iterable_to_str(
            coordinate: 'Tuple[float, float, float, float]') -> str:
        if not (isinstance(coordinate, tuple) or isinstance(coordinate, list)):
            raise TypeError(
                f'{coordinate} is not an instance of tuple or list')

        if len(coordinate) != 4:
            raise ValueError(f'size of {coordinate} is not 4')

        return f'{coordinate[0]},{coordinate[1]},{coordinate[2]},{coordinate[3]}'

    @staticmethod
    def is_valid_coordinate_string(coordinate_string: str) -> bool:
        if not isinstance(coordinate_string, str):
            raise TypeError("not an instance of tuple")

        temp = coordinate_string.split(',')
        if len(temp) != 4:
            return False
        return True

    @staticmethod
    def coordinate_string_to_tuple(coordinate_string: str) -> tuple:
        if not Element.is_valid_coordinate_string(coordinate_string):
            raise ValueError(f"{coordinate_string} is not a valid coordinate")
        temp = map(lambda x: float(x), coordinate_string.split(","))
        return tuple(temp)

    def __repr__(self):
        return f"<{self.__tablename__} id:{self.id} object_class:{self.object_class} box_coordinate:{self.box_coordinate} frame_id:{self.frame_id}>"

    def __str__(self):
        return f"<{self.__tablename__} id:{self.id} object_class:{self.object_class} box_coordinate:{self.box_coordinate} frame_id:{self.frame_id}>"
