from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref
from models.common import Base


class Element(Base):
    __tablename__ = 'element'
    id = Column(Integer, primary_key=True)
    object_class = Column(String)
    box_coordinate = Column(String)
    frame_id = Column(Integer, ForeignKey('frame.id'))
    frame = relationship("frame",
                         backref=backref("elements",
                                         cascade="all, delete-orphan"))

    def __init__(self, object_class: str, box_coordinate: str, frame_id: int,
                 frame):
        self.object_class = object_class
        self.box_coordinate = box_coordinate
        self.frame_id = frame_id
        self.frame = frame

    @staticmethod
    def coordinate_iterable_to_str(
            coordinate: 'Tuple[float, float, float, float]') -> str:
        if not (isinstance(coordinate, tuple) or isinstance(coordinate, list)):
            raise TypeError(f'{coordinate} is not an instance of tuple or list')

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
