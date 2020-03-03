"""
File for Camera table
"""

from sqlalchemy import Column, String, Integer, DateTime, func
from models.common import Base


class Camera(Base):
    """
    Model of Camera
    """
    __tablename__ = 'camera'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    ip = Column(String, nullable=False)
    port = Column(String, nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now())

    def __init__(self, name: str, ip: str, port: int):
        self.name = name
        self.ip = ip
        self.port = port

    def __repr__(self):
        msg = [
            "<", f'{self.__tablename__}', f'id:{self.id}', 'name:{self.name}',
            f'path:{self.ip}', f"frames:{self.frames}", ">"
        ]
        return " ".join(msg)

    def __str__(self):
        msg = [
            "<", f'{self.__tablename__}', f'id:{self.id}', 'name:{self.name}',
            f'path:{self.ip}', f"frames:{self.frames}", ">"
        ]
        return " ".join(msg)
