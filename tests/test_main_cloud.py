import logging
import time

import grpc
import server_diva_pb2_grpc
import server_diva_pb2

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS
from variables import DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT

from models.common import db_session, init_db
from models.video import Video
from models.frame import Frame
from models.element import Element


def query_video(object_name: str, video_name: str):
    session = db_session()
    res = session.query(Video).filter(Video.id == video_id).all()
    if len(res) != 1:
        raise Exception("Duplicated videos")

    temp = session.query(Frame).filter(Frame.video_id == res[0].id).all()
    if len(temp) > 0:
        raise Exception("Table frame is not empty")

    session.remove()
    return res[0].id

def detect_object(object_name: str, video_name: str, video_id: int):
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        _ = stub.detect_object_in_video(server_diva_pb2.object_video_pair(object_name=object_name, video_name=video_name))
    
    time.sleep(3)

    session = db_session()
    temp = session.query(Frame).filter(Frame.video_id == video_id).all()
    if len(temp) == 0:
        raise Exception(f'No frames related to {video_name}')

    _temp = session.query(Element).all()

    return _temp



def simulate_detection():
    OBJECT_NAME = 'bike'
    VIDEO_NAME = 'traffic_cam_vid.mp4'

    v_id = query_video(OBJECT_NAME, VIDEO_NAME)

    elements = detect_object(OBJECT_NAME, VIDEO_NAME)

    # FIXME
    print(elements)

    return elements
