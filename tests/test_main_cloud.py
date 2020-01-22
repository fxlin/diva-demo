import unittest
import time
import os

import grpc
import server_diva_pb2_grpc
import server_diva_pb2

import numpy as np
from PIL import Image

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS
from variables import DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT

from models.common import db_session, init_db
from models.video import Video
from models.frame import Frame
from models.element import Element

from main_cloud import ImageProcessor, FrameProcessor, YOLO_SCORE_THRE

EXAMPLE_IMAGE_DETECTION_RESULT = '0.8701794743537903,77,242,98,278|0.36665189266204834,162,192,170,204'
EXAMPLE_IMAGE_PATH = os.path.join('tests', 'sample_364.jpeg')
TEMP_IMAGE_PATH = os.path.join('tests', 'temp_sample_364.jpeg')


def query_video(object_name: str, video_name: str):
    session = db_session()
    res = session.query(Video).filter(Video.name == video_name).all()
    if len(res) != 1:
        raise Exception("Duplicated videos")

    temp = session.query(Frame).filter(Frame.video_id == res[0].id).all()
    if len(temp) > 0:
        raise Exception("Table frame is not empty")

    db_session.remove()
    return res[0].id


def detect_object(object_name: str, video_name: str, video_id: int):
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        _ = stub.detect_object_in_video(
            server_diva_pb2.object_video_pair(object_name=object_name,
                                              video_name=video_name))

    time.sleep(3)

    session = db_session()
    temp = session.query(Frame).filter(Frame.video_id == video_id).all()
    if len(temp) == 0:
        raise Exception(f'No frames related to {video_name}')

    _temp = session.query(Element).all()

    return _temp


def simulate_detection():
    OBJECT_NAME = 'motorbike'
    VIDEO_NAME = 'traffic_cam_vid.mp4'

    v_id = query_video(OBJECT_NAME, VIDEO_NAME)

    elements = detect_object(OBJECT_NAME, VIDEO_NAME)

    # FIXME
    print(elements)

    return elements


class TestObjectDetection(unittest.TestCase):
    OBJECT_NAME = 'motorbike'
    VIDEO_NAME = 'traffic_cam_vid.mp4'

    def test_query_video(self):
        try:
            temp_res = query_video(self.OBJECT_NAME, self.VIDEO_NAME)
        except Exception as err:
            self.fail(f'calling query_video but got {err}')

        self.assertEqual(2, temp_res, 'id of traffic_cam_vid.mp4 should be 2')


class TestFrameProcessor(unittest.TestCase):
    def test_get_bounding_boxes(self):
        EXPECTED_BOUNDIN_BOXES = [(77, 242, 98, 278), (162, 192, 170, 204)]
        res = FrameProcessor.get_bounding_boxes(EXAMPLE_IMAGE_DETECTION_RESULT)
        for one, two in zip(EXPECTED_BOUNDIN_BOXES, res):
            self.assertTupleEqual(
                one, two,
                f'expect to get identical bounding boxes given the threshold of object detection is {YOLO_SCORE_THRE}'
            )


class TestImageProcessor(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)

    def test_process_frame(self):
        im = Image.open(EXAMPLE_IMAGE_PATH)
        res = FrameProcessor.get_bounding_boxes(EXAMPLE_IMAGE_DETECTION_RESULT)

        ImageProcessor.process_frame(TEMP_IMAGE_PATH, im, res)

        self.assertTrue(os.path.exists(TEMP_IMAGE_PATH), f'file {TEMP_IMAGE_PATH} does not exist')

        # FIXME compare images


if __name__ == "__main__":
    unittest.main()