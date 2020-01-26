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

from main_cloud import ImageProcessor, FrameProcessor, YOLO_SCORE_THRE

EXAMPLE_IMAGE_DETECTION_RESULT = '0.8701794743537903,77,242,98,278|0.36665189266204834,162,192,170,204'
EXAMPLE_IMAGE_PATH = os.path.join('tests', 'sample_364.jpeg')
TEMP_IMAGE_PATH = os.path.join('tests', 'temp_sample_364.jpeg')


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
        # Reference https://blog.csdn.net/JohinieLi/article/details/81012572
        im = Image.open(EXAMPLE_IMAGE_PATH)
        res = FrameProcessor.get_bounding_boxes(EXAMPLE_IMAGE_DETECTION_RESULT)

        ImageProcessor.process_frame(TEMP_IMAGE_PATH, im, res)

        self.assertTrue(os.path.exists(TEMP_IMAGE_PATH),
                        f'file {TEMP_IMAGE_PATH} does not exist')

        # FIXME compare images


if __name__ == "__main__":
    init_db()
    unittest.main()