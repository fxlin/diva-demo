import unittest
import time
import re

import grpc
import server_diva_pb2_grpc
import server_diva_pb2

import numpy as np
from PIL import Image

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS
from variables import DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT

from YOLOv3_grpc import run_det


class TestYOLO(unittest.TestCase):
    VIDEO_NAME = 'traffic_cam_vid.mp4'
    OBJECT_CLASS = 'motorbike'

    def test_run_dect(self):
        im = Image.open('tests/sample_364.jpeg')
        np_im = np.array(im)
        temp = run_det(np_im, self.OBJECT_CLASS)

        self.assertNotEqual(temp, "", "Fail to perform object detection")
        regex_str = r"([0][.][0-9]*[,][0-9]*[,][0-9]*[,][0-9]*[,][0-9]*[|]?)*"
        self.assertRegex(temp, regex, "Invalid result")


if __name__ == "__main__":
    unittest.main()