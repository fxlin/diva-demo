'''
xzl: run with 

# test CPU
CUDA_VISIBLE_DEVICES=-1 python -m tests.test_yolo

# test GPU (devices can be 1,2,3..)
CUDA_VISIBLE_DEVICES=1 python -m tests.test_yolo

'''

# import unittest

import numpy as np
from PIL import Image

# from YOLOv3_grpc import run_det
from YOLOv3_grpc import _detect


# class TestYOLO(unittest.TestCase):
VIDEO_NAME = 'traffic_cam_vid.mp4'
OBJECT_CLASS = 'motorbike'

def test_run_dect():
    for _ in range(1,10):
        im = Image.open('tests/sample_364.jpeg')
        np_im = np.array(im)
        # temp = run_det(np_im, self.OBJECT_CLASS)
        temp = _detect(np_im, 0.3)
        print(temp)
        # self.assertNotEqual(temp, "", "Fail to perform object detection")
        # regex_str = r"([0][.][0-9]*[,][0-9]*[,][0-9]*[,][0-9]*[,][0-9]*[|]?)*"
        # self.assertRegex(temp, regex_str, "Invalid result")

if __name__ == "__main__":
    test_run_dect()