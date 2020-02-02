import unittest

import numpy as np
from PIL import Image

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
        self.assertRegex(temp, regex_str, "Invalid result")


if __name__ == "__main__":
    unittest.main()