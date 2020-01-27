import unittest
import os

import numpy as np
from PIL import Image

from models.common import db_session, init_db
from models.video import Video
from models.frame import Frame

from main_cloud import ImageProcessor, FrameProcessor, YOLO_SCORE_THRE

EXAMPLE_IMAGE_DETECTION_RESULT = '0.8701794743537903,77,242,98,278|0.36665189266204834,162,192,170,204'
EXAMPLE_IMAGE_PATH = os.path.join('tests', 'sample_364.jpeg')
TEMP_IMAGE_PATH = os.path.join('tests', 'temp_sample_364.jpeg')


class TestFrameProcessor(unittest.TestCase):
    OBJECT_NAME = 'motorbike'
    VIDEO_NAME = 'temp_video.mp4'
    VIDEO_FOLDER = 'video'

    def test_get_bounding_boxes(self):
        EXPECTED_BOUNDIN_BOXES = [(77, 242, 98, 278), (162, 192, 170, 204)]
        res = FrameProcessor.get_bounding_boxes(EXAMPLE_IMAGE_DETECTION_RESULT)
        for one, two in zip(EXPECTED_BOUNDIN_BOXES, res):
            self.assertTupleEqual(
                one, two,
                f'expect to get identical bounding boxes given the threshold of object detection is {YOLO_SCORE_THRE}'
            )

    def test_extract_frame_nums(self):
        p = os.path.join(self.VIDEO_FOLDER, self.VIDEO_NAME)
        frame_indices = FrameProcessor.extract_frame_nums(p)
        print(f'sample frames {frame_indices}')
        print(f'video info {FrameProcessor.video_info(p)}')
        self.assertCountEqual(frame_indices, list(set(frame_indices)),
                              "should only contain distinct integers")


class TestImageProcessor(unittest.TestCase):
    def tearDownClass(self):
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