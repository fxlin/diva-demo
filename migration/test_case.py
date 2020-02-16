"""
add fixtures for test case
"""

import os
import sys
import logging
import time
import cv2 as cv
from models.common import db_session, init_db
from variables import CONTROLLER_VIDEO_DIRECTORY
from migration.xsel_server import add_fixtures

video_list = [('temp_video.mp4',
               os.path.join(CONTROLLER_VIDEO_DIRECTORY, 'temp_video.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.getLogger('sqlalchemy').setLevel(logging.INFO)

logging.info("Begin to generate fixtures")

_SOURCE_VIDEO = 'example.mp4'
_SAMPLE_VIDEO = 'temp_video.mp4'
_SAMPLE_FOLDER = CONTROLLER_VIDEO_DIRECTORY

source_path = os.path.join(_SAMPLE_FOLDER, _SOURCE_VIDEO)
# FIXME test
# p = os.path.join(self.SAMPLE_FOLDER, self.SAMPLE_VIDEO)
p = os.path.join(_SAMPLE_FOLDER, _SAMPLE_VIDEO)


def generate_test_video(source_path: str, output_path: str, start_second: int,
                        end_second: int):
    source_video = cv.VideoCapture(source_path)

    if not source_video.isOpened():
        raise Exception("Video is not opened")

    target_width = source_video.get(cv.CAP_PROP_FRAME_WIDTH)  # float
    target_height = source_video.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    target_fps = source_video.get(cv.CAP_PROP_FPS)

    source_video.set(cv.CAP_PROP_POS_FRAMES, target_fps * start_second)
    counter = target_fps * end_second

    _fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output_video = cv.VideoWriter(output_path, _fourcc, target_fps,
                                  (target_width, target_height))

    logging.info(f'time {time.time()}')
    while counter >= 0:
        frame = source_video.read()
        output_video.write(frame)
        counter -= 1
    logging.info(f'time {time.time()} file {p} exists? {os.path.exists(p)}')

    source_video.release()
    output_video.release()

    # 00:00:10 - 00:00:20
    # ffmpeg
    # source_video.trim(start_frame=10 * FPS,
    #                   end_frame=20 * FPS).output(output_path).run()


if __name__ == "__main__":
    init_db()
    add_fixtures(db_session, video_list)
