"""
add fixtures for test case
"""

import os
import sys
import logging
import time
import ffmpeg
from models.common import db_session, init_db
from variables import CONTROLLER_VIDEO_DIRECTORY
from migration.xsel_server import add_fixtures

video_list = [('temp_video.mp4',
               os.path.join(os.curdir, CONTROLLER_VIDEO_DIRECTORY,
                            'temp_video.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.getLogger('sqlalchemy').setLevel(logging.INFO)

logging.info("Begin to generate fixtures")

_SOURCE_VIDEO = 'example.mp4'
_SAMPLE_VIDEO = 'temp_video.mp4'
_SAMPLE_FOLDER = os.path.join('video')

source_path = os.path.join(_SAMPLE_FOLDER, _SOURCE_VIDEO)
# FIXME test
# p = os.path.join(self.SAMPLE_FOLDER, self.SAMPLE_VIDEO)
p = os.path.join(os.curdir, _SAMPLE_FOLDER, _SAMPLE_VIDEO)


def generate_test_video(source_path: str, output_path: str):
    source_video = ffmpeg.input(source_path)

    # 00:00:10 - 00:00:20
    # FPS is 30
    FPS = 30
    logging.info(f'time {time.time()}')
    source_video.trim(start_frame=10 * FPS,
                      end_frame=20 * FPS).output(output_path).run()
    logging.info(f'time {time.time()} file {p} exists? {os.path.exists(p)}')


if __name__ == "__main__":
    init_db()
    add_fixtures(db_session, video_list)
