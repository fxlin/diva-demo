"""
Init DB
"""

import os
import sys
import logging
from models.common import db_session, init_db
from models.video import Video
from variables import CONTROLLER_VIDEO_DIRECTORY

# FIXME
fixture_list = [('sonic.mp4',
                 os.path.join(os.curdir, CONTROLLER_VIDEO_DIRECTORY,
                              'sonic.mp4')),
                ('traffic_cam_vid.mp4',
                 os.path.join(os.curdir, CONTROLLER_VIDEO_DIRECTORY,
                              'traffic_cam_vid.mp4')),
                ('example.mp4',
                 os.path.join(os.curdir, CONTROLLER_VIDEO_DIRECTORY,
                              'example.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.getLogger('sqlalchemy').setLevel(logging.INFO)

logging.info("Begin to initialize DB, wait 5 seconds")


def add_fixtures(db_session, video_list: 'List[Tuple[str, str]]'):
    try:
        for p in video_list:
            v = Video(p[0], p[1])
            db_session.add(v)
        db_session.commit()

        logging.info(
            db_session.query(Video).filter(
                Video.name == video_list[0][0]).all())
    except Exception as err:
        logging.error(err)
        logging.error('Failed to initialize db')
        db_session.rollback()
        exit(1)
    finally:
        db_session.remove()


if __name__ == "__main__":
    init_db()
    add_fixtures(db_session, fixture_list)
    logging.info("Bootstrap DB")
