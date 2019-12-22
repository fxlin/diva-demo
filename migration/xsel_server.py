"""
Init DB
"""

import os
import sys
import logging
from models.common import db_session, init_db
from models.video import Video
from variables import VIDEO_FOLDER

# FIXME
video_list = [('sonic.mp4', os.path.join(os.curdir, VIDEO_FOLDER,
                                         'sonic.mp4')),
              ('traffic_cam_vid.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'traffic_cam_vid.mp4')),
              ('example.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'example.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logging.getLogger('sqlalchemy').setLevel(logging.INFO)

logging.info("Begin to initialize DB, wait 5 seconds")

init_db()

try:
    for p in video_list:
        v = Video(p[0], p[1])
        db_session.add(v)
    db_session.commit()

    logging.info(
        db_session.query(Video).filter(Video.name == video_list[0][0]).one())
except Exception as err:
    logging.error(err)
    logging.error('Failed to initialize db')
    db_session.rollback()
    exit(1)
finally:
    db_session.remove()

logging.info("Bootstrap DB")
