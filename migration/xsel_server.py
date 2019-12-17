"""
Init DB
"""

import logging, time
from models.common import db_session, init_db
from models.video import Video
from variables import VIDEO_FOLDER
import os, sys

# FIXME
video_list = [('sonic.mp4', os.path.join(os.curdir, VIDEO_FOLDER,
                                         'sonic.mp4')),
              ('traffic_cam_vid.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'traffic_cam_vid.mp4')),
              ('example.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'example.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)
logging.getLogger('sqlalchemy').setLevel(logging.INFO)

logger.info("Begin to initialize DB, wait 5 seconds")

init_db()

session = db_session()

try:
    for p in video_list:
        v = Video(p[0], p[1], [])
        session.add(v)
    session.commit()
except Exception as err:
    print(err)
    print('Failed to initialize db')
    session.rollback()
    exit(1)
finally:
    session.remove()

logging.info("Bootstrap DB")
