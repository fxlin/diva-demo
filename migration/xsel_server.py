import logging
from models.common import session_factory, init_db
from models.video import Video
from variables import VIDEO_FOLDER
import os, sys

session = session_factory()

# FIXME
video_list = [('sonic.mp4', os.path.join(os.curdir, VIDEO_FOLDER,
                                         'sonic.mp4')),
              ('traffic_cam_vid.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'traffic_cam_vid.mp4')),
              ('example.mp4',
               os.path.join(os.curdir, VIDEO_FOLDER, 'example.mp4'))]

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(user)-8s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

logger.info("Begin to initialize DB")

init_db()

logger.info("Table created")

session.begin()

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
    session.close()

logging.info("Bootstrap DB")
