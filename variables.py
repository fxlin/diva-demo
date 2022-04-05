import os, threading
import typing
from videostore import VideoLib

# CAMERA_CHANNEL_ADDRESS = '10.10.10.10:10086'  # rpi
CAMERA_CHANNEL_ADDRESS = '127.0.0.1:10086' # the same machine
CAMERA_CHANNEL_PORT = '10086'

#YOLO_CHANNEL_ADDRESS = 'yolo:10088'
YOLO_CHANNEL_ADDRESS = '127.0.0.1:10088'
# YOLO_CHANNEL_ADDRESS = '128.46.76.161:10088'

# set this to '' for testing
# YOLO_CHANNEL_ADDRESS = ''

YOLO_CHANNEL_PORT = "10088"

#DIVA_CHANNEL_ADDRESS = '127.0.0.1:10090'
DIVA_CHANNEL_ADDRESS = '10.10.10.3:10090'
DIVA_CHANNEL_PORT = '10090'

# xzl: below unused? 
IMAGE_PATH = '/media/YOLO-RES-720P/jpg/chaweng-1_10FPS/'
CSV_PATH = '/media/YOLO-RES-720P/out/chaweng-1_10FPS.csv'
OP_FNAME_PATH = '/media/YOLO-RES-720P/exp/chaweng/models/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'
OP_DIR = 'result/ops'

# WEBSERVER
WEB_PICTURE_FOLDER = os.path.join('web', 'static', 'output')

#CFG_RESULT_PATH = 'result'
#CFG_PREVIEW_PATH = 'preview'

CFG_RESULT_PATH = 'server/static/result'
CFG_PREVIEW_PATH = 'server/static/preview'


# RESULT_IMAGE_PATH = os.path.join(RESULT_PATH, 'retrieval_imgs') # unused?

# FIXME
FAKE_IMAGE_DIRECTOR_PATH = '/media/YOLO-RES-720P/jpg/lausanne-1'

'''
# POSTGRES
DEFAULT_POSTGRES_USER = "postgres"
DEFAULT_POSTGRES_PASSWORD = "xsel_postgres"
DEFAULT_POSTGRES_DB = "xsel_test"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_HOST = "mypgdb"
'''

# CAMERA
VIDEO_FOLDER='/data'

# xzl
CFG_QUERY_CMDS = ["PAUSE", "ABORT", "RESUME", "RESET"]

class NO_DESIRED_OBJECT(Exception):
    """does not contain desired object"""
    def __str__(self):
        return """does not contain desired object"""

# local video stores. video_name -> VideoStore
# will be set up by control when it starts & connects to cam
the_videolib_results:VideoLib = VideoLib(CFG_RESULT_PATH)
the_videolib_preview:VideoLib = VideoLib(CFG_PREVIEW_PATH)
