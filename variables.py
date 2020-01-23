import os

CAMERA_CHANNEL_ADDRESS = 'camera:10086'
CAMERA_CHANNEL_PORT = '10086'

YOLO_CHANNEL_ADDRESS = 'yolo:10088'
YOLO_CHANNEL_PORT = "10088"

IMAGE_PATH = '/media/YOLO-RES-720P/jpg/chaweng-1_10FPS/'
CSV_PATH = '/media/YOLO-RES-720P/out/chaweng-1_10FPS.csv'
OP_FNAME_PATH = '/media/YOLO-RES-720P/exp/chaweng/models/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'
OP_DIR = 'result/ops'

# Folder that saves processed video frames
VIDEO_FOLDER = 'video'
RESULT_PATH = 'result'
RESULT_IMAGE_PATH = os.path.join(RESULT_PATH, 'retrieval_imgs')

# FIXME
FAKE_IMAGE_DIRECTOR_PATH = '/media/YOLO-RES-720P/jpg/lausanne-1'

DEFAULT_POSTGRES_USER = "postgres"
DEFAULT_POSTGRES_PASSWORD = "xsel_postgres"
DEFAULT_POSTGRES_DB = "xsel_test"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_HOST = "mypgdb"

# JPG_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/jpg'
# CSV_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/out'
# RES_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/exp'
