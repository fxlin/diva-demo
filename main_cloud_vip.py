"""
Ingest video frames and perform object detection on frames.

xzl: the controller code (?)
"""

import os
import sys
import logging
from concurrent import futures
import time
import threading
from queue import Queue
import cv2
import numpy as np

import grpc
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import cam_cloud_pb2
import cam_cloud_pb2_grpc
import server_diva_pb2_grpc
import common_pb2
from google.protobuf import empty_pb2

from util import ClockLog

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS
from variables import IMAGE_PATH, OP_FNAME_PATH
from variables import DIVA_CHANNEL_PORT, CONTROLLER_VIDEO_DIRECTORY
from variables import RESULT_IMAGE_PATH, CONTROLLER_PICTURE_DIRECTORY

from constants.grpc_constant import INIT_DIVA_SUCCESS

# from sqlalchemy.orm.exc import MultipleResultsFound
# from models.common import db_session, init_db
# from models.camera import Camera
# from models.video import Video, VideoStatus
# from models.frame import Frame, Status
# from models.element import Element

CHUNK_SIZE = 1024 * 100
OBJECT_OF_INTEREST = 'bicycle'
CROP_SPEC = '350,0,720,400'
YOLO_SCORE_THRE = 0.4
DET_SIZE = 608
IMAGE_PROCESSOR_WORKER_NUM = 1
FRAME_PROCESSOR_WORKER_NUM = 1

SHUTDOWN_SIGNAL = threading.Event()
"""
frame task: (video_name, video_path, frame_number, object_of_interest)
image task: (image name, image data, bounding_boxes)
"""
TaskQueue = Queue(0)
ImageQueue = Queue(10)

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class ImageDoesNotExistException(Exception):
    """Image does not exist"""
    pass


_CAMERA_STORAGE = {'jetson': {'host': CAMERA_CHANNEL_ADDRESS}}


class DivaGRPCServer(server_diva_pb2_grpc.server_divaServicer):
    """
    Implement server_divaServicer of gRPC
    """
    # xzl: proxy req ("get metadata of all stored videos") from client to cam
    def get_videos(self, request, context):
        resp = []

        for _, val in _CAMERA_STORAGE.items():
            camera_channel = grpc.insecure_channel(f"{val['host']}")
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
            req = camera_stub.get_videos(empty_pb2.Empty())
            for v in req.videos:
                resp.append(v)

            camera_channel.close()

        return common_pb2.get_videos_resp(videos=resp)

    # xzl: proxy req ("get res of a previous query") from client to cam 
    def get_video(self, request, context):
        if request.camera and request.camera.name in _CAMERA_STORAGE:
            val = _CAMERA_STORAGE[request.camera.name]
            camera_channel = grpc.insecure_channel(f"{val['host']}")
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)

            _camera = common_pb2.Camera(name=request.camera.name,
                                        address=request.camera.address)

            req = camera_stub.get_video(
                common_pb2.VideoRequest(timestamp=request.timestamp,
                                        offset=request.offset,
                                        video_name=request.video_name,
                                        object_name=request.object_name,
                                        camera=_camera))
            camera_channel.close()
        else:
            raise Exception("Error....")

        return req

    # xzl: proxy req ("process video footage") from client to cam
    def process_video(self, request, context):
        logger.info("process_video")
        
        req_camera = request.camera
        new_req_payload = common_pb2.VideoRequest(
            timestamp=request.timestamp,
            offset=request.offset,
            video_name=request.video_name,
            object_name=request.object_name,
            camera=req_camera)

        try:
            camera_channel = grpc.insecure_channel(req_camera.address)
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
            camera_stub.process_video(new_req_payload)
            camera_channel.close()
        except Exception as err:
            logger.warning(err)

        return empty_pb2.Empty()


def grpc_serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_servicer = DivaGRPCServer()
    server_diva_pb2_grpc.add_server_divaServicer_to_server(
        diva_servicer, server)
    server.add_insecure_port(f'[::]:{DIVA_CHANNEL_PORT}')
    server.start()
    logger.info("GRPC server is runing")

    return server

# xzl: unused?
def draw_box(img, x1, y1, x2, y2):
    rw = float(img.shape[1]) / DET_SIZE
    rh = float(img.shape[0]) / DET_SIZE
    x1, x2 = int(x1 * rw), int(x2 * rw)
    y1, y2 = int(y1 * rh), int(y2 * rh)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


# xzl: unused?
def deploy_operator_on_camera(operator_path: str,
                              camStub: cam_cloud_pb2_grpc.DivaCameraStub):
    f = open(operator_path, 'rb')
    op_data = f.read(CHUNK_SIZE)
    while op_data != b"":
        response = camStub.DeployOp(cam_cloud_pb2.Chunk(data=op_data))
        if response.msg != 'OK':
            logger.warning('DIVA deploy op fails!!!')
            return
        op_data = f.read(CHUNK_SIZE)
    f.close()
    camStub.DeployOpNotify(
        cam_cloud_pb2.DeployOpRequest(name='random', crop=CROP_SPEC))

# xzl: unused?
def process_frame(img_name: str, img_data, det_res):
    """
    Take two arguments: one is the image frame data and the other is the
    response from YOLO agent. Draw boxes around the object of interest.
    """
    # print ('client received: ' + det_res)
    if not det_res or len(det_res) == 0:
        return

    res_items = det_res.split('|')
    res_items = [[float(y) for y in x.split(',')] for x in res_items]
    res_items = list(filter(lambda z: z[0] > YOLO_SCORE_THRE, res_items))

    if not res_items or len(res_items) <= 0:
        return

    img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
    for item in res_items:
        x1, y1, x2, y2 = int(item[1]), int(item[2]), int(item[3]), int(item[4])
        img = draw_box(img, x1, y1, x2, y2)

    img_fname = os.path.join(RESULT_IMAGE_PATH, img_name)
    cv2.imwrite(img_fname, img)

# xzl: unused?
def runDiva():
    # Init the communication channels to camera and yolo
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    yoloChannel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
    yoloStub = det_yolov3_pb2_grpc.DetYOLOv3Stub(yoloChannel)

    response = camStub.InitDiva(
        cam_cloud_pb2.InitDivaRequest(img_path=IMAGE_PATH))
    if response.msg != INIT_DIVA_SUCCESS:
        print('DIVA init fails!!!')
        return

    deploy_operator_on_camera(OP_FNAME_PATH, camStub)

    time.sleep(5)
    pos_num = 0
    clog = ClockLog(5)

    for _ in range(10000):
        time.sleep(0.1)  # emulate network
        clog.log('Retrieved pos num: %d' % (pos_num))

        response = camStub.GetFrame(cam_cloud_pb2.GetFrameRequest(name='echo'))
        if not response.name:
            print('no frame returned...')
            continue

        img_name = response.name
        img_data = response.data
        detected_objects = yoloStub.DetFrame(
            det_yolov3_pb2.DetFrameRequest(data=img_data,
                                           name=str(img_name),
                                           cls=OBJECT_OF_INTEREST))

        det_res = detected_objects.res
        process_frame(img_name, img_data, det_res)

        pos_num += 1

    camChannel.close()
    yoloChannel.close()


if __name__ == '__main__':
    # logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

    # init_db()

    logger.info("Init threads")

    # detection_serve()

    _server = grpc_serve()
    _server.wait_for_termination()

    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logger.info('cloud receiver stops!!!')
        SHUTDOWN_SIGNAL.set()
        _server.stop(0)
    except Exception as err:
        logger.warning(err)
        SHUTDOWN_SIGNAL.set()
        _server.stop(1)

    # runDiva()
