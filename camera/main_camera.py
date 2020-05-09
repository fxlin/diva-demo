import time
import os
import sys
import json
import cv2
# from threading import Thread, Event, Lock
from concurrent import futures
import logging
from queue import PriorityQueue

import grpc

import cam_cloud_pb2_grpc
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import common_pb2
from google.protobuf import empty_pb2

from variables import CAMERA_CHANNEL_PORT

from camera.camera_constants import VIDEO_FOLDER, _HOST, _PORT, STATIC_FOLDER, WEB_APP_DNS
from camera.camera_constants import _NAME, _ADDRESS, YOLO_CHANNEL_ADDRESS

# from camera.ml import Operator

# from util import *

CHUNK_SIZE = 1024 * 100
OP_BATCH_SZ = 16

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# FIXME
OP_FNAME_PATH = '/home/bryanjw01/workspace/test_data/source/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'

PQueue = PriorityQueue()


class DivaCameraServicer(cam_cloud_pb2_grpc.DivaCameraServicer):
    img_dir = None
    cur_op_data = bytearray(b'')
    cur_op_name = ''
    send_buf = []
    is_queerying = False
    operator_ = None

    # locker = Lock()

    def cleaup(self):
        self.cur_op_data = bytearray(b'')
        self.cur_op_name = ""

    def __init__(self):
        cam_cloud_pb2_grpc.DivaCameraServicer.__init__(self)
        # self.op = Operator(OP_FNAME_PATH)

        # self.op_worker = OP_WORKER()
        # self.op_worker.setDaemon(True)
        # self.op_worker.start()
        pass

    # rpc get_videos(google.protobuf.Empty) returns (common.get_videos_resp) {};
    # rpc process_video(common.VideoRequest) returns (google.protobuf.Empty) {};
    def process_video(self, request, context):
        # FIXME
        print(f"process_video: {request} ")
        video_path = os.path.join(VIDEO_FOLDER, request.video_name)
        temp_name = request.video_name
        v_name = temp_name.split('/')[-1]
        video_folder_name = '.'.join(v_name.split('.')[:-1])

        target_class = request.object_name

        source = cv2.VideoCapture(video_path)
        counter = (request.offset // 30) * 30

        source.set(cv2.CAP_PROP_POS_FRAMES, request.offset)

        score_obj = {}

        with grpc.insecure_channel(YOLO_CHANNEL_ADDRESS) as channel:
            stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)

            while source.isOpened():
                ret, frame = source.read()
                if not ret:
                    break

                if (counter % 30) == 0:
                    op_score = self.op.predict_image(frame, '350,0,720,400')
                    if (op_score <= 0.3):
                        counter += 1
                        continue
                    else:
                        # FIXME move the following computation to
                        # another thread
                        PQueue.put((op_score, frame))

                    _height, _width, _chan = frame.shape
                    _img = common_pb2.Image(data=frame.tobytes(),
                                            height=_height,
                                            width=_width,
                                            channel=_chan)

                    req = det_yolov3_pb2.DetectionRequest(
                        image=_img,
                        name=f'{counter}.jpg',
                        threshold=0.3,
                        targets=[target_class])
                    resp = stub.Detect(req)

                    exist_target = False

                    temp_score_map = {}

                    # draw bbox on the image
                    for idx, ele in enumerate(resp.elements):
                        if ele.class_name != target_class:
                            continue

                        x1, y1, x2, y2 = ele.x1, ele.y1, ele.x2, ele.y2

                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                              (0, 255, 0), 3)
                        temp_score_map[str(idx)] = ele.confidence
                        exist_target = True

                    if exist_target:
                        img_path = os.path.join(
                            STATIC_FOLDER, *[
                                video_folder_name, target_class, 'images',
                                f'{counter}.jpg'
                            ])
                        cv2.imwrite(img_path, frame)
                        score_obj[str(counter)] = temp_score_map

                counter += 1

        source.release()

        with open(
                os.path.join(STATIC_FOLDER,
                             *[video_folder_name, target_class,
                               'scores.json']), 'w') as fptr:
            fptr.write(json.dumps(score_obj))

        return empty_pb2.Empty()

    def get_videos(self, request, context):
        _files = os.listdir(VIDEO_FOLDER)
        video_files = list(
            filter(lambda x: x.split('.')[-1] in ['mp4', 'mkv'], _files))

        _cam = common_pb2.Camera(name=_NAME, address=_ADDRESS)
        video_list = []
        for f in video_files:
            _video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, f))
            video_folder_name = '.'.join(f.split('.')[:-1])
            video_list.append(
                common_pb2.video_metadata(
                    name=f,
                    camera=_cam,
                    video_url=f'{WEB_APP_DNS}/{video_folder_name}/{f}',
                    images_url='',
                    score_file_url='',
                    frames=int(_video.get(cv2.CAP_PROP_FRAME_COUNT))))

            _video.release()

        return common_pb2.get_videos_resp(videos=video_list)

    def get_video(self, request, context):
        temp_name = request.video_name
        v_name = temp_name.split('/')[-1]
        video_folder_name = '.'.join(v_name.split('.')[:-1])

        object_name = request.object_name
        images_url = f'{WEB_APP_DNS}/{video_folder_name}/{object_name}/images/'

        score_file_url = f'{WEB_APP_DNS}/{video_folder_name}/{object_name}/scores.json'

        _video = cv2.VideoCapture(
            os.path.join(VIDEO_FOLDER, request.video_name))
        frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
        _video.release()
        return common_pb2.video_metadata(name=request.video_name,
                                         frames=frames,
                                         video_url="",
                                         score_file_url=score_file_url,
                                         images_url=images_url,
                                         camera=request.camera,
                                         object_name=object_name)


def serve():
    logger.info('Init camera service')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_cam_servicer = DivaCameraServicer()
    cam_cloud_pb2_grpc.add_DivaCameraServicer_to_server(
        diva_cam_servicer, server)
    server.add_insecure_port(f'[::]:{CAMERA_CHANNEL_PORT}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logger.warning('DivaCloud stop!!!')
        server.stop(0)


if __name__ == '__main__':
    serve()
