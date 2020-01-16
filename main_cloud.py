"""
Ingest video frames and perform object detection on frames.
"""

import os, sys
import logging
from concurrent import futures
import time
import threading
from queue import Queue
from typing import List, Tuple

import cv2
import numpy as np
import ffmpeg

import grpc
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import cam_cloud_pb2
import cam_cloud_pb2_grpc
import server_diva_pb2_grpc
import server_diva_pb2

from util import ClockLog

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS, IMAGE_PATH, OP_FNAME_PATH
from variables import FAKE_IMAGE_DIRECTOR_PATH, DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT
from variables import RESULT_IMAGE_PATH, VIDEO_FOLDER

from constants.grpc_constant import INIT_DIVA_SUCCESS

from models.common import db_session, init_db
from models.video import Video
from models.frame import Frame, Status
from models.element import Element

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

FORMAT = '%(asctime)-15s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)


class ImageProcessor(threading.Thread):
    def run(self):
        while not SHUTDOWN_SIGNAL.is_set():
            while not ImageQueue.empty():
                self.consume_image_task()

    def consume_image_task(self):
        task = ImageQueue.get(block=False)
        if not task or len(task) == 0:
            return

        image_name = task[0]
        image_data = task[1]
        bounding_boxes = task[2]
        self.process_frame(image_name, image_data, bounding_boxes)

    def process_frame(self, img_name: str, img_data, res_items: 'tuple'):
        """
        Take two arguments: one is the image frame data and the other is the
        response from YOLO agent. Draw boxes around the object of interest.
        """
        if not res_items or len(res_items) <= 0:
            return

        img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
        for item in res_items:
            x1, y1, x2, y2 = int(item[1]), int(item[2]), int(item[3]), int(
                item[4])
            img = draw_box(img, x1, y1, x2, y2)

        img_fname = os.path.join(RESULT_IMAGE_PATH, img_name)
        cv2.imwrite(img_fname, img)


class FrameProcessor(threading.Thread):
    def run(self):
        while not SHUTDOWN_SIGNAL.is_set():
            while not TaskQueue.empty():
                self.detect_object()

    @staticmethod
    def video_info(video_path) -> dict:
        if not os.path.exists(video_path):
            raise ValueError(f"path {video_path} does not exist")

        probe = ffmpeg.probe(video_path)

        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise ValueError(f'{video_path} is invalid\nNo video stream found')

        return video_stream

    @staticmethod
    def read_frame_as_jpeg(in_filename: str, frame_num: int):
        out, err = (ffmpeg.input(in_filename).filter(
            'select', 'gte(n,{})'.format(frame_num)).output(
                'pipe:', vframes=1, format='image2',
                vcodec='mjpeg').run(capture_stdout=True))
        return out

    @staticmethod
    def extract_frame_nums(video_path: str) -> 'List[int]':
        video_stream = FrameProcessor.video_info(video_path)
        num_of_frames = int(video_stream['nb_frames'])

        index = 0
        frame_list = []
        while index < num_of_frames:
            frame_list.append(index)
            index += 10
        return frame_list

    @staticmethod
    def extract_one_frame(video_path: str, frame_num: int) -> np.ndarray:
        if not os.path.exists(video_path):
            raise ValueError(f'video does not exist: {video_path}')

        return FrameProcessor.read_frame_as_jpeg(video_path, frame_num)

    @staticmethod
    def get_bounding_boxes(yolo_result: str) -> 'List[Tuple[int, int, int, int]]':
        if not yolo_result or len(yolo_result) == 0:
            return []

        res_items = yolo_result.split('|')
        res_items_float: 'List[List[float]]' = [[float(y) for y in x.split(',')] for x in res_items]
        res_items_end = list(filter(lambda z: z[0] > YOLO_SCORE_THRE, res_items_float))

        if not res_items or len(res_items_end) <= 0:
            return []

        arr = []

        for item in res_items:
            arr.append(
                (int(item[1]), int(item[2]), int(item[3]), int(item[4])))

        return arr

    def detect_object(self):
        """
        task: (video_id, video_path, frame_number, object_of_interest)
        """
        # FIXME to test blocking feature
        _t = time.time()
        task = TaskQueue.get(block=True)
        logging.info(f'Time elapse {time.time() - _t}')

        _start_time = time.time()

        logging.info(f'Task {task}')

        yolo_channel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
        yolo_stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(yolo_channel)

        video_id = task[0]
        video_path = task[1]
        frame_num = task[2]
        object_name = task[3]

        img_name = f'{frame_num}.jpg'
        img_data = self.extract_one_frame(video_path, frame_num)

        logging.debug(f"Sending extracted frame {task[1]} to YOLO")
        detected_objects = yolo_stub.DetFrame(
            det_yolov3_pb2.DetFrameRequest(data=img_data,
                                           name=img_name,
                                           cls=object_name))

        det_res = detected_objects.res
        boxes = self.get_bounding_boxes(det_res)
        logging.debug(f"bounding box of {object_name}: {boxes}")

        session = db_session()
        session.begin()
        try:
            v = session.query(Video).filter(Video.id == video_id).one()
            if boxes:
                session.query(Frame).filter(Frame.name == str(frame_num)).update({'processing_status': Status.Finished})
                for b in boxes:
                    _ele = Element(object_name,
                                   Element.coordinate_iterable_to_str(b),
                                   _frame.frame_id, _frame)
                    session.add(_ele)

                session.commit()

                ImageQueue.put((img_name, img_data, boxes))
        except Exception as err:
            logging.error(f'Working on task {task} but encounter error')
            logging.error(err)
            session.rollback()
        finally:
            session.remove()

        yolo_channel.close()

        logging.info(f'Take {time.time() - _start_time} m second to finish')


class DivaGRPCServer(server_diva_pb2_grpc.server_divaServicer):
    """
    Implement server_divaServicer of gRPC
    """
    def request_frame_path(self, request, context):
        # FIXME should use name to find corresponding folder
        # desired_object_name = request.name
        return server_diva_pb2.directory(
            directory_path=FAKE_IMAGE_DIRECTOR_PATH)

    def detect_object_in_video(self, request, context):
        object_name = request.object_name
        video_name = request.video_name

        session = db_session()

        session.begin()
        try:
            selected_video = session.query(Video).filter(
                Video.name == video_name).one()
            logging.debug(
                f'finding video: {video_name} result: {selected_video}')

            if selected_video:
                frame_ids = FrameProcessor.extract_frame_nums(
                    selected_video.path)
                logging.debug(f"adding {len(frame_ids)} tasks in queue")

                _frame_list = []

                for f_id in frame_ids:
                    _frame_list.append(
                        Frame(str(f_id), selected_video.id, selected_video,
                              Status.Initialized))
                    TaskQueue.put((selected_video.id, selected_video.path,
                                   f_id, object_name))

                session.bulk_save_objects(_frame_list)

            else:
                logging.warning(f'Failed to find video with name {video_name}')

        except Exception as err:
            logging.error(err)
            session.rollback()
        finally:
            session.remove()

        # FIXME should return nothing
        path_arr = []
        return server_diva_pb2.image_paths(path=path_arr)


def grpc_serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_servicer = DivaGRPCServer()
    server_diva_pb2_grpc.add_server_divaServicer_to_server(
        diva_servicer, server)
    server.add_insecure_port(f'[::]:{DIVA_CHANNEL_PORT}')
    server.start()
    logging.info("GRPC server is runing")

    return server


def detection_serve():
    thread_list = []

    for _ in range(FRAME_PROCESSOR_WORKER_NUM):
        temp = FrameProcessor()
        temp.setDaemon(True)
        temp.start()
        thread_list.append(temp)

    for _ in range(IMAGE_PROCESSOR_WORKER_NUM):
        image_worker = ImageProcessor()
        image_worker.setDaemon(True)
        image_worker.start()
        thread_list.append(image_worker)

    logging.info("workers are runing")

    # NOTE should not join since we don't want the program got blocked here
    # for _t in thread_list:
    #     _t.join()


def draw_box(img, x1, y1, x2, y2):
    rw = float(img.shape[1]) / DET_SIZE
    rh = float(img.shape[0]) / DET_SIZE
    x1, x2 = int(x1 * rw), int(x2 * rw)
    y1, y2 = int(y1 * rh), int(y2 * rh)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def deploy_operator_on_camera(operator_path: str,
                              camStub: cam_cloud_pb2_grpc.DivaCameraStub):
    f = open(operator_path, 'rb')
    op_data = f.read(CHUNK_SIZE)
    while op_data != b"":
        response = camStub.DeployOp(cam_cloud_pb2.Chunk(data=op_data))
        if response.msg != 'OK':
            logging.warning('DIVA deploy op fails!!!')
            return
        op_data = f.read(CHUNK_SIZE)
    f.close()
    camStub.DeployOpNotify(
        cam_cloud_pb2.DeployOpRequest(name='random', crop=CROP_SPEC))


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


def testYOLO():
    with grpc.insecure_channel(YOLO_CHANNEL_ADDRESS) as channel:
        stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)
        with open('tensorflow-yolov3/data/demo_data/dog.jpg', 'rb') as f:
            sample_data = f.read()
        response = stub.DetFrame(
            det_yolov3_pb2.DetFrameRequest(data=sample_data,
                                           name='dog.jpg',
                                           cls='dog'))
        print('client received: ' + response.res)
        # client received: 0.9983274,68.61318969726562,156.36236572265625,289.6726379394531,650.7457885742188


if __name__ == '__main__':

    logging.getLogger('sqlalchemy').setLevel(logging.INFO)

    init_db()

    logging.info("Init threads")

    detection_serve()

    _server = grpc_serve()
    _server.wait_for_termination()

    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logging.info('cloud receiver stops!!!')
        SHUTDOWN_SIGNAL.set()
        _server.stop(0)
    except Exception as err:
        logging.warning(err)
        SHUTDOWN_SIGNAL.set()
        _server.stop(1)

    # runDiva()
