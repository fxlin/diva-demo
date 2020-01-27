"""
Ingest video frames and perform object detection on frames.
"""

import os, sys
import traceback
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
# FIXME
from google.protobuf import empty_pb2

from util import ClockLog

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS, IMAGE_PATH, OP_FNAME_PATH
from variables import FAKE_IMAGE_DIRECTOR_PATH, DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT
from variables import RESULT_IMAGE_PATH

from constants.grpc_constant import INIT_DIVA_SUCCESS

from sqlalchemy.orm.exc import MultipleResultsFound
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

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


class ImageProcessor(threading.Thread):
    def run(self):
        while not SHUTDOWN_SIGNAL.is_set():
            while not ImageQueue.empty():
                logger.info('working on the given image')
                self.consume_image_task()

    def consume_image_task(self):
        task = ImageQueue.get(block=True)
        if not task or len(task) == 0:
            return

        image_name = task[0]
        image_data = task[1]
        bounding_boxes = task[2]
        self.process_frame(image_name, image_data, bounding_boxes)

    @staticmethod
    def process_frame(img_name: str, img_data,
                      res_items: List[Tuple[int, int, int, int]]):
        """
        Take two arguments: one is the image frame data and the other is the
        response from YOLO agent. Draw boxes around the object of interest.
        """
        if not res_items:
            return

        img = cv2.cvtColor(np.asarray(img_data), cv2.COLOR_RGB2BGR)
        for item in res_items:
            x1, y1, x2, y2 = item[0], item[1], item[2], item[3]
            img = draw_box(img, x1, y1, x2, y2)

        # FIXME test
        img_fname = img_name
        # img_fname = os.path.join(RESULT_IMAGE_PATH, img_name)

        cv2.imwrite(img_fname, img)

        del img_data
        del img


class FrameProcessor(threading.Thread):
    def run(self):
        while not SHUTDOWN_SIGNAL.is_set():
            while not TaskQueue.empty():
                logger.info("Got a task to do")
                self.detect_object()

    @staticmethod
    def video_info(video_path: str) -> dict:
        if not os.path.exists(video_path):
            raise ValueError(f"path {video_path} does not exist")

        probe = ffmpeg.probe(video_path)

        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise ValueError(f'{video_path} is invalid: No video stream found')

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
    def get_bounding_boxes(yolo_result: str
                           ) -> 'List[Tuple[int, int, int, int]]':
        if not yolo_result or len(yolo_result) == 0:
            return []

        res_items = yolo_result.split('|')
        res_item_coordinates = map(lambda x: x.split(','), res_items)
        res_items_end = list(
            filter(lambda z: float(z[0]) > YOLO_SCORE_THRE,
                   res_item_coordinates))

        del res_items
        del res_item_coordinates

        if not res_items_end:
            return []

        return [tuple(int(b) for b in a[1:]) for a in res_items_end]

    def detect_object(self):
        """
        task: (video_id, video_path, frame_number, object_of_interest)
        """
        # FIXME to test blocking feature
        _t = time.time()
        task = TaskQueue.get(block=True)
        logger.info(f'Time elapse {time.time() - _t}')

        _start_time = time.time()

        logger.info(f'Task {task}')

        yolo_channel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
        yolo_stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(yolo_channel)

        video_id, video_path, = task[0], task[1]
        frame_num, object_name = task[2], task[3]

        session = db_session()
        picked_frame = None

        # FIXME
        logger.info(
            f'before query: {session.query(Frame).filter(Frame.name == str(frame_num)).filter(Frame.processing_status == Status.Initialized).all()}'
        )

        try:
            picked_frame = session.query(Frame).filter(
                Frame.name == str(frame_num)).filter(
                    Frame.processing_status ==
                    Status.Initialized).one_or_none()
        except MultipleResultsFound as m_err:
            # FIXME
            logger.info(
                f'after query: {session.query(Frame).filter(Frame.name == str(frame_num)).filter(Frame.processing_status == Status.Initialized).all()}'
            )

            logger.error(
                f'Found too many frames with name {frame_num}: {m_err}')
        except Exception as err:
            logger.error(err)

        if picked_frame:
            try:
                picked_frame.processing_status = Status.Processing
                session.commit()
            except Exception as err:
                logger.error(err)

            img_name = f'{frame_num}.jpg'
            img_data = self.extract_one_frame(video_path, frame_num)

            logger.debug(f"Sending extracted frame {task[1]} to YOLO")
            detected_objects = yolo_stub.DetFrame(
                det_yolov3_pb2.DetFrameRequest(data=img_data,
                                               name=img_name,
                                               cls=object_name))

            boxes = self.get_bounding_boxes(detected_objects.res)
            logger.debug(f"bounding box of {object_name}: {boxes}")

            try:
                picked_frame.processing_status = Status.Finished
                session.commit()

                if boxes:
                    temp_b = map(
                        lambda x: Element.coordinate_iterable_to_str(x), boxes)
                    ele_list = list(
                        map(
                            lambda y: Element(object_name, y, _frame.frame_id,
                                              _frame), temp_b))
                    session.bulk_save_objects(ele_list)
                    session.commit()

                    ImageQueue.put((img_name, img_data, boxes))
            except Exception as err:
                logger.error(
                    f'Working on task {task} but encounter error: {err}')
                session.rollback()

        db_session.remove()

        yolo_channel.close()

        logger.info(f'Take {time.time() - _start_time} m second to finish')


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

        # FIXME
        logger.error(f'Moterh fuck, they call me {time.time()}')

        object_name = request.object_name
        video_name = request.video_name

        session = db_session()
        try:
            selected_video = session.query(Video).filter(
                Video.name == video_name).one_or_none()
            logger.debug(
                f'finding video: {video_name} result: {selected_video}')

            if selected_video:
                frame_ids = FrameProcessor.extract_frame_nums(
                    selected_video.path)

                _frame_list = [
                    Frame(str(f_id), selected_video.id, selected_video,
                          Status.Initialized) for f_id in frame_ids
                ]

                # FIXME any frames in db???
                temp_res = session.query(Frame).join(Video).filter(
                    Video.name == video_name).all()
                logger.info(f'frames in db: num: {len(temp_res)} {temp_res}')
                gg_set = set([ccc.name for ccc in temp_res])
                zz_set = set([xxx.name for xxx in _frame_list])
                logger.warning(
                    f'Any duplicate between {gg_set.union(zz_set)}?')
                if gg_set.union(zz_set):
                    logger.warning(f'what is this thread???')
                    raise Exception(f'fuck!!!!!!!!!!! why?')

                session.bulk_save_objects(_frame_list)
                session.commit()

                # FIXME
                temp_res_another = session.query(Frame).join(Video).filter(
                    Video.name == video_name).all()
                logger.info(
                    f'frames in db: num: {len(temp_res_another)} = {len(temp_res)} + {len(frame_ids)}'
                )

                for f_id in frame_ids:
                    TaskQueue.put((selected_video.id, selected_video.path,
                                   f_id, object_name))
            else:
                logger.warning(f'Failed to find video with name {video_name}')

        except MultipleResultsFound as m_err:
            logger.error(
                f'Found multiple result when finding video with name {video_name}: {m_err}'
            )
        except Exception as err:
            logger.error(f"Failed to insert frame data into DB: {err}")

            session.rollback()
        finally:
            db_session.remove()

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

    logger.info("workers are runing")

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
            logger.warning('DIVA deploy op fails!!!')
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


if __name__ == '__main__':

    logging.getLogger('sqlalchemy').setLevel(logging.INFO)

    init_db()

    logger.info("Init threads")

    detection_serve()

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
