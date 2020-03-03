import time
import random
import os
import sys
import heapq
import cv2
from threading import Thread, Event
from concurrent import futures
import logging

import grpc
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf

import cam_cloud_pb2
import cam_cloud_pb2_grpc
import server_diva_pb2
import server_diva_pb2_grpc
import common_pb2

from variables import CAMERA_CHANNEL_PORT, OP_DIR, VIDEO_FOLDER
from variables import DIVA_CHANNEL_ADDRESS
from constants.grpc_constant import INIT_DIVA_SUCCESS
from constants.camera import VIDEO_CLIP_LENGTH

# from util import *

CHUNK_SIZE = 1024 * 100
OP_BATCH_SZ = 16

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# FIXME 1. register camera
# FIXME 2. argument parser -> parse CLI arguments


def exectute_operator():
    # FIXME get images from operator
    # FIXME sort images by confidence score
    # FIXME (all) send_image to server

    pass


def send_image_to(image: np.ndarray, video_name: str, host_ip: str,
                  host_port: str, host_name: str, confidence_score: float,
                  image_timestamp: int, offset: int):
    camera_info = server_diva_pb2.camera_info(camera_ip=host_ip,
                                              camera_port=host_port,
                                              name=host_name)

    height, width, channel = image.shape

    image_payload = common_pb2.Image(data=image.tobytes(),
                                     height=height,
                                     width=width,
                                     channel=channel)

    payload = server_diva_pb2.frame_from_camera(
        image=image_payload,
        camera=camera_info,
        confidence_score=confidence_score,
        timestamp=image_timestamp,
        video_name=video_name,
        offset=offset)

    controller_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
    controller_stub = server_diva_pb2_grpc.server_divaStub(controller_channel)
    # FIXME what to do with the response?
    _ = controller_stub.detect_object_in_frame(payload)
    controller_channel.close()

    # FIXME check the content, maybe trim the video now???


class OP_WORKER(Thread):
    def read_images(self, imgs, H, W, crop=(-1, -1, -1, -1)):
        frames = np.zeros((len(imgs), H, W, 3), dtype='float32')
        for i, img in enumerate(imgs):
            frame = cv2.imread(img)
            if crop[0] > 0:
                frame = frame[crop[0]:crop[2], crop[1]:crop[3]]
            frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_NEAREST)
            frames[i, :] = frame
        frames /= 255.0
        return frames

    def __init__(self):
        Thread.__init__(self)
        self.is_run = False
        self.run_imgs = []
        self.batch_size = 16
        self.kill = False
        self.op = None
        self.crop = None
        self.op_updated = True

    def prepare(self, img_dir, img_names, buf, op_fname, crop):
        self.run_imgs = []
        self.op_fname = op_fname
        for img in img_names:
            self.run_imgs.append(os.path.join(img_dir, img))
        self.buf = buf
        self.op_fname = op_fname
        self.op_updated = True
        self.is_run = True
        self.crop = [int(x) for x in crop.split(',')]

    def isRunning(self):
        return self.is_run

    def kill(self):
        K.clear_session()
        self.kill = True

    def stop(self):
        self.is_run = False

    def run(self):
        # clog = ClockLog(5)
        while True:
            if self.kill:
                break
            if not self.is_run:
                time.sleep(1)
                continue
            if self.op_updated:
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
                keras.backend.set_session(tf.Session(config=config))
                self.op = keras.models.load_model(self.op_fname)
                in_h, in_w = self.op.layers[0].input_shape[1:3]
                self.op_updated = False
            imgs = random.sample(self.run_imgs, OP_BATCH_SZ)
            frames = self.read_images(imgs, in_h, in_w, crop=self.crop)
            # TODO: IO shall be seperated thread
            scores = self.op.predict(frames)
            # print ('Predict scores: ', scores)
            for i in range(OP_BATCH_SZ):
                heapq.heappush(self.buf,
                               (-scores[i, 1], imgs[i].split('/')[-1]))
            # print ('Buf size: ', len(self.buf))
            # clog.log('Buf size: %d, total num: %d' %
            #          (len(self.buf), len(self.run_imgs)))


def trim_video(source_path: str, output_path: str, start_second: int,
               end_second: int):
    source_video = cv2.VideoCapture(source_path)

    if not source_video.isOpened():
        raise Exception("Video is not opened")

    target_width = source_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    target_height = source_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    target_fps = source_video.get(cv2.CAP_PROP_FPS)

    source_video.set(cv2.CAP_PROP_POS_FRAMES, target_fps * start_second)
    counter = target_fps * end_second

    _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, _fourcc, target_fps,
                                   (target_width, target_height))

    logging.info(f'time {time.time()}')
    while counter >= 0:
        frame = source_video.read()
        output_video.write(frame)
        counter -= 1
    logging.info(f'time {time.time()} file {output_path} exists? \
            {os.path.exists(output_path)}')

    source_video.release()
    output_video.release()


class DivaCameraServicer(cam_cloud_pb2_grpc.DivaCameraServicer):
    img_dir = None
    cur_op_data = bytearray(b'')
    cur_op_name = None
    send_buf = []

    def __init__(self):
        cam_cloud_pb2_grpc.DivaCameraServicer.__init__(self)
        self.op_worker = OP_WORKER()
        self.op_worker.start()

    def KillOp(self):
        self.op_worker.kill()

    def GetFrame(self, request, context):
        if len(self.send_buf) == 0:
            print('Empty queue, nothing to send')
            return cam_cloud_pb2.Frame(name='', data=b'')
        send_img = heapq.heappop(self.send_buf)[1]
        with open(os.path.join(self.img_dir, send_img), 'rb') as f:
            return cam_cloud_pb2.Frame(name=send_img.split('/')[-1],
                                       data=f.read())

    def InitDiva(self, request, context):
        self.img_dir = request.img_path
        self.send_buf = []
        return cam_cloud_pb2.StrMsg(msg=INIT_DIVA_SUCCESS)

    def DeployOpNotify(self, request, context):
        self.cur_op_name = request.name
        op_fname = os.path.join(OP_DIR, self.cur_op_name)
        with open(op_fname, 'wb') as f:
            f.write(self.cur_op_data)
        self.cur_op_data = bytearray(b'')
        # start op processing
        if self.op_worker.isRunning():
            self.op_worker.stop()
        all_imgs = os.listdir(self.img_dir)
        selected_imgs = [img for img in all_imgs
                         if int(img[:-4]) % 10 == 0]  # 1 FPS
        self.op_worker.prepare(self.img_dir, selected_imgs, self.send_buf,
                               op_fname, request.crop)
        return cam_cloud_pb2.StrMsg(msg='OK')

    def DeployOp(self, request, context):
        self.cur_op_data += request.data
        # print ('XXX', len(request.data))
        return cam_cloud_pb2.StrMsg(msg='OK')

    def DownloadVideo(self, request, context):
        video_name = request.video_name
        offset = request.offset
        timestamp = request.timestamp

        video_source_p = os.path.join(VIDEO_FOLDER, video_name)
        video_output_p = os.path.join(VIDEO_FOLDER, f'{timestamp}.mp4')
        video_data = None

        if not os.path.exists(video_source_p):
            _file = common_pb2.File(data=bytearray(0))
            return cam_cloud_pb2.VideoResponse(
                msg="FAILED",
                status_code=grpc.StatusCode.FAILED_PRECONDITION,
                video=_file)

        try:
            # trim video and make a temp file and send the video clip back
            trim_video(video_name, video_output_p, offset,
                       offset + VIDEO_CLIP_LENGTH)

            logger.debug(
                f'trim video for timestamp: {timestamp} - {timestamp+5}')
            with open(video_output_p, 'r') as fptr:
                video_data = fptr.read().encode()
        except Exception as err:
            _file = common_pb2.File(data=bytearray(0))
            return cam_cloud_pb2.VideoResponse(
                msg=f"{err}",
                status_code=grpc.StatusCode.FAILED_PRECONDITION,
                video=_file)
        finally:
            if os.path.exists(video_output_p):
                logger.debug(f'remove video {video_output_p}')
                os.remove(video_output_p)

        _file = common_pb2.File(data=video_data)
        return cam_cloud_pb2.VideoResponse(msg="OK",
                                           status_code=grpc.StatusCode.OK,
                                           video=_file)


class CameraController(Thread):
    save_video_event = Event()

    def run(self):
        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # FIXME (640, 480) naming???
        curr_time = int(time.time())
        curr_filename = f'{curr_time}_.mp4'
        out = cv2.VideoWriter(curr_filename, fourcc, 30.0, (1280, 720))

        # FIXME check disk space
        while (cap.isOpened()):
            if self.save_video_event.is_set():
                out.release()
                new_file_name = f'{curr_time}_{int(time.time())}.mp4'
                os.rename(curr_filename, new_file_name)

                curr_time = int(time.time())
                curr_filename = f'{curr_time}_.mp4'
                out = cv2.VideoWriter(curr_filename, fourcc, 30.0, (1280, 720))
                self.save_video_event.clear()

            ret, frame = cap.read()
            if ret:
                out.write(frame)

            else:
                break

        cap.release()
        out.release()


def serve():
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
        print('DivaCloud stop!!!')
        diva_cam_servicer.KillOp()
        server.stop(0)


if __name__ == '__main__':
    serve()
