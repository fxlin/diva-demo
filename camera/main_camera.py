import time
import os
import sys
import json
import cv2
from threading import Thread, Event, Lock
from concurrent import futures
import logging
# from queue import Queue
# import heapq

import grpc
# import numpy as np
# import keras
# import keras.backend as K
# import tensorflow as tf

import cam_cloud_pb2_grpc
import server_diva_pb2
import server_diva_pb2_grpc
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import common_pb2
from google.protobuf import empty_pb2

from variables import CAMERA_CHANNEL_PORT
from variables import DIVA_CHANNEL_ADDRESS

from camera.camera_constants import VIDEO_FOLDER, _HOST, _PORT, STATIC_FOLDER, WEB_APP_DNS
from camera.camera_constants import _NAME, _ADDRESS, YOLO_CHANNEL_ADDRESS

# from camera.ml import Operator

# from util import *

CHUNK_SIZE = 1024 * 100
OP_BATCH_SZ = 16

# FIXME change serverity level
FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# QuerryQueue = Queue()

# def send_image_to(image: np.ndarray, video_name: str, host_ip: str,
#                   host_port: str, host_name: str, confidence_score: float,
#                   image_timestamp: int, offset: int):
#     camera_info = server_diva_pb2.camera_info(camera_ip=host_ip,
#                                               camera_port=host_port,
#                                               name=host_name)

#     height, width, channel = image.shape

#     image_payload = common_pb2.Image(data=image.tobytes(),
#                                      height=height,
#                                      width=width,
#                                      channel=channel)

#     payload = server_diva_pb2.frame_from_camera(
#         image=image_payload,
#         camera=camera_info,
#         confidence_score=confidence_score,
#         timestamp=image_timestamp,
#         video_name=video_name,
#         offset=offset)

#     controller_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
#     controller_stub = server_diva_pb2_grpc.server_divaStub(controller_channel)
#     # FIXME what to do with the response?
#     _ = controller_stub.detect_object_in_frame(payload)
#     controller_channel.close()

#     # FIXME check the content, maybe trim the video now???

# class OP_WORKER(Thread):
#     def read_images(self, imgs, H, W, crop=(-1, -1, -1, -1)):
#         frames = np.zeros((len(imgs), H, W, 3), dtype='float32')
#         for i, img in enumerate(imgs):
#             frame = cv2.imread(img)
#             if crop[0] > 0:
#                 frame = frame[crop[0]:crop[2], crop[1]:crop[3]]
#             frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_NEAREST)
#             frames[i, :] = frame
#         frames /= 255.0
#         return frames

#     def __init__(self):
#         Thread.__init__(self)
#         self.is_run = False
#         self.run_imgs = []
#         self.batch_size = 16
#         self.kill = False
#         self.op = None
#         self.crop = None
#         self.op_updated = True

#     def prepare(self, img_dir, img_names, buf, op_fname, crop):
#         self.run_imgs = []
#         self.op_fname = op_fname
#         for img in img_names:
#             self.run_imgs.append(os.path.join(img_dir, img))
#         self.buffer = buf
#         self.op_fname = op_fname
#         self.op_updated = True
#         self.is_run = True
#         self.crop = [int(x) for x in crop.split(',')]

#     def reset(self, buffer, op_fname, crop):
#         self.op_fname = op_fname
#         self.buffer = buffer
#         self.op_updated = True
#         self.is_run = True
#         self.crop = [int(x) for x in crop.split(',')]

#         # self.run_imgs = []
#         # for img in img_names:
#         #     self.run_imgs.append(os.path.join(img_dir, img))

#     def isRunning(self):
#         return self.is_run

#     def kill(self):
#         self.kill = True

#     def stop(self):
#         self.is_run = False

#     def preprocess(self, task: 'Tuple') -> :
#         """
#         task = (start_timestamp (int), end_timestamp (int), object)
#         """
#         start_timestamp, end_timestamp, _ = task
#         video_list = os.listdir(VIDEO_FOLDER)
#         temp_1 = map(lambda x: int(x.replace('.mp4', '')), video_list)
#         temp_2 = filter(lambda y: y >= start_timestamp, temp_1)
#         temp_3 = filter(lambda z: z, z < end_timestamp, temp_2)

#         return list(
#             map(lambda w: os.path.join(VIDEO_FOLDER, f'{w}.mp4'), temp_3))

#     def process_video(self, video_path: str, start_second: int,
#                       end_second: int) -> 'List[int]':
#         source = cv2.VideoCapture(video_path)
#         fps = source.get(cv2.CAP_PROP_FPS)
#         # FIXME set specific time

#         while source.isOpened():
#             ret, frame = source.read()
#             if not ret:
#                 break

#             # FIXME should pre-process the frame? reduce size?
#             result = self.op.predict(frame)

#         source.release()

#     @staticmethod
#     def get_video_list(start: int, end: int) -> 'List[str]':
#         pass

#     @staticmethod
#     def notify_controller():
#         """
#         Telling controller that task xxx is done.
#         """
#         pass

#     def run(self):
#         while True:
#             if self.kill:
#                 break
#             if not self.is_run:
#                 time.sleep(1)
#                 continue

#             if self.op_updated:
#                 # ==========================old============================
#                 # FIXME check whether or not the migration is successfully
#                 # config = tf.ConfigProto()
#                 # config.gpu_options.per_process_gpu_memory_fraction = 0.3
#                 # keras.backend.set_session(tf.Session(config=config))
#                 # self.op = keras.models.load_model(self.op_fname)
#                 # ==========================old============================

#                 self.op = tf.keras.load_model(self.op_fname)
#                 in_h, in_w = self.op.layers[0].input_shape[1:3]
#                 self.op_updated = False

#             # ==========================old============================
#             # imgs = random.sample(self.run_imgs, OP_BATCH_SZ)
#             # frames = self.read_images(imgs, in_h, in_w, crop=self.crop)
#             # TODO: IO shall be seperated thread
#             # scores = self.op.predict(frames)
#             # for i in range(OP_BATCH_SZ):
#             #     heapq.heappush(self.buffer,
#             #                    (-scores[i, 1], imgs[i].split('/')[-1]))
#             # ==========================old============================

#             # TODO.1 process videos
#             # TODO.2 rank result
#             # TODO.3 send image to server
#             querry = QuerryQueue.get(block=True)
#             temp_data = self.preprocess(querry)
#             self.process_video(*temp_data)

#             logger.debug('Buf size: ', len(self.buffer))
#             logger.debug('Buf size: %d, total num: %d' %
#                          (len(self.buffer), len(self.run_imgs)))

#             self.notify_controller()

# def trim_video(source_path: str, output_path: str, start_second: int,
#                end_second: int):
#     source_video = cv2.VideoCapture(source_path)

#     if not source_video.isOpened():
#         raise Exception("Video is not opened")

#     target_width = source_video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#     target_height = source_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
#     target_fps = source_video.get(cv2.CAP_PROP_FPS)

#     source_video.set(cv2.CAP_PROP_POS_FRAMES, target_fps * start_second)
#     counter = target_fps * end_second

#     _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_path, _fourcc, target_fps,
#                                    (target_width, target_height))

#     logging.info(f'time {time.time()}')
#     while counter >= 0:
#         ret, frame = source_video.read()
#         if not ret:
#             break
#         output_video.write(frame)
#         counter -= 1
#     logging.info(f'time {time.time()} file {output_path} exists? \
#             {os.path.exists(output_path)}')

#     source_video.release()
#     output_video.release()

# FIXME
OP_FNAME_PATH = '/home/bryanjw01/workspace/test_data/source/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'


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
        video_path = os.path.join(VIDEO_FOLDER, request.video_name)
        target_class = request.object_name

        source = cv2.VideoCapture(video_path)
        counter = 0

        score_obj = {}
        # |counter |index | score: float

        with grpc.insecure_channel(YOLO_CHANNEL_ADDRESS) as channel:
            stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)

            while source.isOpened():
                ret, frame = source.read()
                if not ret:
                    break

                if (counter % 30) == 0:
                    # send image to process

                    # t_start = time.time()
                    # FIXME disable ML prediction
                    # if (self.op.predict_image(frame, '350,0,720,400') <= 0.3):
                    #     counter += 1
                    #     continue

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

                    # temp_score = []
                    temp_score_map = {}

                    # draw bbox on the image
                    for idx, ele in enumerate(resp.elements):
                        if ele.class_name != target_class:
                            continue

                        x1, y1, x2, y2 = ele.x1, ele.y1, ele.x2, ele.y2

                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2),
                                              (0, 255, 0), 3)
                        temp_score_map[str(idx)] = ele.confidence
                        # temp_score.append(ele.confidence)
                        exist_target = True

                    video_folder_name = '.'.join(
                        request.video_name.split('.')[:-1])
                    if exist_target:
                        # FIXME does not handel spcieal cases!!!!

                        img_path = os.path.join(
                            STATIC_FOLDER, *[
                                video_folder_name, target_class, 'images',
                                f'{counter}.jpg'
                            ])
                        cv2.imwrite(img_path, frame)
                        score_obj[str(counter)] = temp_score_map
                    # t_end = time.time()

                    # for sc in temp_score:
                    #     metric_df = metric_df.append(
                    #         {
                    #             'start_time': t_start,
                    #             'end_time': t_end,
                    #             'diff': t_end - t_start,
                    #             'score': sc,
                    #             'class': target_class
                    #         },
                    #         ignore_index=True)

                counter += 1

        source.release()

        with open(
                os.path.join(STATIC_FOLDER,
                             *[video_folder_name, 'scores.json']),
                'w') as fptr:
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
                    frames=_video.get(cv2.CAP_PROP_FRAME_COUNT)))

            _video.release()

        return common_pb2.get_videos_resp(videos=video_list)

    def get_video(self, request, context):
        video_folder_name = '.'.join(request.name.split('.')[:-1])
        object_name = request.object_name
        images_url = f'{WEB_APP_DNS}/{video_folder_name}/{object_name}/images/'

        score_file_url = f'{WEB_APP_DNS}/{video_folder_name}/scores.json'

        _video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, request.name))
        frames = _video.get(cv2.CAP_PROP_FRAME_COUNT)
        _video.release()
        return common_pb2.video_metadata(name=request.name,
                                         frames=frames,
                                         video_url=request.video_url,
                                         score_file_url=score_file_url,
                                         images_url=images_url,
                                         camera=request.camera,
                                         object_name=object_name)

        # message get_videos_resp {
        #     repeated video_metadata videos = 1;
        # }

        # message video_metadata {
        #     int32 frames = 1;
        #     string score_file_url = 2;
        #     string name = 3;
        #     Camera camera = 4;
        #     string video_url = 5;
        #     string images_url = 6;
        # }

        # message VideoRequest {
        #     int32 timestamp = 1;
        #     int32 offset = 2;
        #     string video_name = 3;
        #     string object_name =4;
        #     Camera camera = 5;
        # }

        # message Camera {
        #     string name = 1;
        #     string address = 2;
        # }

    # # def KillOp(self):
    #     self.op_worker.kill()

    # def GetFrame(self, request, context):
    #     if len(self.send_buf) == 0:
    #         logger.debug('Empty queue, nothing to send')
    #         return cam_cloud_pb2.Frame(name='', data=b'')
    #     data = None
    #     send_img = heapq.heappop(self.send_buf)[1]
    #     with open(os.path.join(self.img_dir, send_img), 'rb') as f:
    #         data = f.read()

    #     return cam_cloud_pb2.Frame(name=send_img.split('/')[-1], data=data)

    # def InitDiva(self, request, context):
    #     self.img_dir = request.img_path
    #     self.send_buf = []
    #     return cam_cloud_pb2.StrMsg(msg=INIT_DIVA_SUCCESS)

    # def initiate_querry(self):
    #     self.locker.acquire()
    #     if self.is_queerying:
    #         return cam_cloud_pb2.StrMsg(msg='NO')

    #     # start op processing
    #     if self.op_worker.isRunning():
    #         self.op_worker.stop()

    #     self.cleaup()
    #     self.is_queerying = True
    #     self.locker.release()

    # def DeployOpNotify(self, request, context):
    #     self.cur_op_name = request.name
    #     op_fname = os.path.join(OP_DIR, self.cur_op_name)

    #     with open(op_fname, 'wb') as f:
    #         f.write(self.cur_op_data)
    #     self.cur_op_data = bytearray(b'')

    #     # all_imgs = os.listdir(self.img_dir)
    #     # selected_imgs = [img for img in all_imgs
    #     #                  if int(img[:-4]) % 10 == 0]  # 1 FPS
    #     # self.op_worker.prepare(self.img_dir, selected_imgs, self.send_buf,
    #     #                        op_fname, request.crop)

    #     return cam_cloud_pb2.StrMsg(msg=INIT_DIVA_SUCCESS)

    # def DeployOp(self, request, context):
    #     self.cur_op_data += request.data
    #     # print ('XXX', len(request.data))
    #     return cam_cloud_pb2.StrMsg(msg='OK')

    # def DownloadVideo(self, request, context):
    #     video_name = request.video_name
    #     offset = request.offset
    #     timestamp = request.timestamp

    #     video_source_p = os.path.join(VIDEO_FOLDER, video_name)
    #     video_output_p = os.path.join(VIDEO_FOLDER, f'{timestamp}.mp4')
    #     video_data = None

    #     logger.info(f'{video_name} offset {offset} timestamp {timestamp}')

    #     if not os.path.exists(video_source_p):
    #         _file = common_pb2.File(data=bytearray(0))
    #         return cam_cloud_pb2.VideoResponse(
    #             msg="FAILED",
    #             status_code=grpc.StatusCode.FAILED_PRECONDITION,
    #             video=_file)

    #     try:
    #         # trim video and make a temp file and send the video clip back
    #         trim_video(video_name, video_output_p, offset,
    #                    offset + VIDEO_CLIP_LENGTH)

    #         logger.debug(
    #             f'trim video for timestamp: {timestamp} - {timestamp+5}')
    #         with open(video_output_p, 'r') as fptr:
    #             video_data = fptr.read().encode()
    #     except Exception as err:
    #         _file = common_pb2.File(data=bytearray(0))
    #         logger.warning(f'Error happened when transmiting video {err}')
    #         return cam_cloud_pb2.VideoResponse(
    #             msg=f"{err}",
    #             status_code=grpc.StatusCode.FAILED_PRECONDITION,
    #             video=_file)
    #     finally:
    #         if os.path.exists(video_output_p):
    #             logger.debug(f'remove video {video_output_p}')
    #             os.remove(video_output_p)

    #     logger.info(f'Reponsing the request')
    #     _file = common_pb2.File(data=video_data)
    #     return cam_cloud_pb2.VideoResponse(msg="OK",
    #                                        status_code=grpc.StatusCode.OK,
    #                                        video=_file)


# class CameraController(Thread):
#     save_video_event = Event()

#     def run(self):
#         cap = cv2.VideoCapture(0)

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         # FIXME (640, 480) naming???
#         curr_time = int(time.time())
#         curr_filename = f'{curr_time}_.mp4'
#         out = cv2.VideoWriter(curr_filename, fourcc, 30.0, (1280, 720))

#         # FIXME check disk space
#         while (cap.isOpened()):
#             if self.save_video_event.is_set():
#                 out.release()
#                 new_file_name = f'{curr_time}_{int(time.time())}.mp4'
#                 os.rename(curr_filename, new_file_name)

#                 curr_time = int(time.time())
#                 curr_filename = f'{curr_time}_.mp4'
#                 out = cv2.VideoWriter(curr_filename, fourcc, 30.0, (1280, 720))
#                 self.save_video_event.clear()

#             ret, frame = cap.read()
#             if ret:
#                 out.write(frame)

#             else:
#                 break

#         cap.release()
#         out.release()


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
        #diva_cam_servicer.KillOp()
        server.stop(0)


if __name__ == '__main__':
    serve()
