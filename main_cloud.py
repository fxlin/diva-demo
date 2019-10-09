"""
Ingest video frames and perform object detection on frames.
"""

import os
from concurrent import futures
import time
import grpc
import cv2
import numpy as np

import det_yolov3_pb2
import det_yolov3_pb2_grpc
import cam_cloud_pb2
import cam_cloud_pb2_grpc
import server_diva_pb2_grpc
import server_diva_pb2

from util import ClockLog

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS, IMAGE_PATH, OP_FNAME_PATH
from variables import FAKE_IMAGE_DIRECTOR_PATH, DIVA_CHANNEL_ADDRESS, DIVA_CHANNEL_PORT

from constants.grpc_constant import INIT_DIVA_SUCCESS

CHUNK_SIZE = 1024 * 100
OBJECT_OF_INTEREST = 'bicycle'
CROP_SPEC = '350,0,720,400'
YOLO_SCORE_THRE = 0.4
DET_SIZE = 608


class DivaGRPCServer(server_diva_pb2_grpc.server_divaServicer):
    """
    Implement server_divaServicer of gRPC
    """
    def __init__(self):
        super().__init__(self)

    def request_frame_path(self, request, context):
        # FIXME should use name to find corresponding folder
        # desired_object_name = request.name
        return server_diva_pb2.directory(
            directory_path=FAKE_IMAGE_DIRECTOR_PATH)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_servicer = DivaGRPCServer()
    server_diva_pb2_grpc.add_server_divaServicer_to_server(
        diva_servicer, server)
    server.add_insecure_port(f'[::]:{DIVA_CHANNEL_PORT}')
    server.start()
    server.wait_for_termination()


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
            print('DIVA deploy op fails!!!')
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
    img_fname = os.path.join('result/retrieval_imgs/', img_name)
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
    serve()
    # runDiva()
