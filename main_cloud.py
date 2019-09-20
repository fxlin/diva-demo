import os
import grpc
import time
import cv2
import numpy as np

import det_yolov3_pb2
import det_yolov3_pb2_grpc
import cam_cloud_pb2
import cam_cloud_pb2_grpc

from util import *

CHUNK_SIZE = 1024 * 100
cls = 'bicycle'
crop = '350,0,720,400'
yolo_score_thre = 0.4
img_path = '/media/teddyxu/WD-4TB/hybridvs_data/YOLO-RES-720P/jpg/chaweng-1_10FPS/'
csv_path = '/media/teddyxu/WD-4TB/hybridvs_data/YOLO-RES-720P/out/chaweng-1_10FPS.csv'
op_fname = '/media/teddyxu/WD-4TB/hybridvs_data/YOLO-RES-720P/exp/chaweng/models/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'
det_sz = 608



def draw_box(img, x1, y1, x2, y2):
    rw = float(img.shape[1]) / det_sz
    rh = float(img.shape[0]) / det_sz
    x1, x2 = int(x1 * rw), int(x2 * rw)
    y1, y2 = int(y1 * rh), int(y2 * rh)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def runDiva():
    camChannel =  grpc.insecure_channel('localhost:10086')
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel) 
    yoloChannel =  grpc.insecure_channel('localhost:10088')
    yoloStub = det_yolov3_pb2_grpc.DetYOLOv3Stub(yoloChannel)

    response = camStub.InitDiva(cam_cloud_pb2.InitDivaRequest(
        img_path=img_path))
    if response.msg != 'OK':
        print ('DIVA init fails!!!')
        return

    f = open(op_fname, 'rb')
    op_data = f.read(CHUNK_SIZE)
    while op_data != b"":
        response = camStub.DeployOp(cam_cloud_pb2.Chunk(data=op_data))
        if response.msg != 'OK':
            print ('DIVA deploy op fails!!!')
            return
        op_data = f.read(CHUNK_SIZE)
    f.close()
    camStub.DeployOpNotify(cam_cloud_pb2.DeployOpRequest(
        name='random', crop=crop))

    time.sleep(5)
    pos_num = 0
    clog = ClockLog(5)
    for i in range(10000):
        time.sleep(0.1) # emulate network
        clog.print('Retrieved pos num: %d' % (pos_num))
        response = camStub.GetFrame(cam_cloud_pb2.GetFrameRequest(name='echo'))
        img_name = response.name
        if img_name == '':
            print ('no frame returned...')
            continue
        img_data = response.data
        response = yoloStub.DetFrame(det_yolov3_pb2.DetFrameRequest(
            data = img_data, name = str(img_name), cls = cls))
        det_res = response.res
        # print ('client received: ' + det_res)
        if len(det_res) == 0:
            continue
        res_items = det_res.split('|')
        res_items = [[float(y) for y in x.split(',')] for x in res_items]
        res_items = [x for x in res_items if x[0] > yolo_score_thre]
        if len(res_items) > 0:
            img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
            for item in res_items:
                x1, y1, x2, y2 = int(item[1]), int(item[2]), int(item[3]), int(item[4])
                img = draw_box(img, x1, y1, x2, y2)
            img_fname = os.path.join('result/retrieval_imgs/', img_name)
            cv2.imwrite(img_fname, img)
        pos_num += 1
        

    camChannel.close()
    yoloChannel.close()

def testYOLO():
    with grpc.insecure_channel('localhost:10088') as channel:
        stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)
        with open('tensorflow-yolov3/data/demo_data/dog.jpg', 'rb') as f:
            sample_data = f.read()
        response = stub.DetFrame(det_yolov3_pb2.DetFrameRequest(
            data = sample_data, name = 'dog.jpg', cls = 'dog'))
        print ('client received: ' + response.res)

if __name__ == '__main__':
    runDiva()
