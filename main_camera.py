import time
import itertools
import argparse
import numpy as np
import pandas as pd
import sklearn
import random
import sklearn.metrics
import time
import os
import sys
import json
import heapq
import gc
import json
import heapq
import hashlib
import tempfile
import cv2
import math
import pickle
from threading import Thread
from concurrent import futures
import logging
import grpc
import keras
import keras.backend as K
import cam_cloud_pb2
import cam_cloud_pb2_grpc

from util import *

CHUNK_SIZE = 1024 * 100
OP_BATCH_SZ = 16
op_dir = './result/ops'

class OP_WORKER(Thread):
    def read_images(self, imgs, H, W, crop=(-1,-1,-1,-1)):
        frames = np.zeros((len(imgs), H, W, 3), dtype='float32')
        for i, img in enumerate(imgs):
            frame = cv2.imread(img)
            if crop[0] > 0:
                frame = frame[crop[0]:crop[2],crop[1]:crop[3]]
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
        clog = ClockLog(5)
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
            frames = self.read_images(imgs, in_h, in_w, crop=self.crop) #TODO: IO shall be seperated thread
            scores = self.op.predict(frames)
            # print ('Predict scores: ', scores)
            for i in range(OP_BATCH_SZ):
                heapq.heappush(self.buf, (-scores[i, 1], imgs[i].split('/')[-1]))
            # print ('Buf size: ', len(self.buf))
            clog.log('Buf size: %d, total num: %d' % (len(self.buf), len(self.run_imgs)))

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
            print ('Empty queue, nothing to send')
            return cam_cloud_pb2.Frame(name='', data=b'')
        send_img = heapq.heappop(self.send_buf)[1]
        with open(os.path.join(self.img_dir, send_img), 'rb') as f:
            return cam_cloud_pb2.Frame(
                    name=send_img.split('/')[-1],
                    data=f.read())
    def InitDiva(self, request, context):
        self.img_dir = request.img_path
        self.send_buf = []
        return cam_cloud_pb2.StrMsg(msg='OK')
    def DeployOpNotify(self, request, context):
        self.cur_op_name = request.name
        op_fname = os.path.join(op_dir, self.cur_op_name)
        with open(op_fname, 'wb') as f:
            f.write(self.cur_op_data)
        self.cur_op_data = bytearray(b'')
        # start op processing
        if self.op_worker.isRunning():
            self.op_worker.stop()
        all_imgs = os.listdir(self.img_dir)
        selected_imgs = [img for img in all_imgs if int(img[:-4]) % 10 == 0] # 1 FPS
        self.op_worker.prepare(
                self.img_dir, selected_imgs, self.send_buf, op_fname, request.crop)
        return cam_cloud_pb2.StrMsg(msg='OK')
    def DeployOp(self, request, context):
        self.cur_op_data += request.data
        # print ('XXX', len(request.data))
        return cam_cloud_pb2.StrMsg(msg='OK')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_cam_servicer = DivaCameraServicer()
    cam_cloud_pb2_grpc.add_DivaCameraServicer_to_server(
            diva_cam_servicer, server)
    server.add_insecure_port('[::]:10086')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        print ('DivaCloud stop!!!')
        diva_cam_servicer.KillOp()
        server.stop(0)

if __name__ == '__main__':
    serve()
