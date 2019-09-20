#! /usr/bin/env python
# coding=utf-8

import os
import time
from concurrent import futures
import logging
import grpc
import sys
import cv2
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import time
from PIL import Image
import numpy as np
import tensorflow as tf
sys.path.insert(0, './tensorflow-yolov3/core')
import utils
import keras

IMAGE_H, IMAGE_W = 608, 608
classes = utils.read_coco_names('./tensorflow-yolov3/data/coco.names')
num_classes = len(classes)
gpu_nms_graph = tf.Graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
keras.backend.set_session(tf.Session(config=config))

input_tensor, output_tensors = utils.read_pb_return_tensors(
        gpu_nms_graph,
        "./tensorflow-yolov3/checkpoint/yolov3_gpu_nms.pb",
        ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
sess = tf.Session(graph=gpu_nms_graph)
def run_det(img, cls):
    start = time.time()
    boxes, scores, labels = sess.run(
            output_tensors, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
    ret = []
    for i in range(len(boxes)):
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        if label != cls: continue
        res_str = ','.join([str(x) for x in ([score] + bbox.tolist())])
        print ('Detect res: ', res_str)
        ret.append(res_str)
    print("YOLOv3-det time: %.2f ms" % (1000*(time.time()-start)))
    return '|'.join(ret)

class DetYOLOv3Servicer(det_yolov3_pb2_grpc.DetYOLOv3Servicer):
    def DetFrame(self, request, context):
        img_data = request.data
        cls = request.cls
        name = request.name
        # img = Image.frombytes('RGBA', (720, 1280), img_data, decoder_name='jpeg', 'jpg') #TODO: resolution shall not be hard-coded
        img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
        img = cv2.resize(img, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        det_res = run_det(img, cls)
        return det_yolov3_pb2.Score(res=det_res)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    det_yolov3_pb2_grpc.add_DetYOLOv3Servicer_to_server(
            DetYOLOv3Servicer(), server)
    server.add_insecure_port('[::]:10088')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        print ('YOLOv3-det receiver stops!!!')
        server.stop(0)

if __name__ == '__main__':
    serve()
