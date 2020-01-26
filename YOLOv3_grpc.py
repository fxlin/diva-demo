#! /usr/bin/env python
# coding=utf-8

import os
import time
from concurrent import futures
import logging
import grpc
import sys
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from variables import YOLO_CHANNEL_PORT
import det_yolov3_pb2
import det_yolov3_pb2_grpc

# FIXME workround
# sys.path.insert(
#     0, './third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core')

# input_size   = 416
# image_path   = "./docs/kite.jpg"

from tensorflow_yolov3_config import cfg
import tensorflow_yolov3_utils as utils
from tensorflow_yolov3 import YOLOv3, decode

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger =  logging.getLogger(__name__)

IMAGE_H, IMAGE_W = 608, 608
# classes = util.read_coco_names('./tensorflow-yolov3/data/coco.names')
# num_classes = len(classes)
# gpu_nms_graph = tf.Graph()

input_layer = tf.keras.layers.Input([IMAGE_H, IMAGE_W, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")

CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)


def filter_bbox(bboxes: 'List',
                target_class: str,
                classes=CLASSES,
                threshold=0.3) -> str:
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    filtered_bbox = []

    for _, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])

        if classes[class_ind] != target_class or score < threshold:
            continue

        # FIXME should return number or string? ex: (coor[0], coor[1], coor[2], coor[3])
        coor_str = ','.join(
            str(number) for number in ([score] + coor.tolist()))

        filtered_bbox.append(coor_str)

    return '|'.join(filtered_bbox)


def run_det(image_data: np.ndarray, target_class: str) -> str:
    start = time.time()

    original_image_size = image_data.shape[:2]

    image_data = utils.image_preporcess(image_data, [IMAGE_H, IMAGE_W])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    # FIXME what's the correct value of intput size if the shape of the image
    # is not square
    # input_size
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, IMAGE_H,
                                     0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    temp = filter_bbox(bboxes=bboxes, target_class=target_class)

    logger.info("YOLOv3-det time: %.2f ms" % (time.time() - start))

    return temp


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# keras.backend.set_session(tf.Session(config=config))

# input_tensor, output_tensors = utils.read_pb_return_tensors(
#     gpu_nms_graph, "./tensorflow-yolov3/checkpoint/yolov3_gpu_nms.pb",
#     ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
# sess = tf.Session(graph=gpu_nms_graph)

# def run_det(img, cls):
#     start = time.time()
#     boxes, scores, labels = sess.run(
#         output_tensors, feed_dict={input_tensor: np.expand_dims(img, axis=0)})
#     ret = []
#     for i in range(len(boxes)):
#         bbox, score, label = boxes[i], scores[i], classes[labels[i]]
#         if label != cls:
#             continue
#         res_str = ','.join([str(x) for x in ([score] + bbox.tolist())])
#         print('Detect res: ', res_str)
#         ret.append(res_str)
#     print("YOLOv3-det time: %.2f ms" % (1000 * (time.time() - start)))
#     return '|'.join(ret)


class DetYOLOv3Servicer(det_yolov3_pb2_grpc.DetYOLOv3Servicer):
    def DetFrame(self, request, context):
        img_data = request.data
        cls = request.cls
        name = request.name
        # img = Image.frombytes('RGBA', (720, 1280), img_data, decoder_name='jpeg', 'jpg')
        # TODO: resolution shall not be hard-coded
        img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
        img = cv2.resize(img, (IMAGE_H, IMAGE_W),
                         interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        det_res = run_det(img, cls)
        return det_yolov3_pb2.Score(res=det_res)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    det_yolov3_pb2_grpc.add_DetYOLOv3Servicer_to_server(
        DetYOLOv3Servicer(), server)
    server.add_insecure_port(f'[::]:{YOLO_CHANNEL_PORT}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        print('YOLOv3-det receiver stops!!!')
        server.stop(0)


if __name__ == '__main__':
    serve()
