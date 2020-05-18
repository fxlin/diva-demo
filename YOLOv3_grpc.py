#! /usr/bin/env python
# coding=utf-8

''' xzl: the main func of the yolov3 service 

it appears VIP students rewrote the way keras/yolov3 are invoked, therefore
some of Mengwei's code no longer works (e.g. scaling images, etc)
'''

import time
from concurrent import futures
import threading
from typing import List
import gc
import cv2
import logging
import grpc
import sys
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
logger = logging.getLogger(__name__)

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

# xzl: the list of all class names
CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)


def filter_bbox(bboxes: 'List', target_class: List[str],
                classes=CLASSES) -> str:
    """
    filter out element with undesired labels
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format
    coordinates.
    """
    filtered_bbox = []
    target_set = set(target_class)

    for _, bbox in enumerate(bboxes):
        class_ind = int(bbox[5])

        if target_class and classes[class_ind] not in target_set:
            continue

        filtered_bbox.append(bbox)

    return filtered_bbox


def _filter_bbox(bboxes: 'List', target_class: List[str],
                 classes=CLASSES) -> str:
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format
    coordinates.
    """

    filtered_bbox = []
    target_set = set(target_class)

    for _, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])

        if target_class and classes[class_ind] not in target_set:
            continue

        # FIXME should return number or string?
        # ex: (coor[0], coor[1], coor[2], coor[3])
        coor_str = ','.join(
            str(number) for number in ([score] + coor.tolist()))

        filtered_bbox.append(coor_str)

    return '|'.join(filtered_bbox)


def bboxes_to_grpc_elements(bboxes: List[List[float]]):
    """
    List[det_yolov3_pb2.Element]
    """
    ele_list = []
    for bbox in bboxes:

        coor = np.array(bbox[:4], dtype=np.int32)
        class_ind = int(bbox[5])

        # NOTE if we decide to use different class set, make sure
        # we update classess accordingly
        data = {
            'confidence': bbox[4],
            'class_name': CLASSES[class_ind],
            'x1': coor[0],
            'y1': coor[1],
            'x2': coor[2],
            'y2': coor[3]
        }

        ele_list.append(det_yolov3_pb2.Element(**data))

    return ele_list


# xzl: the core object detector. will return a list of bboxes
def _detect(image_data: np.ndarray, score_threshold) -> 'List[List[float]]':
    """
    List all elements detected in the given images
    """
    original_image_size = image_data.shape[:2]

    image_data = utils.image_preporcess(np.copy(image_data),
                                        [IMAGE_H, IMAGE_W])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    prev_time = time.time()
    pred_bbox = model.predict(image_data)
    curr_time = time.time()
    exec_time = curr_time - prev_time

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    # FIXME what's the correct value of intput size if the shape of the image
    # is not square
    # input_size
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, IMAGE_H,
                                     score_threshold)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    logger.info("YOLOv3-det time: %.2f ms" % (1000 * exec_time))

    return bboxes


GC_TIMER = time.time()

def gc_execution():
    global GC_TIMER
    if (time.time() - GC_TIMER) >= 60.0 * 5:
        gc.collect()
        GC_TIMER = time.time()


class DetYOLOv3Servicer(det_yolov3_pb2_grpc.DetYOLOv3Servicer):
    # xzl: old one, return a score
    def DetFrame(self, request, context):
        """
        Search one specific object in the given image
        """
        img_data = request.image
        object_class = request.cls
        name = request.name

        logger.info(f'Processing {name} with target amid at {object_class}')

        # img = Image.frombytes('RGBA', (720, 1280),
        # img_data, decoder_name='jpeg', 'jpg')
        # TODO: resolution shall not be hard-coded

        img = np.frombuffer(img_data.data, dtype=np.uint8).reshape(
            (img_data.height, img_data.width, img_data.channel))

        bboxes = _detect(img, 0.3)
        det_res = _filter_bbox(bboxes=bboxes, target_class=[object_class])

        return det_yolov3_pb2.Score(res=det_res)

    # xzl: new interface done by vip project, return a list of bbs
    # expect img_data to be raw pixel arrays
    def DetectVIP(self, request, context):
        """
        Search all or certain objects in the given image.
        If targets is empty, keep all elements. Otherwise, only keep the target
        class
        """
        img_data = request.image
        threshold = request.threshold
        name = request.name
        targets = request.targets

        msg = f'Processing {name}, expecting to have score above {threshold}'
        logger.info(msg)

        # img = Image.frombytes('RGBA', (720, 1280),
        # img_data, decoder_name='jpeg', 'jpg')
        # TODO: resolution shall not be hard-coded

        img = np.frombuffer(img_data.data, dtype=np.uint8).reshape(
            (img_data.height, img_data.width, img_data.channel))

        bboxes = _detect(img, threshold)

        if targets:
            bboxes = filter_bbox(bboxes=bboxes, target_class=targets)

        element_list = bboxes_to_grpc_elements(bboxes)

        return det_yolov3_pb2.DetectionOutput(elements=element_list)

    def Detect(self, request, context):
        """
        Search all or certain objects in the given image.
        If targets is empty, keep all elements. Otherwise, only keep
        """
        img_data = request.image.data
        # threshold = request.threshold # TODO
        threshold = 0.3
        name = request.name
        targets = request.targets

        msg = f'Processing {name}, expecting to have score above {threshold}'
        logger.info(msg)

        raw_img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
        
        #raw_img = cv2.resize(raw_img, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_NEAREST)
        #raw_img = raw_img / 255.0
        # img = Image.frombytes('RGBA', (720, 1280),
        # img_data, decoder_name='jpeg', 'jpg')
        # TODO: resolution shall not be hard-coded

        bboxes = _detect(raw_img, threshold)
        
        print("bboxes", bboxes)

        # xzl: filter detection res with target class
        if targets:
            bboxes = filter_bbox(bboxes=bboxes, target_class=targets)

        element_list = bboxes_to_grpc_elements(bboxes)

        return det_yolov3_pb2.DetectionOutput(elements=element_list)
    
def test_draw_bb(raw_img, bboxes):
    for bb in bboxes:
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        confidence = bb[4]
        classid = bb[5]
        print(f"class {CLASSES[classid]}, confidence {confidence}")
        # nb: only accept int coordinates
        raw_img = cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite("/data/diva-fork/output.jpg", raw_img)
        
def test1_imread():
    # f = open('/data/hybridvs-demo/tensorflow-yolov3/data/demo_data/dog.jpg', 'rb')
    raw_img = cv2.imread('/data/hybridvs-demo/tensorflow-yolov3/data/demo_data/dog.jpg')
    bboxes = _detect(raw_img, 0.3)
    print("bboxes", bboxes)
        
def test2_imdecode():
    f = open('/data/hybridvs-demo/tensorflow-yolov3/data/demo_data/dog.jpg', 'rb')
    img_data = f.read()
    raw_img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
    # can't do the following. otherwise won't work....
    #raw_img = cv2.resize(raw_img, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_NEAREST)
    #raw_img = raw_img / 255.0    
    bboxes = _detect(raw_img, 0.3)
    print("bboxes", bboxes)    
    
    test_draw_bb(raw_img, bboxes)
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    det_yolov3_pb2_grpc.add_DetYOLOv3Servicer_to_server(
        DetYOLOv3Servicer(), server)
    server.add_insecure_port(f'[::]:{YOLO_CHANNEL_PORT}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logger.warn('YOLOv3-det receiver stops!!!')


if __name__ == '__main__':
    logger.info('initializing yolo service')
    serve()
    #test1_imread()
    #test2_imdecode()
