import os
import tensorflow.compat.v1 as tf
import numpy as np
import heapq
import cv2
#  FIXME:  need to figure out the optimal way of choosing a neural network. For now this will be harcoded.
OP_FNAME_PATH = '/data/YOLO-RES-720P/exp/chaweng/models/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'

#  FIXME: directory for images that are currently used for testing the neural network
IMAGE_DIR = '/config/Testing_Images'


class OP():

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.keras.backend.set_session(tf.Session(config=config))

    def __init__(self, op_fname):
        self.run_imgs = list()
        self.crop = None
        self.op = tf.keras.models.load_model(op_fname)
        self.frames = list()
        self.in_h, self.in_w = self.op.layers[0].input_shape[1:3]

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

    def prepare(self, img_dir, img_name, crop):
        self.run_imgs = list()
        self.run_imgs.append(os.path.join(img_dir, img_name))
        self.crop = [int(x) for x in crop.split(',')]
        in_h, in_w = self.op.layers[0].input_shape[1:3]
        self.frames = self.read_images(self.run_imgs, self.in_h, self.in_w,
                                       self.crop)

    def run(self):
        output = list()
        scores = self.op.predict(self.frames)
        heapq.heappush(output, (scores[0, 1], self.run_imgs[0].split('/')[-1]))
        return output