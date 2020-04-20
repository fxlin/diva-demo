import numpy as np
import tensorflow.compat.v1 as tf
import cv2


class Operator():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.keras.backend.set_session(tf.Session(config=config))

    def __init__(self, op_fname):
        self.run_imgs = list()
        self.crop = None
        self.op = tf.keras.models.load_model(op_fname)
        self.frames = list()
        self.in_h, self.in_w = self.op.layers[0].input_shape[1:3]

    def predict_image(self, frame, crop):
        # self.run_imgs = []
        # self.run_imgs.append(os.path.join(IMAGE_DIR, img_name))
        # frame = cv2.imread(self.run_imgs[0])

        self.crop = [int(x) for x in crop.split(',')]
        in_h, in_w = self.op.layers[0].input_shape[1:3]

        frames = np.zeros((1, in_h, in_w, 3), dtype='float32')
        if self.crop[0] > 0:
            frame = frame[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3]]
        frame = cv2.resize(frame, (in_h, in_w),
                           interpolation=cv2.INTER_NEAREST)
        frames[0, :] = frame

        self.frames = frames / 255.0

        scores = self.op.predict(self.frames)

        return scores[0, 1]
