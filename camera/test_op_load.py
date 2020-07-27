import time
import os
import hashlib
import json
import pickle
import numpy as np

#from .mengwei_util import *

import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


op_fname = '/home/pi/Documents/diva-data/ops/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'
weights_fname = '/home/pi/Documents/diva-data/weights/chaweng-a3d16c61813043a2711ed3f5a646e4eb.h5'
json_fname = '/home/pi/Documents/diva-data/json/chaweng.json'

def generate_conv_net_base(
        input_shape, nb_classes,
        nb_dense, nb_filters, nb_layers, lr_mult=0.001,
        kernel_size=(3, 3), stride=(1, 1)):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0],
                            padding='same',
                            input_shape=input_shape,
                            # subsample=stride
                            ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    nb_filter_multiplier = 2
    
    for i in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters * nb_filter_multiplier, 3, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        nb_filter_multiplier += 1

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # loss = get_loss()
    # model.compile(loss=loss,
    #               optimizer=get_optimizer(lr_mult),
    #               metrics=computed_metrics)
    return model

def read_cfg(fname):
    with open(fname) as f:
        cfgs = json.load(f)
    for cfg_str in cfgs:
        cfg_str = cfg_str.encode('utf-8')
        cfg_md5 = hashlib.md5(cfg_str).hexdigest()
        if cfg_md5 == op_fname.split('-')[-1][:-5]:
            return cfg_str

def gen_model(cfg_str):
    cfg = json.loads(cfg_str)
    params = [int(t) for t in cfg['model'].split(',')]
    model = generate_conv_net_base(
        (cfg['resol'][0], cfg['resol'][1], 3), 2, *params)
    return model

def load_model_fast(fname):
    cfg_str = read_cfg(json_fname)
    t = time.time()
    op = gen_model(cfg_str)
    print(f">>> init {time.time() - t} sec")
    op.load_weights(weights_fname)
    return op

for i in range(5):
    tf.keras.backend.clear_session()
    t = time.time()
    op = keras.models.load_model(op_fname)
    in_h, in_w = op.layers[0].input_shape[1:3]
    print(f">>>>>>> old load. {time.time() - t} sec")

    tf.keras.backend.clear_session()
    # t = time.time()
    # json_file = open("/tmp/model.json", 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # op = tf.keras.models.model_from_json(loaded_model_json)
    # print(f">>>>>>>>>>>>>>>> load json. {time.time() - t} sec")
    t = time.time()
    op = load_model_fast(op_fname)
    print (f">>>>>>> new load. {time.time() - t} sec")

pred_num = 5
total_sec = 0
batch_sz = 32
for i in range(pred_num):
    t = time.time()
    fake_img = np.random.rand(batch_sz, *op.layers[0].input_shape[1:4])
    op.predict(fake_img)
    total_sec += time.time() - t
print (f">>>>>>> predict speed: {total_sec / pred_num / batch_sz} sec per image")
