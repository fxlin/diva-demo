'''
test written by Mengwei, cherry picked from br:echo.

run it on camera:

1. changes the paths so that they point to actual ops/weight files
2. run: python -m tests.test_op_load

test results:
Rpi3
>>>>>>> old load. 30.185352325439453 sec
>>> init 7.794313192367554 sec
>>>>>>> new load. 10.7500741481781 sec
>>>>>>> old load. 29.80506992340088 sec
>>> init 8.417441129684448 sec
>>>>>>> new load. 11.316202402114868 sec
>>>>>>> old load. 29.73972225189209 sec
>>> init 7.526222467422485 sec
>>>>>>> new load. 10.394251346588135 sec
>>>>>>> old load. 30.050033569335938 sec
>>> init 7.630579471588135 sec
>>>>>>> new load. 10.547085046768188 sec
>>>>>>> old load. 30.104559183120728 sec
>>> init 7.52424430847168 sec
>>>>>>> new load. 10.385059356689453 sec
>>>>>>> predict speed: 0.08198363184928895 sec per image


Rpi4
>>>>>>> old load. 8.808716058731079 sec
>>> init 2.4105212688446045 sec
>>>>>>> new load. 3.2289435863494873 sec
>>>>>>> old load. 8.260964632034302 sec
>>> init 2.5707414150238037 sec
>>>>>>> new load. 3.3534374237060547 sec
>>>>>>> old load. 8.531449794769287 sec
>>> init 2.180997848510742 sec
>>>>>>> new load. 2.9619052410125732 sec
>>>>>>> old load. 8.558156490325928 sec
>>> init 2.4492411613464355 sec
>>>>>>> new load. 3.236464738845825 sec
>>>>>>> old load. 8.185439586639404 sec
>>> init 2.583303689956665 sec
>>>>>>> new load. 3.368257999420166 sec
>>>>>>> predict speed: 0.025540335476398467 sec per image

'''

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
