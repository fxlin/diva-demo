import keras
import keras.backend as K
import tensorflow as tf
import time


op_fname = "../ops/chaweng-0"
#op_fname = "/tmp/chaweng-0"
#op_fname = "./chaweng-0"

t = time.time()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# global tf session XXX TODO: only once
keras.backend.set_session(tf.Session(config=config))
print(f">>>>>>>>>>>>>>>> op worker: keras init done. {time.time() - t} sec")

t = time.time()
op = keras.models.load_model(op_fname)
in_h, in_w = op.layers[0].input_shape[1:3]
print(f">>>>>>>>>>>>>>>> op worker: loaded op {op_fname}. {in_h} x {in_w}. {time.time() - t} sec")

K.clear_session()

t = time.time()
op = keras.models.load_model(op_fname)
in_h, in_w = op.layers[0].input_shape[1:3]
print(f">>>>>>>>>>>>>>>> op worker: loaded op {op_fname}. {in_h} x {in_w}. {time.time() - t} sec")
