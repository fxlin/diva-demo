#!/usr/bin/env python3

import tensorflow as tf

# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)


''' sample output

(venv) xl6yq@gpusrv14 (master)[tests]$ ./test-tf-cudnn.py                                                                                                     
2022-04-05 14:28:27.300121: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (
oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA                                                           
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.                                                                   
2022-04-05 14:28:28.064173: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6423 MB 
memory:  -> device: 0, name: Quadro RTX 4000, pci bus id: 0000:60:00.0, compute capability: 7.5                                                               
2022-04-05 14:28:28.940069: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8101                                                      
(4, 26, 26, 2)                                                                                                                                                

'''