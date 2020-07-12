# Quickstart

## Overview

The whole system has three modules: 

1. the web server. e.g. a server machine. GPU not needed.
2. the YOLO service. A machine with GPU is preferred. 
3. the cam module. e.g. Rpi3. An x86 server is okay. GPU not required. 

1 & 2 can run on the same machine. 

## How to run

### The web server + controller (on local prevision2)
```
source venv3.7/bin/activate
# launch the web + backend
./run-web.py
# or, launch the console mode, for debugging the backend only
./run-console.py
```

### The camera service (on local precision2)
```
./run-cam.sh
```

### The YOLO service (on precision, TITAN V)
```
source venv-yolo/bin/activate
python ./YOLOv3_grpc.py
```

Then point the browser to: http://10.10.10.3:5006/server

## How to build

Grab the source code 

```{shell}
git clone ${this_repo}

# grab tf-yolov3 as a submodule
git submodule update --init --recursive
```

Python. We need 3.7 which is not default in older Ubuntu (e.g. 18.04)

```
# prep... last digit means priority 
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10

sudo apt install python3.7
sudo apt install python3-pip
sudo apt install libatlas-base-dev #numpy
sudo apt install python3.7-dev # note the version. needed for pip to install/build some packages, e.g. psutil 
```

Opencv2 dependencies (probably should switch to PIL)
```
sudo apt install libqtgui4/stable
sudo apt install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test \
libgstreamer1.0-0/stable
```

**Prep virtualenv, which is needed for Tensorflow installation**

```
# per tf's official instructions: virtual env for tf in order to  
pip3 install -U pip virtualenv

virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip3 list
```

**Install Tensorflow1 etc on camera**

If using Rpi, use rapsbian 9. On debian/rpi64 - pip has no tf package

The camera needs its own **venv** (on local storage, better not on NFS). Make one. 

As of now, our cam code works with tf1. It has some compat issue with the latest tf2. 
See comments in main_camera.py and https://github.com/keras-team/keras/issues/13336

```
# to check available tf versions, 
#  pip3 install tensorflow==
  
pip3 install tensorflow==1.13.1df -
pip3 install pandas
# pip3 install opencv-python # no longer needed ... introduced too much dep. bad. 
pip3 install numpy \
pandas keras sklearn 

# need a specific opencv version for rpi
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/issues/67
pip3 install opencv-python==3.4.6.27

# for getting camera realtime resource usage
pip install psutil
# for getting extra OS info
# pip install py-cpuinfo

pip3 install pillow  # python image library
#pip install flask flask_table # for webserver to gen table
#pip install WTForms # for webserver to render forms
pip install coloredlogs # easy tracing
pip3 install grpcio-tools
pip install zc.lockfile # to avoid multiple running instances
```

**On server**

```
pip3 install tensorflow

ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3_backbone.py
ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3_common.py
ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3_config.py
ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3_dataset.py
ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3.py
ln -sf third_party/TensorFlow2_0_Examples/Object_Detection/YOLOV3/core/tensorflow_yolov3_utils.py

# NB: tf v2.2 has CPU/GPU branches unified
# need to create a separate env for python-yolov3, which depends on resize() func absent in tf-1
# 		the func is tf.image.resize(input_layer...
pip3 install easydict # needed by tf-yolov3
pip3 install opencv-python 

# gen grpc code
python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/*

# Init DB
# docker stop mypgdb && docker rm mypgdb && make run-postgres && sleep 10 && make init-postgres && make fixture-postgres

# xzl: on the camera side. 
# pip3 install flask

# a.	Go to project directory (keep the codebase up to date) && configure variables in camera/camera_constants.py
# b.	env FLASK_ENV=production python3 -m camera.app &
# c.	python3 -m camera.main_camera


#make setup-env
#make run-yolo
#make run-cloud

```

## Important paths

**Video frames** 

hybridvs_data/YOLO-RES-720P/jpg/XXXX
where XXX is the video name. 

### assumption:

video naming: video is named as `${scene}-${seg}_XXfps-AAxBB`
${scene}, e.g.  "purdue", "chaweng", ...
${seg}, optional, video segment id. 
XXXfps is an optional hint, used to determine FPS
AAxBB is an optional hint, used to determine resolution

**Cam models (ops)**

./ops/

opname: ops on the camera side are named as ${scene}-0, ${scene}-1, ${scene}-2...
op num: 0, 1, 2. .. up to 5

with symbolic links

## host env

https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu


```
sudo apt install python3.7
```


# troubleshooting

## complains because no python alternatives are  installed

debian@debian-rpi64:~$ 
update-alternatives --list python
update-alternatives: error: no alternatives for python

### "install" alternatives
```
# the integer number at the end of each command denotes a priority. Higher number means higher priority
#sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.4 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2

sudo update-alternatives --config python
```

```
# create new 3.7 venv
virtualenv --system-site-packages -p python3.7 ./venv3.7
```

```
# or upgrade existing venv (not working??)
sudo apt install python3.7-venv
python -m venv --upgrade YOUR_VENV_DIRECTORY
```

## tensorflow GPU config (for yolov3)

libcudnn shall be 7.6.4 or 7.6.5
are we using cuda9 or cuda10??

```
(venv) xzl@precision (demo+)[diva-fork]$ python -V
Python 3.7.3

(venv) xzl@precision (demo+)[diva-fork]$    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2020-05-19 21:46:00.853652: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-05-19 21:46:00.971251: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:03:00.0 name: TITAN V computeCapability: 7.0
coreClock: 1.455GHz coreCount: 80 deviceMemorySize: 11.78GiB deviceMemoryBandwidth: 607.97GiB/s
2020-05-19 21:46:00.973346: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-05-19 21:46:01.002394: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-05-19 21:46:01.019205: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-05-19 21:46:01.023369: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-05-19 21:46:01.053923: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-05-19 21:46:01.059088: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-05-19 21:46:01.098227: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-19 21:46:01.101290: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-05-19 21:46:01.107914: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 21:46:01.238111: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 2394595000 Hz
2020-05-19 21:46:01.239304: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x3fa3370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 21:46:01.239345: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-05-19 21:46:01.449578: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x403d820 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-05-19 21:46:01.449626: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): TITAN V, Compute Capability 7.0
2020-05-19 21:46:01.451540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:03:00.0 name: TITAN V computeCapability: 7.0
coreClock: 1.455GHz coreCount: 80 deviceMemorySize: 11.78GiB deviceMemoryBandwidth: 607.97GiB/s
2020-05-19 21:46:01.451611: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-05-19 21:46:01.451640: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-05-19 21:46:01.451666: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-05-19 21:46:01.451691: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-05-19 21:46:01.451716: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-05-19 21:46:01.451741: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-05-19 21:46:01.451768: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-19 21:46:01.454548: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-05-19 21:46:01.473081: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-05-19 21:46:01.475062: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-19 21:46:01.475085: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0
2020-05-19 21:46:01.475097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N
2020-05-19 21:46:01.483356: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11002 MB memory) -> physical GPU (device: 0, name: TITAN V, pci bus id: 0000:03:00.0, compute capability: 7.0)
tf.Tensor(24.069794, shape=(), dtype=float32)

(venv) xzl@precision (demo+)[diva-fork]$ nvidia-smi
Tue May 19 21:47:32 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.64.00    Driver Version: 440.64.00    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN V             Off  | 00000000:03:00.0  On |                  N/A |
| 30%   43C    P8    26W / 250W |     37MiB / 12058MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1556      G   /usr/lib/xorg/Xorg                            35MiB |
+-----------------------------------------------------------------------------+


(venv) xzl@precision (demo+)[diva-fork]$ dpkg -l |grep cuda|more
ii  cuda-command-line-tools-10-1                         10.1.243-1                                               amd64        CUDA command-line tools
ii  cuda-command-line-tools-9-0                          9.0.176-1                                                amd64        CUDA command-line tools
ii  cuda-compiler-10-1                                   10.1.243-1                                               amd64        CUDA compiler
ii  cuda-core-9-0                                        9.0.176.3-1                                              amd64        CUDA core tools
ii  cuda-cublas-9-0                                      9.0.176.4-1                                              amd64        CUBLAS native runtime libraries
ii  cuda-cudart-10-1                                     10.1.243-1                                               amd64        CUDA Runtime native Libraries
ii  cuda-cudart-9-0                                      9.0.176-1                                                amd64        CUDA Runtime native Libraries
ii  cuda-cudart-dev-10-1                                 10.1.243-1                                               amd64        CUDA Runtime native dev links, headers
ii  cuda-cudart-dev-9-0                                  9.0.176-1                                                amd64        CUDA Runtime native dev links, headers
ii  cuda-cufft-10-1                                      10.1.243-1                                               amd64        CUFFT native runtime libraries
ii  cuda-cufft-9-0                                       9.0.176-1                                                amd64        CUFFT native runtime libraries
ii  cuda-cufft-dev-10-1                                  10.1.243-1                                               amd64        CUFFT native dev links, headers
ii  cuda-cuobjdump-10-1                                  10.1.243-1                                               amd64        CUDA cuobjdump
ii  cuda-cupti-10-1                                      10.1.243-1                                               amd64        CUDA profiling tools interface.
ii  cuda-curand-10-1                                     10.1.243-1                                               amd64        CURAND native runtime libraries
ii  cuda-curand-9-0                                      9.0.176-1                                                amd64        CURAND native runtime libraries
ii  cuda-curand-dev-10-1                                 10.1.243-1                                               amd64        CURAND native dev links, headers
ii  cuda-cusolver-10-1                                   10.1.243-1                                               amd64        CUDA solver native runtime libraries
ii  cuda-cusolver-9-0                                    9.0.176-1                                                amd64        CUDA solver native runtime libraries
ii  cuda-cusolver-dev-10-1                               10.1.243-1                                               amd64        CUDA solver native dev links, headers
ii  cuda-cusparse-10-1                                   10.1.243-1                                               amd64        CUSPARSE native runtime libraries
ii  cuda-cusparse-9-0                                    9.0.176-1                                                amd64        CUSPARSE native runtime libraries
ii  cuda-cusparse-dev-10-1                               10.1.243-1                                               amd64        CUSPARSE native dev links, headers
ii  cuda-documentation-10-1                              10.1.243-1                                               amd64        CUDA documentation
ii  cuda-driver-dev-10-1                                 10.1.243-1                                               amd64        CUDA Driver native dev stub library
ii  cuda-driver-dev-9-0                                  9.0.176-1                                                amd64        CUDA Driver native dev stub library
ii  cuda-gdb-10-1                                        10.1.243-1                                               amd64        CUDA-GDB
ii  cuda-gpu-library-advisor-10-1                        10.1.243-1                                               amd64        CUDA GPU Library Advisor.
ii  cuda-libraries-10-1                                  10.1.243-1                                               amd64        CUDA Libraries 10.1 meta-package
ii  cuda-libraries-dev-10-1                              10.1.243-1                                               amd64        CUDA Libraries 10.1 development meta-package
ii  cuda-license-10-1                                    10.1.243-1                                               amd64        CUDA licenses
ii  cuda-license-10-2                                    10.2.89-1                                                amd64        CUDA licenses
ii  cuda-license-9-0                                     9.0.176-1                                                amd64        CUDA licenses
ii  cuda-memcheck-10-1                                   10.1.243-1                                               amd64        CUDA-MEMCHECK
ii  cuda-misc-headers-10-1                               10.1.243-1                                               amd64        CUDA miscellaneous headers
ii  cuda-misc-headers-9-0                                9.0.176-1                                                amd64        CUDA miscellaneous headers
ii  cuda-npp-10-1                                        10.1.243-1                                               amd64        NPP native runtime libraries
ii  cuda-npp-dev-10-1                                    10.1.243-1                                               amd64        NPP native dev links, headers
ii  cuda-nsight-10-1                                     10.1.243-1                                               amd64        CUDA nsight
ii  cuda-nsight-compute-10-1                             10.1.243-1                                               amd64        NVIDIA Nsight Compute
ii  cuda-nsight-systems-10-1                             10.1.243-1                                               amd64        NVIDIA Nsight Systems
ii  cuda-nvcc-10-1                                       10.1.243-1                                               amd64        CUDA nvcc
ii  cuda-nvdisasm-10-1                                   10.1.243-1                                               amd64        CUDA disassembler
ii  cuda-nvgraph-10-1                                    10.1.243-1                                               amd64        NVGRAPH native runtime libraries
ii  cuda-nvgraph-dev-10-1                                10.1.243-1                                               amd64        NVGRAPH native dev links, headers
ii  cuda-nvjpeg-10-1                                     10.1.243-1                                               amd64        NVJPEG native runtime libraries
ii  cuda-nvjpeg-dev-10-1                                 10.1.243-1                                               amd64        NVJPEG native dev links, headers
ii  cuda-nvml-dev-10-1                                   10.1.243-1                                               amd64        NVML native dev links, headers
ii  cuda-nvprof-10-1                                     10.1.243-1                                               amd64        CUDA Profiler tools
ii  cuda-nvprune-10-1                                    10.1.243-1                                               amd64        CUDA nvprune
ii  cuda-nvrtc-10-1                                      10.1.143-1                                               amd64        NVRTC native runtime libraries
ii  cuda-nvrtc-dev-10-1                                  10.1.243-1                                               amd64        NVRTC native dev links, headers
ii  cuda-nvtx-10-1                                       10.1.243-1                                               amd64        NVIDIA Tools Extension
ii  cuda-nvvp-10-1                                       10.1.243-1                                               amd64        CUDA nvvp
ii  cuda-repo-ubuntu1604                                 10.1.243-1                                               amd64        cuda repository configuration files
ii  cuda-samples-10-1                                    10.1.243-1                                               amd64        CUDA example applications
ii  cuda-sanitizer-api-10-1                              10.1.243-1                                               amd64        CUDA Sanitizer API
ii  cuda-toolkit-10-1                                    10.1.243-1                                               amd64        CUDA Toolkit 10.1 meta-package
ii  cuda-tools-10-1                                      10.1.243-1                                               amd64        CUDA Tools meta-package
ii  cuda-visual-tools-10-1                               10.1.243-1                                               amd64        CUDA visual tools
```