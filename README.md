# diva-fork

Quick Start

```{shell}
# activate ssh-agent for github access
eval "$(ssh-agent -s)"
ssh-add <SSH_KEY_PATH> # add ssh key into ssh-agent

git clone ${this_repo}

# grab tf-yolov3 as a submodule
git submodule update --init --recursive

# prep
update-alternatives --install /usr/bin/python python /usr/bin/python3 10


# per tf's official instructions: virtual env for tf in order to  
pip3 install -U pip virtualenv

virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip3 list

# cam has compat issue with latest tf2. see comments in main_camera there  
pip3 install tensorflow-1.13.1 
pip3 install pandas
pip3 install opencv-python 
pip3 install numpy \
pandas keras sklearn \
opencv-python flask 
pip3 install pillow  # python image library
pip install flask_table # for webserver to gen table
pip install WTForms # for webserver to render forms

# python-yolov3 depends on resize() func absent in tf-1
# in a separate virtualenv
# tf.image.resize(input_layer...
pip3 install easydict # needed by tf-yolov3
pip3 install opencv-python 

# gen grpc code
# pip3 install grpcio-tools
python3 -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/*

# Init DB
docker stop mypgdb && docker rm mypgdb && make run-postgres && sleep 10 && make init-postgres && make fixture-postgres

# xzl: on the camera side. 
pip3 install flask

a.	Go to project directory (keep the codebase up to date) && configure variables in camera/camera_constants.py
b.	env FLASK_ENV=production python3 -m camera.app &
c.	python3 -m camera.main_camera


make setup-env
make run-yolo
make run-cloud

```

## WEB

```{shell}
# Run web server: python -m web.web_server
# Output directory: ./web/static/output
```

## TODO

* [ ] Landmakr video clip handling
    * take snapshot --> process it --> collect information
* [ ] Deploy operator
* [ ] start querying --> train operator based on collected landmark frames --> Deploy operator --> ranked video frames --> sending imgs -> YOLO again --> ok --> ask camera send video clip (5 sec)

* Deploy operator & ranked video frames

* yolo_service api: searching for certain object(s):
    * search_objects(image, [object_names]) -> bbox

* contraoller api: receive video and store video on disk

            # FIXME step 1. find camera information
            # element exist
            # step 2. creating video record, requesting video
            # step 3. implementing new api interface to receive video
            #  from camera
            # step 4. process video