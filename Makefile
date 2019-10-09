NETWORK_NAME=diva-network

VIDEO_DATA_PATH=/media/teddyxu/WD-4TB/hybridvs_data
VIDEO_DATA_PATH_IN_CONTAINER=/media/

CAMERA_RESULT_PATH=${PWD}/result/ops
CAMERA_RESULT_PATH_IN_CONTAINER=/var/yolov3/result/ops

run-all:
	@make run-yolo
	@make run-camera
	@make run-cloud
	@make run-web

start-network:
	docker network create --subnet 172.20.0.0/16 --ip-range 172.20.240.0/20 ${NETWORK_NAME}

remove-network:
	docker network rm ${NETWORK_NAME}

run-cloud:
	docker run --network=${NETWORK_NAME} -d --gpus all --name cloud -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro diva-cloud:latest

run-web:
	docker run --network=${NETWORK_NAME} -d --gpus all --name web -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro diva-web:latest

run-camera:
	docker run --network=${NETWORK_NAME}  -d --gpus all --name camera -v ${CAMERA_RESULT_PATH}:${CAMERA_RESULT_PATH_IN_CONTAINER} diva-camera:latest

run-yolo:
	docker run --network=${NETWORK_NAME}  -d --gpus all --name=yolo diva-yolo:latest

build-base:
	docker build  -t diva-base:latest -f docker/Dockerfile.base .

build-yolo:
	docker build  -t diva-yolo:latest -f docker/Dockerfile.yolo .

build-cloud:
	docker build  -t diva-cloud:latest -f docker/Dockerfile.cloud .

build-camera:
	docker build  -t diva-camera:latest -f docker/Dockerfile.camera .

build-web:
	docker build  -t diva-web:latest -f docker/Dockerfile.web .
