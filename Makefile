NETWORK_NAME=diva-network

# xzl: for cloud ... not useful?
VIDEO_DATA_PATH=/media/teddyxu/WD-4TB/hybridvs_data
VIDEO_DATA_PATH_IN_CONTAINER=/media/

CAMERA_RESULT_PATH=${PWD}/result/ops
CAMERA_RESULT_PATH_IN_CONTAINER=/var/yolov3/result/ops
DOCKER_USERNAME=wen777

DEFAULT_POSTGRES_USER=postgres
DEFAULT_POSTGRES_PASSWORD=xsel_postgres
DEFAULT_POSTGRES_DB=xsel_test
DEFAULT_POSTGRES_PORT=5432
DEFAULT_POSTGRES_HOST=mypgdb

WEB_SERVER_PORT=10000
YOLO_SERVICE_PORT=10088
CLOUD_SERVICE_PORT=10090

PERSISTENT_VOLUME='/tmp/diva_test'
WEB_SERVER_IMAGE_VOLUME='/var/web/web/static/output'
CONTROLLER_SERVER_IMAGE_VOLUME='/var/yolov3/web/static/output'

#USE_GPU='--gpus all'
USE_GPU=

run-all:
	@make run-cloud
	sleep 2
	@make run-yolo
	@make run-camera
	@make run-webserver

stop-and-remove:
	@docker stop cloud yolo webserver camera
	@docker rm cloud yolo webserver camera

start-network:
	docker network create --subnet 172.20.0.0/16 --ip-range 172.20.240.0/20 ${NETWORK_NAME}

remove-network:
	docker network rm ${NETWORK_NAME}

setup-env: start-network
	@echo "check if ${PERSISTENT_VOLUME} exists"
	@[ ! -d "${PERSISTENT_VOLUME}" ] && mkdir -p "${PERSISTENT_VOLUME}"

run-cloud:
	docker run --network=${NETWORK_NAME} -it -d -p ${CLOUD_SERVICE_PORT}:${CLOUD_SERVICE_PORT} --name cloud -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro -v ${PERSISTENT_VOLUME}:${CONTROLLER_SERVER_IMAGE_VOLUME} ${DOCKER_USERNAME}/diva-cloud:latest

run-cloud-without-disk:
	docker run --network=${NETWORK_NAME} -d -p ${CLOUD_SERVICE_PORT}:${CLOUD_SERVICE_PORT} --name cloud ${DOCKER_USERNAME}/diva-cloud:latest

run-camera:
	docker run --network=${NETWORK_NAME} -d --name camera -v ${CAMERA_RESULT_PATH}:${CAMERA_RESULT_PATH_IN_CONTAINER} ${DOCKER_USERNAME}/diva-camera:latest

run-camera-pi:
	docker run --network=${NETWORK_NAME} --device=/dev/vcsm --device=/dev/vchiq -d ${USE_GPU} --name camera -v ${CAMERA_RESULT_PATH}:${CAMERA_RESULT_PATH_IN_CONTAINER} ${DOCKER_USERNAME}/diva-camera:latest

run-yolo:
	docker run --network=${NETWORK_NAME} -d ${USE_GPU} --name=yolo ${DOCKER_USERNAME}/diva-yolo:latest

run-yolo-with-port:
	docker run --network=${NETWORK_NAME} -d ${USE_GPU} --name=yolo -p ${YOLO_SERVICE_PORT}:${YOLO_SERVICE_PORT} ${DOCKER_USERNAME}/diva-yolo:latest

run-webserver:
	docker run --network=${NETWORK_NAME} -it -p ${WEB_SERVER_PORT}:${WEB_SERVER_PORT} -v ${PERSISTENT_VOLUME}:${WEB_SERVER_IMAGE_VOLUME} -d --name=webserver ${DOCKER_USERNAME}/diva-webserver:latest

run-postgres:
	docker run --network=${NETWORK_NAME} --name ${DEFAULT_POSTGRES_HOST} -p ${DEFAULT_POSTGRES_PORT}:${DEFAULT_POSTGRES_PORT} \
		-e POSTGRES_PASSWORD=${DEFAULT_POSTGRES_PASSWORD} \
		-e POSTGRES_DB=${DEFAULT_POSTGRES_DB} -d postgres:latest

init-postgres:
	docker run --network=${NETWORK_NAME} --rm -it --name init_db  ${DOCKER_USERNAME}/diva-cloud:latest python -m migration.xsel_server

fixture-postgres:
	docker run --network=${NETWORK_NAME} --rm -it --name init_db  ${DOCKER_USERNAME}/diva-cloud:latest python -m migration.test_case

build-base:
	docker build  -t ${DOCKER_USERNAME}/diva-base:latest -f docker/Dockerfile.base .

build-yolo:
	docker build  -t ${DOCKER_USERNAME}/diva-yolo:latest -f docker/Dockerfile.yolo .

build-cloud:
	docker build  -t ${DOCKER_USERNAME}/diva-cloud:latest -f docker/Dockerfile.cloud .

build-camera:
	docker build  -t ${DOCKER_USERNAME}/diva-camera:latest -f docker/Dockerfile.camera .

build-webserver:
	docker build  -t ${DOCKER_USERNAME}/diva-webserver:latest -f docker/Dockerfile.webserver .

build-docker: build-base build-cloud build-camera build-yolo build-webserver
	@echo "building docker image for all components"

push-docker: build-docker
	docker push ${DOCKER_USERNAME}/diva-yolo:latest
	docker push ${DOCKER_USERNAME}/diva-cloud:latest
	docker push ${DOCKER_USERNAME}/diva-camera:latest
	docker push ${DOCKER_USERNAME}/diva-webserver:latest

test-cloud:
	docker run --network=${NETWORK_NAME} -it --rm --name cloud  ${DOCKER_USERNAME}/diva-cloud:latest python -m tests.test_main_cloud

test-yolo:
	docker run --network=${NETWORK_NAME} -it --rm ${USE_GPU} --name=test_yolo ${DOCKER_USERNAME}/diva-yolo:latest python -m tests.test_yolo

test-integration:
	docker run --network=${NETWORK_NAME} -it --rm --name test_integration ${DOCKER_USERNAME}/diva-cloud:latest python -m tests.test_integration

test-fake:
	docker run --network=${NETWORK_NAME} -it --rm --name test_fake -v /tmp/test/video:/var/yolov3/tests/video -v /tmp/test/img:/var/yolov3/tests/img ${DOCKER_USERNAME}/diva-cloud:latest python -m tests.fake

