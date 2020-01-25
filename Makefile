NETWORK_NAME=diva-network

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

run-cloud:
	docker run --network=${NETWORK_NAME} -it -d --gpus all --name cloud -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro ${DOCKER_USERNAME}/diva-cloud:latest

run-camera:
	docker run --network=${NETWORK_NAME} -it -d --gpus all --name camera -v ${CAMERA_RESULT_PATH}:${CAMERA_RESULT_PATH_IN_CONTAINER} ${DOCKER_USERNAME}/diva-camera:latest

run-yolo:
	docker run --network=${NETWORK_NAME} -it  -d --gpus all --name=yolo ${DOCKER_USERNAME}/diva-yolo:latest

run-yolo-with-port:
	docker run --network=${NETWORK_NAME} -it  -d --gpus all --name=yolo -p ${YOLO_SERVICE_PORT}:${YOLO_SERVICE_PORT} ${DOCKER_USERNAME}/diva-yolo:latest

run-webserver:
	docker run --network=${NETWORK_NAME} -it -p ${WEB_SERVER_PORT}:${WEB_SERVER_PORT} -d --name=webserver ${DOCKER_USERNAME}/diva-webserver:latest

run-postgres:
	docker run --network=${NETWORK_NAME} -it --name ${DEFAULT_POSTGRES_HOST} -p ${DEFAULT_POSTGRES_PORT}:${DEFAULT_POSTGRES_PORT} \
		-e POSTGRES_PASSWORD=${DEFAULT_POSTGRES_PASSWORD} \
		-e POSTGRES_DB=${DEFAULT_POSTGRES_DB} -d postgres:latest

init-postgres:
	docker run --network=${NETWORK_NAME} --rm -it --name init_db  ${DOCKER_USERNAME}/diva-cloud:latest python -m migration.xsel_server

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
	docker run --network=${NETWORK_NAME} -it --rm --name cloud -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro ${DOCKER_USERNAME}/diva-cloud:latest python -m tests.test_main_cloud

test-yolo:
	docker run --network=${NETWORK_NAME} -it --rm --gpus all --name=test_yolo ${DOCKER_USERNAME}/diva-yolo:latest python -m tests.test_yolo

test-integration:
	docker run --network=${NETWORK_NAME} -it --rm --name test_integration -v ${VIDEO_DATA_PATH}:${VIDEO_DATA_PATH_IN_CONTAINER}:ro ${DOCKER_USERNAME}/diva-cloud:latest python -m tests.test_integration
