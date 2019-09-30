run-cloud:
	docker run -d --gpus all --name cloud --mount source=/media/teddyxu/WD-4TB/,target=/media/ diva-cloud:latest

run-camera:
	docker run -d --gpus all --name camera --mount source=result/ops,target=result/ops diva-camera:latest

run-yolo:
	docker run -d --gpus all --rm --name=yolo diva-yolo:latest

build-base:
	docker build  -t diva-base:latest -f docker/Dockerfile.base .

build-yolo:
	docker build  -t diva-yolo:latest -f docker/Dockerfile.yolo .

build-cloud:
	docker build  -t diva-cloud:latest -f docker/Dockerfile.cloud .

build-camera:
	docker build  -t diva-camera:latest -f docker/Dockerfile.camera .
