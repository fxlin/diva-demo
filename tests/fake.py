import os
import logging
import time
import pandas as pd
import cv2

import grpc

import common_pb2
import det_yolov3_pb2
import det_yolov3_pb2_grpc

from variables import YOLO_CHANNEL_ADDRESS

VIDEO_SOURCE = '/var/yolov3/web/static/video/example.mp4'

target_class = 'motorbike'

print('start')


def trim_video(source_path: str, output_path: str, start_second: int,
               end_second: int):
    source_video = cv2.VideoCapture(source_path)

    if not source_video.isOpened():
        raise Exception("Video is not opened")

    target_width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_fps = source_video.get(cv2.CAP_PROP_FPS)

    source_video.set(cv2.CAP_PROP_POS_FRAMES, target_fps * start_second)
    counter = target_fps * end_second

    _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, _fourcc, target_fps,
                                   (target_width, target_height))

    logging.info(f'time {time.time()}')
    while counter >= 0:
        ret, frame = source_video.read()
        if not ret:
            break
        output_video.write(frame)
        counter -= 1
    logging.info(f'time {time.time()} file {output_path} exists? \
         {os.path.exists(output_path)}')

    source_video.release()
    output_video.release()


source = cv2.VideoCapture(VIDEO_SOURCE)
counter = 0

bbox_size_map = {}

metric_df = pd.DataFrame(columns=['start_time', 'end_time', 'diff', 'score'])

with grpc.insecure_channel(YOLO_CHANNEL_ADDRESS) as channel:
    stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)

    while source.isOpened():
        ret, frame = source.read()
        if not ret:
            break

        if (counter % 30) == 0:
            # send image to process

            t_start = time.time()

            _height, _width, _chan = frame.shape
            _img = common_pb2.Image(data=frame.tobytes(),
                                    height=_height,
                                    width=_width,
                                    channel=_chan)

            req = det_yolov3_pb2.DetectionRequest(image=_img,
                                                  name=f'{counter}.jpg',
                                                  threshold=0.3,
                                                  targets=[target_class])
            resp = stub.Detect(req)

            exist_target = False

            temp_score = []

            # draw bbox on the image
            for ele in resp.elements:
                if ele.class_name != target_class:
                    continue
                exist_target = True
                x1, y1, x2, y2 = ele.x1, ele.y1, ele.x2, ele.y2

                bbox_size = abs(x1 - x2) * abs(y1 - y2)
                bbox_size_map[counter] = bbox_size

                new_img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),
                                        3)

                temp_score.append(ele.confidence)

                cv2.imwrite(f'tests/img/{target_class}/{counter}.jpg', new_img)

            if exist_target:
                # trim the video
                trim_video(VIDEO_SOURCE,
                           f'tests/video/{target_class}/{counter}.mp4',
                           counter // 30, (counter // 30) + 5)

            t_end = time.time()

            temp_list = [[t_start, t_end, t_start - t_end, c]
                         for c in temp_score]
            temp_df = pd.DataFrame(
                temp_list, columns=['start_time', 'end_time', 'diff', 'score'])

            metric_df.append(temp_df)

        counter += 1

metric_df.to_csv(f'tests/img/{target_class}_score.csv')

# new_list = sorted(list(bbox_size_map.items()), key=lambda x: x[1])
# fake_score = np.linspace(start=0.3, stop=1.0, num=len(new_list))
# name_score_mapping = {}
# for i, v in zip(new_list, fake_score):
#     name = f'{i[0]}.jpg'
#     score = v
#     name_score_mapping[name] = score
# with open(f'{target_class}_score.txt', 'w') as txt_fptr:
#     json.dump(name_score_mapping, txt_fptr)

source.release()

print("Done")

exit(0)

# message Image {
#     // bytes come from opencv .tobytes() function
#     bytes data = 1;
#     int32 height = 2;
#     int32 width = 3;
#     int32 channel = 4;
# }

# message DetectionRequest {
#     common.Image image = 1;
#     string name = 2;
#     float threshold = 3;
#     repeated string targets = 4;
# }

# message Element {
#     string class_name = 1;
#     double confidence = 2;
#     int32 x1 = 3;
#     int32 y1 = 4;
#     int32 x2 = 5;
#     int32 y2 = 6;
# }

# message DetectionOutput {
#     repeated Element elements = 1;
# }
