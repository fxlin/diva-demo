from __future__ import print_function
import sys
import logging
import grpc
import server_diva_pb2
import server_diva_pb2_grpc
from variables import DIVA_CHANNEL_ADDRESS
from flask import Flask, render_template, request, flash, redirect
from flask import jsonify,  send_from_directory
import json
import os
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
# import grpc-tools
import requests
from builtins import True

OUTPUT_DIR = './result/retrieval_imgs/' # xzl: query results for the web server? 
app = Flask(__name__, static_url_path='/static') # xzl: static contents of the website
app.config['DEBUG'] = True


FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# xzl: return give files to the client. why /uploads?/
@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

# xzl: get metadata of all stored videos.
# send a json response to the browser. will the webpage parse it? 
@app.route('/request_video',methods=['GET', 'POST'])
def request_video():
    logger.info('got a req: get metadata for all videos')
    frames = list()
    name = list()
    camera_name = list()
    camera_address = list()
    video_url = list()
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        response = stub.get_videos(google_dot_protobuf_dot_empty__pb2.Empty())
        print(response)
    # response = response.videos[2]
    
    # xzl: "name" seems a list of video names. this seems weird    
    for i in range(4):  # xzl: FIXME. hard coded 
        frames.append(response.videos[i].frames)
        name.append(response.videos[i].name)
        camera_name.append(response.videos[i].camera.name)
        camera_address.append(response.videos[i].camera.address)
        video_url.append(response.videos[i].video_url)
    print(frames)
    return jsonify(frame=frames, video=name, image_URL='',
                   video_URL=video_url, score_URL='', camera_address=camera_address, camera_name=camera_name)
    
# xzl: client asks to process a specific video & display results (where does "request" come from?)
@app.route('/display', methods=['GET', 'POST'])
def display():
    logger.info('got a req: query')
    obj = request.json['object']
    video = request.json['video'].split('/')[-1]
    cam_name = request.json['camera_name']
    cam_address = request.json['camera_address']
    start = request.json['timestamp']
    end = request.json['offset']
    print(end)
    print(obj, video, cam_name, cam_address, start, end)
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        response = stub.process_video(server_diva_pb2.common__pb2.VideoRequest(timestamp=0, offset=int(end),
                                                                               video_name=video, object_name=obj,
                                                                               camera={'name': cam_name, 'address': cam_address}))
        # xzl: get query res. just URLs pointing to the res. not res data
        response = stub.get_video(server_diva_pb2.common__pb2.VideoRequest(timestamp=0, offset=int(end),
                                                                               video_name=video,
                                                                               object_name='motorbike',
                                                                               camera={'name': cam_name,
                                                                                       'address': cam_address}))
    print(response)
    image_list = list()
    images_URL = response.images_url
    conf_URL = response.score_file_url

    # xzl: why send confidence as a URL? not making sense

    print(conf_URL)
    while True:
        try:
            confidence = requests.get(conf_URL)
            confidence = json.loads(confidence.content)
        except:
            continue
        break
    
    # xzl: gen a list of image urls, higher confidence first (?)
    for i in confidence:
        image_list.append(images_URL + i + ".jpg")
    print(image_list, confidence)
    return jsonify(image_url=image_list, conf=confidence)


# @app.route('/retrieve', methods=['GET', 'POST'])
# def retrieve_frames():
#     time = list()
#     video_name = request.json['video'].split('/')[-1]
#     print(video_name)
#     name, status = query.request_frames(video_name)
#     for i in name:
#         time.append(str(int(i) // 10))
#     name = ','.join(name)
#     time = ','.join(time)
#     print(time)
#     if not status:
#         print('not finished')
#         return jsonify(file=name, t=time, status=False)
#     print('finished')
#     return jsonify(file=name, t=time, status=True)

# xzl: for showing progress?
@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    logger.info('got a req: refresh')
    frames = request.json['frame']
    name = request.json['video']
    camera_name = request.json['camera_name']
    camera_address = request.json['camera_address']
    video_url = request.json['video_URL']
    print(frames, name, camera_name, camera_address, video_url)
    # print('ok')
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        response = stub.get_video(timestamp=0, offset=1000, video_name="example.mp4", object_name="motorbike", camera={'name': "jetson", 'address': "3.134.196.116:17911"})
        print(response)
    images_URL = response.images_url
    conf_URL = response.score_file_url
    print(conf_URL, images_URL)
    # with uurlopen(conf_URL) as conf:
    #     #     confidence = json.loads(conf.read())
    confidence = 0
    image_name = images_URL.split('/')[-1].split('.')
    image_name = image_name[0]
    return jsonify(img=images_URL, conf=confidence, image_name=image_name)

# xzl: what is this for?
@app.route('/temp', methods=['GET', 'POST'])
def temp():
    logger.info('got a req: temp')
    obj = request.json['object']
    print(obj)
    confidence_file = obj + '_score.txt'
    with open(os.path.join('./web/static/test1/img', confidence_file), 'r') as f:
        conf = f.read()
    # print(conf)
    confidence = json.loads(conf)
    print(confidence)
    images = os.listdir(os.path.join('./web/static/test1/img', obj))
    videos = os.listdir(os.path.join('./web/static/test1/video', obj))
    images.sort()
    videos.sort()
    print(len(images))
    return jsonify(img=images, vid=videos, conf=confidence, obj=obj)


# @app.route('/ret_num', methods=['GET', 'POST'])
# def ret_num():
#     time = list()
#     video_name = request.json['video'].split('/')[-1]
#     num_processed, total = query.num_frames(video_name)
#     return jsonify(processed=num_processed, total=total)


# @app.route('/coordinates', methods=['GET', 'POST'])
# def retrieve_coordinates():
#     video = request.json['video'].split('/')[-1]
#     coordinates = query.request_coordinates(video)
#     return jsonify(coordinates)


if __name__ == '__main__':
    app.run('0.0.0.0', 10000)
