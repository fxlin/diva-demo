from __future__ import print_function
import logging
import grpc
import server_diva_pb2
import server_diva_pb2_grpc
from variables import DIVA_CHANNEL_ADDRESS
from flask import Flask, render_template, request
from flask import jsonify, send_from_directory
import web.query as query


OUTPUT_DIR = './result/retrieval_imgs/'
app = Flask(__name__, static_url_path='/static')
app.config['DEBUG'] = True



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/display', methods=['GET', 'POST'])
def display():
    obj = request.json['object']
    video = request.json['video'].split('/')[-1]
    print(obj, video)
    logging.basicConfig()
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        #  response = stub.request_frame_path(server_diva_pb2.query_statement(name='request directory'))
        #  FIXME: replace with other function call
        _ = stub.detect_object_in_video(server_diva_pb2.object_video_pair(object_name=obj,  video_name=video))
    return jsonify(file='good')


@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve_frames():
    time = list()
    video_name = request.json['video'].split('/')[-1]
    print(video_name)
    name, status = query.request_frames(video_name)
    for i in name:
        time.append(str(int(i) // 10))
    name = ','.join(name)
    time = ','.join(time)
    print(time)
    if not status:
        print('not finished')
        return jsonify(file=name, t=time, status=False)
    print('finished')
    return jsonify(file=name, t=time, status=True)


@app.route('/ret_num', methods=['GET', 'POST'])
def ret_num():
    time = list()
    video_name = request.json['video'].split('/')[-1]
    num_processed, total = query.num_frames(video_name)
    return jsonify(processed=num_processed, total=total)


@app.route('/coordinates', methods=['GET', 'POST'])
def retrieve_coordinates():
    video = request.json['video'].split('/')[-1]
    coordinates = query.request_coordinates(video)
    return jsonify(coordinates)


if __name__ == '__main__':
    app.run('0.0.0.0', 10000)

