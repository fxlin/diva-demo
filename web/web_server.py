from __future__ import print_function
import logging
import grpc
import server_diva_pb2
import server_diva_pb2_grpc
from variables import DIVA_CHANNEL_ADDRESS
from flask import Flask, render_template, request
from flask import jsonify, send_from_directory
import query


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
        _ = stub.detect_object_in_video(server_diva_pb2.object_video_pair(object_name=obj,  video_name=video))
    return jsonify(file='good')


@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve_frames():
    time = list()
    video_name = request.json['video'].split('/')[-1]
    print(video_name)
    name = query.request_frames(video_name)
    if not name:
        print('not finished')
        return jsonify(file=name)
    print('finished')
    for i in name:
        time.append(str(int(i.split('.')[0]) // 10))
    name = ','.join(name)
    time = ','.join(time)
    return jsonify(file=name, t=time)


@app.route('/coordinates', methods=['GET', 'POST'])
def retrieve_coordinates():
    frame_id = request.json['frame']
    time = request.json['time']
    coordinates = query.request_coordinates(frame_id, time)
    return jsonify(coordinates=coordinates)

if __name__ == '__main__':
    app.run('0.0.0.0', 10000)

