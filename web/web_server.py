from __future__ import print_function
import logging
import grpc
import os
import server_diva_pb2
import server_diva_pb2_grpc

from variables import DIVA_CHANNEL_ADDRESS

from flask import Flask, render_template, request
from flask import jsonify, send_from_directory



OUTPUT_DIR = './result/retrieval_imgs/'
app = Flask(__name__, static_url_path='/static')

POSTGRES = {
 'user': 'postgres',
 'pw': 'silverTip',
 'db': 'flaskmovie',
 'host': 'localhost',
 'port': '10000',
}
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES


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
        #response = stub.request_frame_path(server_diva_pb2.query_statement(name='request directory'))
        response = stub.detect_object_in_video(server_diva_pb2.object_video_pair(object_name=obj,  video_name=video))
    # pic_files = list()
    # pic_files.append(response.directory_path)
    # if os.path.exists(response.directory_path):
    #     temp = os.listdir(response.directory_path)
    #     for files in temp:
    #         if '.png' in files or '.jpg' in files or '.jpeg' in files:
    #             x, _ = files.split('.')
    #             x = int(x) // 10
    #             pic_files.append(files)
    #             pic_files.append(str(x))
    # temp = ','.join(pic_files)
    temp = 'good'
    return jsonify(file=temp)

if __name__ == '__main__':
    app.run('0.0.0.0', 10000)

