from __future__ import print_function
import logging
import os

import bottle
import grpc

import server_diva_pb2
import server_diva_pb2_grpc

from variables import DIVA_CHANNEL_ADDRESS


directory = os.getcwd()
app = bottle.Bottle()

# Change many values at once
app.config.update({
    'autojson': False,
    'sqlite.db': ':memory:',
    'myapp.param': 'value',
    'app.host': 'localhost',
    'app.port': 10000,
})


@app.hook('after_request')
def enable_cors():
    """
    You need to add some headers to each request.
    Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
    """
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS, GET'
    bottle.response.headers[
        'Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@app.get('/<filename:path>', method="GET")  # ":path, the filter as path format"
def server_static(filename):
    return bottle.static_file(filename, root=directory)


@app.route('/hello')
def hello():
    return "Hello World!"


@app.route('/')
def index():
    return bottle.static_file('./template/index.html', root=directory)


@app.route('/display', method="POST")
def query():
    logging.basicConfig()
    with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        stub = server_diva_pb2_grpc.server_divaStub(channel)
        response = stub.request_frame_path(server_diva_pb2.query_statement(name='request directory'))
    pic_files = list()
    pic_files.append(response.directory_path)
    if os.path.exists(response.directory_path):
        temp = os.listdir(response.directory_path)
        for files in temp:
            if '.png' in files or '.jpg' in files or '.jpeg' in files:
                pic_files.append(files)
    temp = ','.join(pic_files)
    return tuple(temp)


if __name__ == '__main__':
    app.config.load_config('conf.ini')
    bottle.run(app, host='localhost', port=10000, debug=True)
