"""
simple web app for serving static files
"""

import os
from flask import Flask, abort, render_template
from flask import send_from_directory, make_response
from flask.logging import default_handler
from camera.camera_constants import STATIC_FOLDER
# from flask import Response

BASE_DIR = STATIC_FOLDER
APP_PORT = 8000

app = Flask(__name__, static_folder=BASE_DIR)
app.config.from_object(__name__)
app.logger.removeHandler(default_handler)


@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
def dir_listing(req_path):

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        app.logger.error(abs_path)
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        # return send_file(abs_path)
        # return send_from_directory(BASE_DIR, req_path.split('/')[-1])
        return send_from_directory(BASE_DIR, req_path)

    # Show directory contents
    app.logger.error(f'BASE_DIR: {BASE_DIR}')
    app.logger.error(f'abs_path: {abs_path}')
    files = os.listdir(abs_path)
    response = make_response(render_template('files.html', files=files))
    # response.headers['X-Parachutes'] = 'parachutes are cool'
    return response


if __name__ == '__main__':  # pragma: no cover
    app.run(port=APP_PORT)
