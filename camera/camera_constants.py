'''
VIDEO_FOLDER = "~/workspace/test_data/source/"
_HOST = '0.tcp.ngrok.io'
_PORT = '17171'
_ADDRESS = f'{_HOST}:{_PORT}'
_NAME = 'jetson'
YOLO_CHANNEL_ADDRESS = "128.46.74.209:10088"
STATIC_FOLDER = '/tmp/camera'
WEB_APP_DNS = 'https://63858849.ngrok.io:8000'
'''

# xzl, for same-machine docker dev
VIDEO_FOLDER = "/var/diva/cam-workspace/" 

# xzl: will be passed back to browser 
_HOST = 'camera'  
_PORT = '10086' # why inconsistent?
_ADDRESS = f'{_HOST}:{_PORT}'
_NAME = 'jetson' # intent for indexing multi cams? 

YOLO_CHANNEL_ADDRESS = "yolo:10088"
STATIC_FOLDER = '/tmp/camera'
WEB_APP_DNS = 'https://camera:8000'
