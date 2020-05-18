'''
xzl: flask table def 

cf: https://flask-table.readthedocs.io/en/stable/#quick-start

'''

from flask_table import Table, Col, LinkCol

class VideoList(Table):
    name = Col('Name')
    n_frames = Col('#Frames')
    n_missing_frames = Col('#MissingFrames')
    fps = Col('fps')
    start = Col('start')
    end = Col('end')
    duration = Col('duration')
    query = LinkCol('Query', 'query', url_kwargs=dict(videoname='name'))
    

class QueryList(Table):
    qid = Col('qid')
    status = Col('status')
    n_frames_recv_cam = Col('n_frames_recv_cam')
    n_frames_processed_cam = Col('n_frames_processed_cam')
    n_frames_recv_yolo = Col('n_frames_recv_yolo')
    n_frames_processed_yolo = Col('n_frames_processed_yolo')
    target_class = Col('target_class')
    #query = LinkCol('Pause', 'pause', url_kwargs=dict(qid='qid'))