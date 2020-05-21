'''
xzl: flask table def 

cf: 
https://flask-table.readthedocs.io/en/stable/#quick-start

'''

from flask_table import Table, Col, LinkCol

class VideoList(Table):
    video_name = Col('Name')
    n_frames = Col('#Frames')
    n_missing_frames = Col('#MissingFrames')
    frame_id_min = Col('FrameMin')
    frame_id_max = Col('FrameMax')
    fps = Col('fps')
    start = Col('start')
    end = Col('end')
    duration = Col('duration')
    query = LinkCol('Query', 'query', url_kwargs=dict(videoname='video_name'))

class QueryList(Table):
    qid = Col('qid')
    
    status_cloud = Col('status_cloud')
    
    status_cam = Col('status_cam')
    ts_comp_cam = Col('ts_comp_cam')
    
    n_frames_sent_cam = Col('n_frames_sent_cam')
    n_frames_processed_cam = Col('n_frames_processed_cam')
    n_frames_total_cam = Col('n_frames_total_cam')
        
    n_frames_recv_yolo = Col('n_frames_recv_yolo')
    n_frames_processed_yolo = Col('n_frames_processed_yolo')
    
    target_class = Col('target_class')
    show_results = LinkCol('Results', 'query_results', url_kwargs=dict(qid='qid'))
    #query = LinkCol('Pause', 'pause', url_kwargs=dict(qid='qid'))
        
# a list of summaries of all query results
class QueryResultsList(Table):
#    qid = Col('qid')
    status = Col('status')
    n_frames_recv_cam = Col('n_frames_sent_cam')
    n_frames_processed_cam = Col('n_frames_processed_cam')
    n_frames_recv_yolo = Col('n_frames_recv_yolo')
    n_frames_processed_yolo = Col('n_frames_processed_yolo')
    target_class = Col('target_class')
    results = LinkCol('Results', 'query_results', url_kwargs=dict(qid='qid'))