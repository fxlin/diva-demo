"""
Ingest video frames and perform object detection on frames.

xzl: the controller code (?)
"""

import os
import sys
import logging, coloredlogs
from concurrent import futures
import time
import threading
from queue import Queue
import cv2
import copy
import shutil
import numpy as np
import PIL

import grpc
import det_yolov3_pb2
import det_yolov3_pb2_grpc
import cam_cloud_pb2
import cam_cloud_pb2_grpc
import server_diva_pb2_grpc
import common_pb2
from google.protobuf import empty_pb2

import traceback

from util import ClockLog

from variables import CAMERA_CHANNEL_ADDRESS, YOLO_CHANNEL_ADDRESS
from variables import IMAGE_PATH, OP_FNAME_PATH
from variables import DIVA_CHANNEL_PORT
from variables import *  # xzl

from dataclasses import dataclass, field, asdict
import typing

import zc.lockfile # detect& avoid multiple instances

from constants.grpc_constant import INIT_DIVA_SUCCESS

# careful...
from camera.main_camera import FrameMapToFrameStates
from variables import the_videolib_results, the_videolib_preview

CHUNK_SIZE = 1024 * 100
OBJECT_OF_INTEREST = 'bicycle'
CROP_SPEC = '350,0,720,400'
YOLO_SCORE_THRE = 0.4
DET_SIZE = 608
IMAGE_PROCESSOR_WORKER_NUM = 1
FRAME_PROCESSOR_WORKER_NUM = 1

"""
frame task: (video_name, video_path, frame_number, object_of_interest)
image task: (image name, image data, bounding_boxes)
"""
TaskQueue = Queue(0)
ImageQueue = Queue(10)

'''
%(pathname)s Full pathname of the source file where the logging call was issued(if available).
%(filename)s Filename portion of pathname.
%(module)s Module (name portion of filename).
%(funcName)s Name of function containing the logging call.
%(lineno)d Source line number where the logging call was issued (if available).
'''
# FOR'MAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
#FORMAT = '%(levelname)8s %(thread)d %(threadName)s %(message)s'  # xzl: simpler
FORMAT = '%(levelname)8s {%(module)s:%(lineno)d} %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

coloredlogs.install(fmt=FORMAT, level='DEBUG', logger=logger)

class ImageDoesNotExistException(Exception):
    """Image does not exist"""
    pass


@dataclass
class FrameResults():
    #name: str
    frame_id: int
    elements: typing.Any  # from grpc resp
    high_confidence: float = -1  # highest confidence among all bbs on this frame
    n_bboxes: int = 0

_CAMERA_STORAGE = {'jetson': {'host': CAMERA_CHANNEL_ADDRESS}}

# a snapshot of current query progress...
@dataclass
class QueryProgressSnapshot():
    ts: float # since epoch
    framestates: typing.Dict[int, str]
    n_frames_sent_cam: int = 0
    n_frames_processed_cam: int = 0
    n_frames_total_cam: int = 0
    n_frames_recv_yolo: int = 0
    n_frames_processed_yolo: int = 0

# the info of a query kept by the cloud
@dataclass
class QueryInfoCloud():
    qid: int
    status_cloud: str
    status_cam: str
    target_class: str
    results: typing.List[FrameResults]
    # there seems little point of keeping the whole history...
    # bound the len of the list
    progress_snapshots: typing.List[QueryProgressSnapshot] # higher idx == more recent snapshot of progress

    video_name: str = ""
    ts_comp_cam: float = 0

    # the scratch space for yolo stats. will be copied to snapshot
    n_frames_recv_yolo: int = 0
    n_frames_processed_yolo: int = 0

    n_bytes_sent_cam: int = 0
    cam_mem_usage_percent: float = 0

    # deprecated -- use progress_snapshots instead
    n_frames_sent_cam: int = 0
    n_frames_processed_cam: int = 0
    n_frames_total_cam: int = 0


# to be consistent with cam_cloud_pb2.VideoMetadata
@dataclass
class VideoInfo():
    video_name: str
    n_frames: int
    n_missing_frames: int
    fps: int
    duration: float
    frame_id_min: int
    frame_id_max: int
    start: float = 0
    end: float = 0

# metadata of queries ever executed
# the_queries = {} # qid:query_metadata
the_queries: typing.Dict[int, QueryInfoCloud] = {}
the_queries_lock = threading.Lock()
logger.info('----------------the queries init-----------------')
#traceback.print_stack()  # dbg


# a query's results -- as a collection (simple) in the_queries[qid]?
# web thread can have its own way to org/present/render it

# the_query_results = {} # qid: a priority queyue
# $the_query_results_lock = threading.Lock()

# given query id
# return: query info, None if no query exists
def query_status(qid):
    if (qid >= len(the_queries)):
        return None
    return the_queries[qid]

# clean up a subdir (preview or results)
# prefix=results,preview (full path); subdir=qid,video_name
def _CleanStoredFrames(prefix:str, subdir:str):
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    path = os.path.join(prefix, subdir)
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"cleaned {path}")
        elif not os.path.exists(path):
            os.mkdir(path)
        else:
            logger.error(f'{path} exists but not a dir? abort')
            sys.exit(1)
    except Exception as err:
        logger.error(err)

# clean up any stale results for an upcoming query (qid)
def clean_results_frames(qid:int):
    _CleanStoredFrames(CFG_RESULT_PATH, f'{qid}')

def clean_preview_frames(video_name:str):
    _CleanStoredFrames(CFG_PREVIEW_PATH, video_name)

# deprecated. see VideoStore
# save a result frame. draw bboxes
# @img is the image.data loaded from jpg file and passed over grpc
# return: full path of the stored frame
def _SaveResultFrame(request, elements) -> str:
    qid = request.qid
    img_data = request.image.data
    raw_img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)

    # render bboxes
    for idx, ele in enumerate(elements):
        x1, y1, x2, y2 = int(ele.x1), int(ele.y1), int(ele.x2), int(ele.y2)
        cv2.rectangle(raw_img, (x1, y1), (x2, y2),
                      (0, 255, 0), 3)
        # print("draw--->", x1, y1, x2, y2)

    frame_name = f'{request.frame_id}'
    if not (frame_name.endswith('.jpg') or frame_name.endswith('.JPG')):
        frame_name += '.jpg'
    path = os.path.join(CFG_RESULT_PATH, f'{qid}', frame_name)

    try:
        cv2.imwrite(path, raw_img)
    except Exception as err:
        logger.error(err)
        path = None

    return path

# sample @n_frames from the given video from the cam. save them locally
# return: a list of frame ids, in integers
def download_video_preview_frames_0(v:cam_cloud_pb2.VideoMetadata, n_frames:int) -> typing.List[int]:
    delta = int((v.frame_id_max - v.frame_id_min) / n_frames)
    assert(delta > 10)

    fl = []

    for i in range(n_frames):
        frameid = v.frame_id_min + delta * i
        try:
            print(f"get preview frame id {frameid}")
            img = get_video_frame(v.video_name, frameid)
            save_video_frame(v.video_name, frameid, CFG_PREVIEW_PATH, True)
            logger.info(f'saved preview frame size is {len(img.data)}')
            fl.append(frameid)
        except Exception as err:
            logger.error(err)
            sys.exit(1)

    return fl


# sample @n_frames from the given video from the cam. save them locally
# return: a list of frame ids, in integers
# not cleaning existing local cache
# using VideoStore interface
def download_video_preview_frames(v:cam_cloud_pb2.VideoMetadata, n_frames:int, res=[128,128]) -> typing.List[int]:
    delta = int((v.frame_id_max - v.frame_id_min) / n_frames)
    assert(delta > 10)

    vs = the_videolib_preview.GetVideoStore(v.video_name)

    fl = []

    for i in range(n_frames):
        frameid = v.frame_id_min + delta * i
        try:
            print(f"get preview frame id {frameid}")
            img = get_video_frame(v.video_name, frameid)
            print(f"got preview frame id {frameid}")
            vs.StoreFrame(frameid, img, res)
            logger.info(f'saved preview frame size is {len(img.data)}')
            fl.append(frameid)
        except Exception as err:
            logger.error(err)
            sys.exit(1)

    return fl

# forget query res in mem. to be invoked by web
def query_cleanup(qid):
    return


def _query_control(qid, command) -> str:
    if command not in CFG_QUERY_CMDS:
        logger.error("unknown cmd %s" % command)
        return "FAILED"

    try:
        camera_channel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
        camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
        resp = camera_stub.ControlQuery(
            cam_cloud_pb2.ControlQueryRequest(qid=qid, command=command)
        )
        camera_channel.close()
        return resp.msg

    except Exception as err:
        logger.warning(err)


def query_abort(qid):
    msg = _query_control(qid, "ABORT")
    return msg


def query_resume():
    msg = _query_control(-1, "RESUME")
    return msg


def query_pause():
    msg = _query_control(-1, "PAUSE")
    return msg


def query_reset():
    msg = _query_control(-1, "RESET")
    return msg

# only pull stats (but not framestates) from the cam.
def query_progress(qid: int) -> QueryInfoCloud:
    try:
        camera_channel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
        camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
        resp = camera_stub.GetStats(
            cam_cloud_pb2.ControlQueryRequest(qid=qid)  # overloaded
        )
        camera_channel.close()
    except Exception as err:
        logger.warning(err)
        sys.exit(1)

        # also update cloud's query metadata
    with the_queries_lock:
        the_queries[resp.qid].n_frames_processed_cam = resp.n_frames_processed
        the_queries[resp.qid].n_frames_sent_cam = resp.n_frames_sent
        the_queries[resp.qid].n_bytes_sent_cam = resp.n_bytes_sent
        the_queries[resp.qid].n_frames_total_cam = resp.n_frames_total
        the_queries[resp.qid].cam_mem_usage_percent = resp.mem_usage_percent
        the_queries[resp.qid].status_cam = resp.status
        the_queries[resp.qid].ts_comp_cam = resp.ts_comp
        return copy.deepcopy(the_queries[resp.qid])

    '''
    return QueryInfoCloud(
            qid = resp.qid,
            video_name = resp.video_name, 
            n_frames_processed_cam= resp.n_frames_processed,
            n_frames_total_cam= resp.n_frames_total,
            n_frames_sent_cam= resp.n_frames_sent,
            status_cam= resp.status,
            ts_comp_cam= resp.ts_comp        
        )
    '''

    '''
    return {'qid' : resp.qid, 
            'video_name' : resp.video_name, 
            'n_frames_processed_cam' : resp.n_frames_processed,
            'n_frames_total_cam' : resp.n_frames_total,
            'n_frames_sent_cam' : resp.n_frames_sent,
            'status_cam' : resp.status,
            'ts_comp_cam' : resp.ts_comp}
    '''


# pull the framestates from cam.
# gen a progress snapshot. append to the list of snapshots
# do NOT call with holding the_queries_lock
# return a deepcopy of the snapshot created, or None if failed
def create_query_progress_snapshot(qid: int) -> QueryProgressSnapshot:
    with the_queries_lock:
        if not qid in the_queries:
            return None

    try:
        camera_channel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
        camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
        resp = camera_stub.GetQueryFrameStates(
            cam_cloud_pb2.ControlQueryRequest(qid=qid)  # overloaded
        )
        camera_channel.close()
    except Exception as err:
        logger.warning("create_query_progress_snapshot")
        sys.exit(1)

    if not resp:
        return None

    fm_states = resp.frame_states

    if resp.frame_states == "": # nothing. query may have been reset
        return None

    fs = FrameMapToFrameStates(resp)

    with the_queries_lock:
        q = the_queries[qid]
        n_frames_recv_yolo = q.n_frames_recv_yolo
        n_frames_processed_yolo = q.n_frames_processed_yolo
        for f in q.results:
            #fs[f.frame_id] = 'r'
            # a bit hack. storing confidence as string. should not break other states
            assert(f.high_confidence < 1.0)
            fs[f.frame_id] = f"{f.high_confidence}"

    ps = QueryProgressSnapshot(
        ts = time.time(), # the server's ts
        framestates = fs,
        n_frames_processed_cam = fm_states.count('p'),
        n_frames_sent_cam = fm_states.count('s'),
        n_frames_total_cam = len(fm_states),
        n_frames_recv_yolo = n_frames_recv_yolo,
        n_frames_processed_yolo = n_frames_processed_yolo
    )

    ps1 = copy.deepcopy(ps)
    with the_queries_lock:
        the_queries[qid].progress_snapshots.append(ps)
        # fix -- keep the snapshot queue bounded
        if len(the_queries[qid].progress_snapshots) > 3:
            the_queries[qid].progress_snapshots.pop(0)

    return ps1


# do NOT call with holding the_queries_lock
# return a deepcopy of the most recent snapshot
def get_latest_query_progress(qid: int) -> QueryProgressSnapshot:
    with the_queries_lock:
        try:
            return copy.deepcopy(the_queries[qid].progress_snapshots[-1])
        except Exception as err:
            #print(err)
            logger.error(f"failed to get most recent prog. qid {qid}", err)
            return None


# return: a *single* query's results. frames sorted by score in descending order
# return deepcopy
# will NOT contact camera
# {name: XXX, n_bboxes: XXX, high_confidence: XXX, [elements...]}
# see SubmitFrame()            
def query_results(qid, to_sort=True) -> typing.List[FrameResults]:
    res: typing.List[FrameResults] = []

    with the_queries_lock:
        if not qid in the_queries:
            return None
        res = copy.deepcopy(the_queries[qid].results)

    if to_sort:
        res.sort(key=lambda s: -s.high_confidence)  # descending order
    return res


# nonblocking. will auto fill request.qid. return: qid. -1 if failed
# on failure: perhaps should negotiation a qid
def query_submit(video_name: str, op_names: typing.List[str], crop,
                 target_class: str, frameskip:int = 1) -> int:
    global the_queries

    # auto assign qid
    with the_queries_lock:
        qid = len(the_queries)

    # init the query ... must do it before sending query out
    with the_queries_lock:
        the_queries[qid] = QueryInfoCloud(
            qid=qid,
            status_cloud='SUBMITTED',
            status_cam='UNKNOWN',
            video_name=video_name,
            target_class=target_class,
            results=[],
            progress_snapshots = []
        )

    # clean_results_frames(qid)

    request = cam_cloud_pb2.QueryRequest(video_name=video_name,
                                         op_names=op_names,
                                         crop=crop,
                                         target_class=target_class,
                                         qid=qid, frameskip=frameskip)

    try:
        camera_channel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
        camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
        resp = camera_stub.SubmitQuery(request)
        if not resp.msg.startswith('OK'):
            # "FAIL: qid 1 exists. suggested=2"
            tokens = resp.msg.split('suggested=')
            if len(tokens) > 1:
                newqid = int(tokens[1])
                with the_queries_lock:
                    the_queries[newqid] = the_queries.pop(qid)
                    the_queries[newqid].qid = newqid
                logger.warning(f"resubmit query with qid {newqid}")
                request.qid = newqid
                resp = camera_stub.SubmitQuery(request)
                # no more try
        camera_channel.close()
    except Exception as err:
        logger.error('failed to submit a query', err)
        return -1

    logger.info("submitted a query, video_name %s qid %d. camera says %s" % (video_name, qid, resp.msg))
    logger.info(f"the_queries {id(the_queries)}")

    if not resp.msg.startswith('OK'):
        return -1
    else:
        # get the video store & clean any frames there
        the_videolib_results.AddVideoStore(f'{qid}').CleanStoredFrames()
        return qid

# deprecated
# will return a list of videos. to be called by web server
def list_videos_cam() -> cam_cloud_pb2.VideoList:
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    resp = camStub.ListVideos(empty_pb2.Empty())

    # note: won't print out fields w value 0        
    print("get video list from camera: ---> ")
    return resp.videos


def list_videos() -> typing.List[VideoInfo]:
    global the_videolib_preview

    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    resp = camStub.ListVideos(empty_pb2.Empty())

    if len(resp.videos) == 0:
        return None

    vl = []
    for v in resp.videos:
        vl.append(VideoInfo(
            video_name=v.video_name,
            n_frames=v.n_frames,
            n_missing_frames=v.n_missing_frames,
            fps=v.fps,
            start=v.start,
            end=v.end,
            duration=v.duration,
            frame_id_min=v.frame_id_min,
            frame_id_max=v.frame_id_max
        ))

    the_videolib_preview.AddVideoStore(v.video_name)

    return vl

def cam_specs() -> str:
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    resp = camStub.GetCamSpecs(empty_pb2.Empty())

    return resp.msg

# promote a range of frames
def promote_frames(qid: int, fid_start: int,
                   fid_end: int, is_promote:bool=True) -> str:
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)

    frame_ids = [x for x in range(fid_start, fid_end)]
    map = cam_cloud_pb2.FrameMap(frame_ids=frame_ids, frame_states="")

    if is_promote:
        resp = camStub.PromoteFrames(map)
    else:
        resp = camStub.DemoteFrames(map)
    return resp.msg

def demote_frames(qid: int, fid_start: int, fid_end: int) -> str:
    return promote_frames(qid, fid_start, fid_end, is_promote=False)

# return a list of queries. to be called by web server
# see query_submit for query metadata
def list_queries_cloud() -> typing.List[QueryInfoCloud]:
    qid_list = []
    with the_queries_lock:
        for qid, _ in the_queries.items():
            qid_list.append(qid)

    for qid in qid_list:
        assert (qid >= 0)
        query_progress(qid)

    query_list: typing.List[QueryInfoCloud] = []
    with the_queries_lock:
        for id, q in the_queries.items():
            query_list.append(copy.deepcopy(q))
        return query_list


def get_video_frame(video_name: str, frame_id: int) -> common_pb2.Image:
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    resp = camStub.GetVideoFrame(cam_cloud_pb2.GetVideoFrameRequest(video_name=video_name, frame_id=frame_id))

    return resp

# get a video frame from cam, save to a local file under prefix/video_name/frame_id.jpg
# return: full frame path
def save_video_frame(video_name: str, frame_id: int, prefix: str, thumbnail: bool = False) -> str:
    img = get_video_frame(video_name = video_name, frame_id = frame_id)
    videodir = os.path.join(prefix, video_name)
    if not os.path.isdir(videodir):
        try:
            os.mkdir(videodir)
        except Exception as err:
            print(err)
            sys.exit(1)

    frame_path = os.path.join(videodir, f'{frame_id}.jpg')
    thumbnail_path = os.path.join(videodir, f'{frame_id}.thumbnail.jpg')
    try:
        # don't know how many leading 0s...
        with open(frame_path, 'wb') as f:
            f.write(img.data)
        im = PIL.Image.open(frame_path)
        im.thumbnail([128, 128])  # 128x128 thumbnail
        im.save(thumbnail_path)
    except Exception as err:
        logger.error("cannot write to file", err)
    else:
        logger.info(f"written to {frame_path}")
    return frame_path


class DivaGRPCServer(server_diva_pb2_grpc.server_divaServicer):
    """
    Implement server_divaServicer of gRPC
    """

    def __init__(self):
        server_diva_pb2_grpc.server_divaServicer.__init__(self)
        self.clog = ClockLog(5)  # max every 5 sec 

    # xzl: proxy req ("get metadata of all stored videos") from client to cam
    def get_videos(self, request, context):
        resp = []

        for _, val in _CAMERA_STORAGE.items():
            camera_channel = grpc.insecure_channel(f"{val['host']}")
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
            req = camera_stub.get_videos(empty_pb2.Empty())
            for v in req.videos:
                resp.append(v)

            camera_channel.close()

        return common_pb2.get_videos_resp(videos=resp)

    # xzl: proxy req ("get res of a previous query") from client to cam 
    def get_video(self, request, context):
        if request.camera and request.camera.name in _CAMERA_STORAGE:
            val = _CAMERA_STORAGE[request.camera.name]
            camera_channel = grpc.insecure_channel(f"{val['host']}")
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)

            _camera = common_pb2.Camera(name=request.camera.name,
                                        address=request.camera.address)

            req = camera_stub.get_video(
                common_pb2.VideoRequest(timestamp=request.timestamp,
                                        offset=request.offset,
                                        video_name=request.video_name,
                                        object_name=request.object_name,
                                        camera=_camera))
            camera_channel.close()
        else:
            raise Exception("Error....")

        return req

    # invoke yolo over gRPC in a sync fashion
    # return a list of textual bb results. 
    # leaving rendering to web thread
    def InvokeYolo(self, img_msg, target_class, threshold=0.3):
        channel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
        stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)

        req = det_yolov3_pb2.DetectionRequest(
            image=img_msg,
            name='',
            threshold=threshold,
            targets=[target_class])

        resp = stub.Detect(req)
        # print ('client received: ', resp)    
        return resp.elements

    # recv a frame submitted from cam. parse it. 
    def SubmitFrame(self, request, context):
        global the_queries

        '''
        logger.info("got a frame. id %d frame %d bytes. queries %d"
                    %(request.frame_id, len(request.image.data), id(the_queries)))

        with the_queries_lock:
            if (len(the_queries) == 0):
                logger.error(f"submitframe1 bug??? - no queries  {len(the_queries)}")
            else: 
                logger.info(f"++++++ submitframe1 ok - {len(the_queries)}")
        '''

        qid = request.qid
        with the_queries_lock:
            try:
                the_queries[qid].n_frames_sent_cam += 1
                target_class = the_queries[qid].target_class
            except Exception as err:
                logger.error(f"SubmitFrame: bug - cloud has no query  {len(the_queries)}")
                logger.error(f"queries:{id(the_queries)} {the_queries.keys()}")
                sys.exit(1)

        #logger.info("======= invoking yolo.....")
        elements = self.InvokeYolo(img_msg=request.image, target_class=target_class)
        #logger.info("+++++++  yolo returns .....")

        if len(elements) > 0:
            the_videolib_results.GetVideoStore(f'{qid}').StoreFrameBBoxes(request.frame_id, request.image, elements)
            #saved = _SaveResultFrame(request, elements) # save the frame locally
            #logger.info("res frame saved to " + saved)

            # keep metadata in mem
            frame_res = FrameResults(frame_id=request.frame_id,
                                     n_bboxes=len(elements), elements=elements)

            for idx, ele in enumerate(elements):
                if ele.confidence > frame_res.high_confidence:
                    frame_res.high_confidence = ele.confidence

            with the_queries_lock:
                the_queries[qid].results.append(frame_res)

        # update stats
        n_proc = 0
        with the_queries_lock:
            the_queries[qid].n_frames_processed_yolo += 1
            n_proc = the_queries[qid].n_frames_processed_yolo
            if len(elements) > 0:
                the_queries[qid].n_frames_recv_yolo += 1

        '''
        with the_queries_lock:
            if (len(the_queries) == 0):
                logger.error(f"submitframe2 bug??? - no queries  {len(the_queries)}")
        '''

        return cam_cloud_pb2.StrMsg(msg=f'OK frame {n_proc}')

    # old
    # xzl: proxy req ("process video footage") from client to cam. blocking until completion
    def process_video(self, request, context):
        logger.info("process_video")

        req_camera = request.camera
        new_req_payload = common_pb2.VideoRequest(
            timestamp=request.timestamp,
            offset=request.offset,
            video_name=request.video_name,
            object_name=request.object_name,
            camera=req_camera)

        try:
            camera_channel = grpc.insecure_channel(req_camera.address)
            camera_stub = cam_cloud_pb2_grpc.DivaCameraStub(camera_channel)
            camera_stub.process_video(new_req_payload)
            camera_channel.close()
        except Exception as err:
            logger.warning(err)

        return empty_pb2.Empty()


def thread_progress_snapshot(interval_sec:int):
    logger.warning(f"------------ progress snapshot start running id = {threading.get_ident()}")

    # XXX stop track completed queries
    while True:
        with the_queries_lock:
            qids = the_queries.keys()
        for qid in qids:
            create_query_progress_snapshot(qid)
            logger.warning(f"took a prog snapshot for qid {qid}")
        time.sleep(interval_sec)

# https://raspberrypi.stackexchange.com/questions/22005/how-to-prevent-python-script-from-running-more-than-once
the_instance_lock = None

def grpc_serve():
    global the_instance_lock
    #global the_videolib_results, the_videolib_preview

    #traceback.print_stack()  # dbg

    # move these to variables?
    #the_videolib_results = VideoLib(CFG_RESULT_PATH)
    #the_videolib_preview = VideoLib(CFG_PREVIEW_PATH)

    try:
        the_instance_lock = zc.lockfile.LockFile('/tmp/diva-cloud')
        logger.debug("grabbed diva lock")
    except zc.lockfile.LockError:
        logger.error("cannot lock file. are we running multiple instances?")
        sys.exit(1)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_servicer = DivaGRPCServer()
    server_diva_pb2_grpc.add_server_divaServicer_to_server(
        diva_servicer, server)
    server.add_insecure_port(f'[::]:{DIVA_CHANNEL_PORT}')
    server.start()
    logger.info("--------- cloud GRPC server is runing --------------------")

    # traceback.print_stack() # dbg

    '''
    tps = threading.Thread(target=thread_progress_snapshot, args=(1,))
    tps.start()
    assert(tps.is_alive())
        
    return server, tps
    '''

    return server, None

# xzl: unused?
def draw_box(img, x1, y1, x2, y2):
    rw = float(img.shape[1]) / DET_SIZE
    rh = float(img.shape[0]) / DET_SIZE
    x1, x2 = int(x1 * rw), int(x2 * rw)
    y1, y2 = int(y1 * rh), int(y2 * rh)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


# xzl: unused?
def deploy_operator_on_camera(operator_path: str,
                              camStub: cam_cloud_pb2_grpc.DivaCameraStub):
    f = open(operator_path, 'rb')
    op_data = f.read(CHUNK_SIZE)
    while op_data != b"":
        response = camStub.DeployOp(cam_cloud_pb2.Chunk(data=op_data))
        if response.msg != 'OK':
            logger.warning('DIVA deploy op fails!!!')
            return
        op_data = f.read(CHUNK_SIZE)
    f.close()
    camStub.DeployOpNotify(
        cam_cloud_pb2.DeployOpRequest(name='random', crop=CROP_SPEC))

'''
# xzl: unused?
def process_frame(img_name: str, img_data, det_res):
    """
    Take two arguments: one is the image frame data and the other is the
    response from YOLO agent. Draw boxes around the object of interest.
    """
    # print ('client received: ' + det_res)
    if not det_res or len(det_res) == 0:
        return

    res_items = det_res.split('|')
    res_items = [[float(y) for y in x.split(',')] for x in res_items]
    res_items = list(filter(lambda z: z[0] > YOLO_SCORE_THRE, res_items))

    if not res_items or len(res_items) <= 0:
        return

    img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
    for item in res_items:
        x1, y1, x2, y2 = int(item[1]), int(item[2]), int(item[3]), int(item[4])
        img = draw_box(img, x1, y1, x2, y2)

    img_fname = os.path.join(RESULT_IMAGE_PATH, img_name)
    cv2.imwrite(img_fname, img)


# xzl: unused?
def runDiva():
    # Init the communication channels to camera and yolo
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    yoloChannel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
    yoloStub = det_yolov3_pb2_grpc.DetYOLOv3Stub(yoloChannel)

    response = camStub.InitDiva(
        cam_cloud_pb2.InitDivaRequest(img_path=IMAGE_PATH))
    if response.msg != INIT_DIVA_SUCCESS:
        print('DIVA init fails!!!')
        return

    deploy_operator_on_camera(OP_FNAME_PATH, camStub)

    time.sleep(5)
    pos_num = 0
    clog = ClockLog(5)

    for _ in range(10000):
        time.sleep(0.1)  # emulate network
        clog.log('Retrieved pos num: %d' % (pos_num))

        response = camStub.GetFrame(cam_cloud_pb2.GetFrameRequest(name='echo'))
        if not response.name:
            print('no frame returned...')
            continue

        img_name = response.name
        img_data = response.data
        detected_objects = yoloStub.DetFrame(
            det_yolov3_pb2.DetFrameRequest(data=img_data,
                                           name=str(img_name),
                                           cls=OBJECT_OF_INTEREST))

        det_res = detected_objects.res
        process_frame(img_name, img_data, det_res)

        pos_num += 1

    camChannel.close()
    yoloChannel.close()
'''

# test funcs with cam
def test_cam():
    req = cam_cloud_pb2.QueryRequest()

    qid = query_submit(video_name='chaweng-1_10FPS',
                       op_names=['random'], crop='-1,-1,-1,-1', target_class='motorbike')

    time.sleep(10)

    with the_queries_lock:
        n_frames_sent_cam = the_queries[qid].n_frames_sent_cam
    logger.info("received %d frames from cam. now pause query." % n_frames_sent_cam)

    query_pause(qid)
    time.sleep(10)

    print(query_progress(qid))

    query_resume()
    time.sleep(10)

    with the_queries_lock:
        n_frames_sent_cam = the_queries[qid].n_frames_sent_cam
    logger.info("received %d frames from cam." % n_frames_sent_cam)

    print(list_queries_cloud)


# invoke yolo over grpc 
# cf: testYOLO() in mengwei's main_cloud.py
def test_yolo():
    channel = grpc.insecure_channel(YOLO_CHANNEL_ADDRESS)
    stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)
    f = open('/data/hybridvs-demo/tensorflow-yolov3/data/demo_data/dog.jpg', 'rb')

    _img = common_pb2.Image(data=f.read(),
                            height=0,
                            width=0,
                            channel=0)
    req = det_yolov3_pb2.DetectionRequest(
        image=_img,
        name='dog.jpg',
        threshold=0.3,
        targets=['dog'])
    resp = stub.Detect(req)

    print('client received: ', resp)

    '''
    exist_target = False

    temp_score_map = {}

    # xzl: a resp from server: a list of BBs
    # draw bbox on the image    
    frame = cv2.imread(img)
     
    # there may be multiple objs
    for idx, ele in enumerate(resp.elements):
        if ele.class_name != target_class:
            continue

        x1, y1, x2, y2 = ele.x1, ele.y1, ele.x2, ele.y2

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 3)
        temp_score_map[str(idx)] = ele.confidence
        exist_target = True

    # xzl: save the (annotated image) on local disk .. for web serving later
    if exist_target:
        img_path = os.path.join(
            STATIC_FOLDER, *[
                video_folder_name, target_class, 'images',
                f'{counter}.jpg'
            ])
        cv2.imwrite(img_path, frame)
        score_obj[str(counter)] = temp_score_map
    '''


def test_list_videos():
    '''
    camChannel = grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS)
    camStub = cam_cloud_pb2_grpc.DivaCameraStub(camChannel)
    resp = camStub.ListVideos(empty_pb2.Empty())
         
    # note: won't print out fields w value 0        
    print("get video list from camera:", resp)
    '''
    #videos = list_videos_cam()

    videos = list_videos()
    logger.warning(f'got {len(videos)} videos')

    '''
    for r in videos:
        print(r.video_name, r.n_missing_frames, r.frame_id_min, r.frame_id_max)
        for _ in range(3):
            # frameid = random.randint(r.frame_id_min, r.frame_id_max)
            frameid = int((r.frame_id_min + r.frame_id_max) / 2)
            print(f"Get frame id {frameid}")
            img = get_video_frame(r.video_name, frameid)
            print(f'frame size is {len(img.data)}')
            # get  again. save to local disk this time
            save_video_frame(r.video_name, frameid, '/tmp')
    '''

    # test save --- old
    '''
    for v in videos:
        clean_preview_frames(v.video_name)
        download_video_preview_frames(v, 5)
    '''

    for v in videos:
        the_videolib_preview.AddVideoStore(v.video_name).CleanStoredFrames()
        download_video_preview_frames(v, 5)

def console():
    while True:
        msg = f'''
        l:list_videos_cam L:list_queries_cloud q:send a sample query
        R: reset
        pX - query_progress for qid X
        aX - pause query qid X
        rX - resume query qid X
        '''
        ss = input(msg)
        s = ss.split()

        if len(s) < 1:
            continue
        if s[0] == 'lv':
            test_list_videos()
        elif s[0] == 'lq':
            resp = list_queries_cloud()
            print(f"{len(resp)} queries found:")
            if len(resp) == 0:
                continue
            # print header
            '''
            keys = asdict(resp[0]).keys()
            for k in keys:
                print(k, end=" ")
            print()
            '''
            for info in resp:
                '''
                vs = asdict(info).values()
                for v in vs:
                    print(v, end=" ")
                '''

                if (len(info.progress_snapshots) > 0):
                    last = info.progress_snapshots[-1]
                    sdic = asdict(last)
                    sdic['framestates'] = "removed"
                    print("most recent prog", sdic)

                dic = asdict(info)
                dic['results'] = len(dic['results']) # overwrite it to be # of results
                dic['progress_snapshots'] = len(dic['progress_snapshots'])
                for k, v in dic.items():
                    print(f"{k}={v}", end=" ")
                print()
            #print(resp)
        elif s[0] == 'progress':
            if len(s) < 2:
                print("need qid")
                continue
            resp = query_progress(int(s[1]))
            print(resp)
        elif s[0] == 'pause':
            resp = query_pause()
            print(resp)
        elif s[0] == 'results':
            if len(s) < 2:
                print("need qid")
                continue
            resp = query_results(int(s[1]))
            if not resp:
                print("no results. bad qid?")
            else:
                print("query results:")
                for r in resp:
                    print(f"{r.name} {r.n_bboxes} {r.high_confidence}")
        elif s[0] == 'resume':
            resp = query_resume()
            print(resp)
        elif s[0] == 'query':
            resp = query_submit(video_name='chaweng-1_10FPS',
                                op_names=['random','random'],
                                crop='-1,-1,-1,-1',
                                target_class='motorbike', frameskip=10)
            print('qid:', resp)
        elif s[0] == 'promo':
            resp = promote_frames(-1, 80000, 85000)
            print('resp:', resp)
        elif s[0] == 'demo':
            resp = demote_frames(-1, 80000, 85000)
            print('resp:', resp)
        elif s[0] == 'reset':
            resp = query_reset()
            print(resp)
        elif s[0] == 'yolo':
            test_yolo()
        elif s[0] == 'prog':
            s = create_query_progress_snapshot(qid=0)
            sdic = asdict(s)
            sdic['framestates'] = "removed"
            print(sdic)
        elif s[0] == 'log': # test logging facility
            logger.debug("This is a debug log")
            logger.info("This is an info log")
            logger.critical("This is critical")
            logger.error("An error occurred")
        else:
            print("unknown cmd")


if __name__ == '__main__':

    logger.info("test cloud controller")

    _server, _tps = grpc_serve()
    # _server.wait_for_termination() # xzl: won't return. dont need. 

    # test_cam()
    # test_yolo()
    # test_list_videos()

    console()

    try:
        while True:
            time.sleep(60 * 60 * 24)

    except KeyboardInterrupt:
        logger.info('cloud receiver stops!!!')
        SHUTDOWN_SIGNAL.set()
        _server.stop(0)
    except Exception as err:
        logger.warning(err)
        SHUTDOWN_SIGNAL.set()
        _server.stop(1)

    # runDiva()
