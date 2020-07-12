#!/usr/bin/env python3.7

'''
the main func of the cam service

server example, callbacks, cf:
https://docs.bokeh.org/en/latest/docs/user_guide/server.html

'''
#import queue
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # not working?

import sys
import json
#import cv2
import threading
import heapq
import copy
import random

from google.protobuf.timestamp_pb2 import *
from google.protobuf.duration_pb2 import *


# from threading import Thread, Event, Lock
from concurrent import futures
import logging, coloredlogs
#from queue import PriorityQueue

import psutil
import platform

import grpc

import keras
import keras.backend as K

import cam_cloud_pb2_grpc
#import det_yolov3_pb2
#import det_yolov3_pb2_grpc
import common_pb2
import cam_cloud_pb2
import server_diva_pb2_grpc # for submitting frames

import videostore
from google.protobuf import empty_pb2

# since 3.7
# cf https://realpython.com/python-data-classes/#basic-data-classes
# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html#built-in-types
from dataclasses import dataclass, field
import typing

import zc.lockfile # detect& avoid multiple instances

from variables import CAMERA_CHANNEL_PORT
from variables import DIVA_CHANNEL_ADDRESS # for submitting frames
from variables import  * # xzl
from videostore import VideoStore

from camera.camera_constants import VIDEO_FOLDER, _HOST, _PORT, STATIC_FOLDER, WEB_APP_DNS
from camera.camera_constants import _NAME, _ADDRESS, YOLO_CHANNEL_ADDRESS

# Mengwei's util
from .mengwei_util import *

# -- from Mengwei's cloud side --- #
# the "video_name" in a query will be used to locate subdir
# the_img_dirprefix = '/media/teddyxu/WD-4TB-NEW/hybridvs_data/YOLO-RES-720P/jpg/chaweng-1_10FPS/'
the_img_dirprefix = 'hybridvs_data/YOLO-RES-720P/jpg'
the_csv_dir = '/media/teddyxu/WD-4TB-NEW/hybridvs_data/YOLO-RES-720P/out/chaweng-1_10FPS.csv'

# the default op fname if not specified
the_op_fname = '/media/teddyxu/WD-4TB/hybridvs_data/YOLO-RES-720P/exp/chaweng/models/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'


CHUNK_SIZE = 1024 * 100
OP_BATCH_SZ = 16
# the_op_dir = './result/ops'
the_op_dir = './ops'

#FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
#FORMAT = '%(levelname)8s %(thread)d %(threadName)s %(lineno)d %(message)s'
FORMAT = '%(levelname)8s {%(module)s:%(lineno)d} %(threadName)s %(message)s'
#FORMAT = '{%(module)s:%(lineno)d} %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

coloredlogs.install(fmt=FORMAT, level='DEBUG', logger=logger)
#coloredlogs.install(level='DEBUG', logger=logger)

# FIXME (xzl: unused?)
# OP_FNAME_PATH = '/home/bryanjw01/workspace/test_data/source/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'

'''
@dataclass
class FrameMap():
    frame_ids: typing.List[int]
    frame_states: str
    # see cam_cloud.proto
'''
@dataclass
class FrameInfo():
    #name: str
    video_name : str
    frame_id : int
    cam_score: float
    #local_path: str
    qid: int

# for tracking the current query
@dataclass
class QueryInfo():
    # each frame_id: . init, r ranked s sent
    # will compress before sending to gRPC
    op_names: typing.List[str]
    op_fnames: typing.List[str]
    framestates: typing.Dict[int, str]
    workqueue : typing.List[FrameInfo] # maybe put a separate lock over this
    backqueue : typing.List[FrameInfo] # maybe put a separate lock over this
    op_index:int # the current op index
    qid: int = -1    # current qid
    crop: typing.List[int] = field(default_factory = lambda: [-1,-1,-1,-1])
    img_dir: str = ""
    video_name: str = ""
    # run_imgs : typing.List[str] = field(default_factory = lambda: []) # maybe put a separate lock over this


# todo: combine it to queryinfo
@dataclass
class QueryStat():
    qid : int = -1    # current qid
    video_name : str = ""
#    op_name : str = ""
    crop : typing.List[int] = field(default_factory = lambda: [-1,-1,-1,-1])
    status : str = "UNKNOWN"
    n_frames_processed : int = 0
    n_frames_sent : int = 0
    n_frames_total : int = 0

# for the current query: all frames (metadata such as fullpaths etc) to be processed. 
# workers will pull from here
the_query_lock = threading.Lock()
the_query = None # QueryInfo()

# queries ever executed
the_stats: typing.Dict[int, QueryStat] = {}
the_stats_lock = threading.Lock()

the_uploader_stop_req = 0

# TODO: use Python3's PriorityQueue
# buf for outgoing frames. only holding image metadata (path, score, etc.)
# out of which msgs for outgoing frames (including metadata) will be assembled 
the_send_buf: typing.List[FrameInfo] = []
the_buf_lock = threading.Lock()
the_cv = threading.Condition(the_buf_lock)

'''
ev_uploader_stop_req = threading.`Event`()
ev_uploader_stop_req.clear()
ev_uploader_stop_done = threading.Event()
ev_uploader_stop_done.clear()
ev_uploader_resume_req = threading.Event()

ev_worker_stop_req = threading.Event()
ev_worker_stop_req.clear()
ev_worker_stop_done = threading.Event()
'''

#PQueue = PriorityQueue()

'''
# caller must hold the query lock
def frameid_to_abspath(fid: int) -> str:
    assert(the_query.imgdir != "")
    return os
'''

def FrameMapToFrameStates(fm:cam_cloud_pb2.FrameMap) -> typing.Dict[int,str]:
    fids = fm.frame_ids
    states = fm.frame_states
    dic = {}

    assert(len(fids) == len(states))
    for idx, fid in enumerate(fids):
        dic[fid] = states[idx]
    return dic

def FrameStatesToFrameMap(fs:typing.Dict[int,str]) ->cam_cloud_pb2.FrameMap:
    # fs = {5: 'a', 3: 'b', 4: 'c'}
    # s = [(5, 'a'), (3, 'b'), (4, 'c')]
    s = [(fid, st) for fid, st in fs.items()]
    #sorted(s, key=lambda x: x[0])
    s.sort(key=lambda x: x[0])
    fids = [x[0] for x in s]
    states = [x[1] for x in s]
    return cam_cloud_pb2.FrameMap(frame_ids = fids, frame_states = ''.join(states))

the_video_stores: typing.Dict[str, VideoStore] = {}

# will terminate after finishing up the current query
class OP_WORKER(threading.Thread):
    # xzl: read from disk a batch of images. 
    # @imgs: a list of img filenames
    # @return: an array of image data
    def read_images(self, imgs, H, W, crop=(-1,-1,-1,-1)):
        frames = np.zeros((len(imgs), H, W, 3), dtype='float32')
        for i, img in enumerate(imgs):
            frame = cv2.imread(img)
            if crop[0] > 0:
                frame = frame[crop[0]:crop[2],crop[1]:crop[3]]
            frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_NEAREST)
            frames[i, :] = frame
        frames /= 255.0
        return frames

    def __init__(self):
        threading.Thread.__init__(self)
        self.batch_size = 16
        self.kill = False   # notify the worker thread to terminate. not to be set externally. call stop() instead
        self.op = None # the actual op

    '''
    def isRunning(self):
        return self.is_run
    '''

    def stop(self):
        K.clear_session() # xzl: will there be race condition?
        self.kill = True

    # load the_query.op_fname. will set the_query.op_index == op_index if okay
    # return: op_index, in_h, in_w,
    # op_index == -1 if no more op to load, otherwise index of the loaded op
    def loadNextOp(self, op_index:int) -> typing.Tuple[int, int, int]:
        with the_query_lock:
            if op_index >= len(the_query.op_fnames):
                return -1, 0, 0
            op_fname = the_query.op_fnames[op_index]
            assert (op_fname != "")
            # the server may give us some op names that do not exist. if so, we treat it
            # as more ops to load
            if not os.path.exists(op_fname) or not os.path.isfile(op_fname):
                logger.critical(f"cannot load op {op_fname}")
                return -1, 0, 0

            the_query.op_index = op_index

        logger.critical("op worker: loading op: %s" % op_fname)

        if op_index == 0: # init keras once...
            # xzl: tf1 only
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            # global tf session XXX TODO: only once
            keras.backend.set_session(tf.Session(config=config))
            logger.critical("op worker: keras init done")

        # xzl: workaround in tf2. however, Keras=2.2.4 + TF 2
        # will cause the following issue.
        # https://github.com/keras-team/keras/issues/13336
        # workaround: using tf1 as of now...
        '''
        config = tf.compat.v1.ConfigProto() 
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        '''

        self.op = keras.models.load_model(op_fname)
        in_h, in_w = self.op.layers[0].input_shape[1:3]
        logger.critical(f"op worker: loaded op {op_index} {op_fname}. {in_h} x {in_w}")

        return op_index, in_h, in_w

    def run(self):
        logger.warning(f"op worker start running id = {threading.get_ident()}")
        clog = ClockLog(5)  # xzl: in Mengwei's util

        with the_query_lock:
            qid = the_query.qid
            print (f'{the_query.qid}  {[on for on in the_query.op_names]}')
            assert(qid >= 0)
            # op_index = the_query.op_index
        op_index, in_h, in_w = self.loadNextOp(op_index=0)
        assert(op_index != -1)

        while True:
            if self.kill:
                logger.info("worker: got a stop notice. bye!")
                return

            # pull work from global buf
            # the worker won't lost a batch of images, as it won't terminate here
            # imgs = random.sample(self.run_imgs, OP_BATCH_SZ)
            # sample & remove, w/o changing the order of run_imgs
            # https://stackoverflow.com/questions/16477158/random-sample-with-remove-from-list

            # op is the ONLY producer.
            with the_query_lock:
                frames = []
                while len(frames) < OP_BATCH_SZ:
                    # len(the_query.work_queue) if len(the_query.work_queue) < OP_BATCH_SZ else OP_BATCH_SZ
                    try:
                        frames.append(the_query.workqueue.pop(0))  # from the beginning
                    except IndexError:
                        break
                crop = the_query.crop
                vn = the_query.video_name
                vs = the_video_stores[vn]
                op_index = the_query.op_index

            if len(frames) == 0:  # we got nothing
                op_index, in_h, in_w = self.loadNextOp(op_index + 1)
                if op_index == -1:
                    # with the_query_lock: # can't do this, as uploading may be ongoing
                    #    the_query.qid = -1
                    with the_query_lock:
                        if len(the_query.backqueue) == 0:  # no work left even in backqueue
                            with the_stats_lock:  # will deadlock??
                                if the_stats[qid].status != 'COMPLETED':  # there may be other op workers
                                    the_stats[qid].status = 'COMPLETED'
                                    # seconds since epoch. protobuf's timestamp types seem too complicated...
                                    the_stats[qid].ts_comp = time.time()  # float
                            logger.info("opworker: nothing in run_frames. bye!")
                            return
                        else:  # some work in backbuf. time to work on them
                            random.shuffle(the_query.backqueue) # shuffle so we evenly process all windows
                            while True:
                                try:
                                    f = the_query.backqueue.pop()
                                    the_query.workqueue.append(f)
                                    the_query.framestates[f.frame_id] = '.'
                                except IndexError:
                                    break
                            logger.warning(f"opworker: move {len(the_query.workqueue)} items from backbuf.")
                            #continue
                    op_index, in_h, in_w = self.loadNextOp(0)  # backbuf: start from op0. must call this w/o lock
                    continue
                else:  # new op loaded. pull all frames from sendqueue back to workqueue
                    with the_buf_lock:
                        assert (len(the_query.workqueue) == 0) #XXX query lock?
                        # NB: high score goes to the front of workqueue
                        while len(the_send_buf) > 0:
                            the_query.workqueue.append(heapq.heappop(the_send_buf)[1])
                    logger.info(f"opworker: load a new op. move {len(the_query.workqueue)} back to workqueue")
                    continue

            # we got a batch of frames to work on
            fids = [f.frame_id for f in frames]
            frame_paths = [vs.GetFramePath(fid) for fid in fids]

            # TODO: IO shall be seperated thread
            # xzl: fixing this by having multiple op workers
            t0 = time.time()
            images = self.read_images(frame_paths, in_h, in_w, crop)
            t1 = time.time()
            scores = self.op.predict(images)
            # print ('Predict scores: ', scores)
            t2 = time.time()
            logger.critical(f'{len(images)} images. load in {t1-t0} sec, predict {t2-t1} secs')

            with the_query_lock:
                for x in fids:
                    # op0-'1', op1-'2', op2-'3'...  op10? '1'
                    the_query.framestates[x] = str(the_query.op_index+1)[0]

            # push scored frames (metadata) to send buf
            with the_cv: # auto grab send buf lock
                for i in range(len(frames)):
                    frames[i].cam_score = scores[i, 1]
                    heapq.heappush(the_send_buf,
                        (-scores[i, 1] + random.uniform(0.00001, 0.00002),
                            frames[i]
                        )
                    )
                the_cv.notify()

            with the_stats_lock:
                the_stats[qid].n_frames_processed += len(frames)

            # XXX locking
            clog.print(f'''Op worker: send_buf {len(the_send_buf)}'''
                        + f'''      work_queue {len(the_query.workqueue)}'''
                        + f'''       back_buf {len(the_query.backqueue)}''')

# a thread keep uploading images from the_send_buf to the server
def thread_uploader():
    logger.warning(f"------------ uploader start running id = {threading.get_ident()}")

    total_frames = 0

    #try:
    server_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
    server_stub = server_diva_pb2_grpc.server_divaStub(server_channel)

    logger.warning('uploader: channel connected')
    clog = ClockLog(5)  # xzl: in Mengwei's util

    while True:
        # logger.info("uploader: wait for op workers to produce a frame...")

        with the_cv: # auto grab lock
            while (len(the_send_buf) == 0 and not the_uploader_stop_req):
                the_cv.wait()
            if (the_uploader_stop_req):
                server_channel.close()
                logger.critical('uploader: got a stop req. bye!')
                return

            send_frame = heapq.heappop(the_send_buf)[1]
            # print ('uploader: Buf size: ', len(the_send_buf))
        #f = open(os.path.join(the_img_dirprefix, send_frame['name']), 'rb')

        vn = send_frame.video_name
        #with the_video_stores[vn].lock:
        _img = the_video_stores[vn].GetFrame(send_frame.frame_id)

        '''
        f = open(send_frame.local_path, 'rb')        
        # assemble the request ...
        # _height, _width, _chan = frame.shape # TODO: fill in with opencv2
        _height, _width, _chan = 0, 0, 0
        _img = common_pb2.Image(data=f.read(),
                                height=_height,
                                width=_width,
                                channel=_chan)
        '''

        req = common_pb2.DetFrameRequest(
            image=_img,
            frame_id=send_frame.frame_id,
            cam_score=send_frame.cam_score,
            cls = 'XXX', # TODO
            qid = send_frame.qid
        )

        # logger.warning("uploader:about to submit one frame")

        # https://stackoverflow.com/a/7663441
        for attempt in range (5):
            try:
                resp = server_stub.SubmitFrame(req)
                #total_frames += 1
            except Exception as e:
                logger.error(e)
                # reconnect 
                server_channel.close()
                time.sleep(1)
                server_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
                server_stub = server_diva_pb2_grpc.server_divaStub(server_channel)
            else:
                # logger.info("cloud accepts frame saying:" + resp.msg)
                total_frames += 1
                with the_query_lock:
                    qid = the_query.qid
                    assert (qid >= 0)
                    the_query.framestates[send_frame.frame_id] = 's'
                with the_stats_lock:
                    the_stats[qid].n_frames_sent += 1
                if attempt > 0:
                    logger.warning("------------- we've reconnected!--------")
                break
        else:
            logger.warning("we failed all attempts")
        # logger.warning("uploader:submitted one frame. server says " + resp.msg)

        clog.print("uploader: sent %d frames since born" %(total_frames))

    #except Exception as err:
    #    print("exception is ", err)
    #    logger.error("uploader: channel seems broken. bye!")
    #    return

the_uploader = None # threading.Thread(target=thread_uploader)
the_op_worker = None # XXX can be multiple workers

# make reentrant with locks?   
def _GraceKillOpWorkers():
    global the_op_worker

    if not the_op_worker:
        return

    if (not the_op_worker.is_alive()):
        the_op_worker = None
        return

    logger.info("_GraceKillOpWorkers: request op worker to stop...")

    the_op_worker.stop()
    # XXX signal cv? 
    the_op_worker.join(3)
    assert(not the_op_worker.is_alive())

    the_op_worker = None

    logger.info("_GraceKillOpWorkers: op worker stopped")

def _StartOpWorkerIfDead():
    global the_op_worker

    logger.info("_StartOpWorkerIfDead: request op worker start...")

    '''
    if (the_op_worker.is_alive()):
        logger.info("_StartOpWorkerIfDead: already started??")
        return 
    '''
    the_op_worker = OP_WORKER()
    the_op_worker.start()
    assert(the_op_worker.is_alive())

    logger.info("_StartOpWorkerIfDead: started")


def _GraceKillUploader():
    global the_uploader
    global the_uploader_stop_req

    if not the_uploader:
        return

    if not the_uploader.is_alive():
        the_uploader = None
        return

    assert(the_uploader_stop_req == 0)
    the_uploader_stop_req = 1
    with the_cv:
        the_cv.notify()
    logger.info("_GraceKillUploader: request uplaoder to stop...")

    the_uploader.join(3)
    assert(not the_uploader.is_alive())
    the_uploader_stop_req = 0
    logger.info("_GraceKillUploader: uploader stopped...")

    the_uploader = None

def _StartUploaderIfDead():
    global the_uploader

    logger.info("_StartUploaderIfDead")

    the_uploader = threading.Thread(target=thread_uploader)
    the_uploader.start()
    assert(the_uploader.is_alive())

    logger.info("_StartUploaderIfDead: started")

    '''
    if not the_uploader:
        the_uploader = threading.Thread(target=thread_uploader)
        
    if not the_uploader.is_alive():
        the_uploader = threading.Thread(target=thread_uploader)     
        the_uploader.start()
        logger.info("_StartUploaderIfDead")
    '''


class DivaCameraServicer(cam_cloud_pb2_grpc.DivaCameraServicer):

    cur_op_data = bytearray(b'') # xzl: the buffer for receiving op

    def __init__(self):
        cam_cloud_pb2_grpc.DivaCameraServicer.__init__(self)

        #self.op_worker = OP_WORKER()  # xzl: only one worker??
        #self.op_worker.start()  # xzl: they will sleep

        # self.op_worker = OP_WORKER()
        # self.op_worker.setDaemon(True)
        # self.op_worker.start()
        pass

    # propagate the self's query info to the op worker
    # (re-)gen the frame list for op worker. update the worker's working list, 
    # op, crop, etc.
    # NOT resuming the worker

    # xzl: cloud told us - all data chunks of an op has been sent. 
    def DeployOpNotify(self, request, context):
        _GraceKillOpWorkers()

        with the_query_lock:
            the_query.op_name = request.name
            the_query.op_fname = os.path.join(the_op_dir, the_query.op_name)
            the_query.crop = request.crop
            of = the_query.op_fname

        with open(of, 'wb') as f:
            f.write(self.cur_op_data)
        self.cur_op_data = bytearray(b'')

        _StartOpWorkerIfDead()  # will reload ops, etc.        
        return cam_cloud_pb2.StrMsg(msg='OK DeployOpNotify')

    # xzl: keep receiving op (a chunk of data) from cloud. 
    def DeployOp(self, request, context):
        self.cur_op_data += request.data
        # print ('XXX', len(request.data))
        return cam_cloud_pb2.StrMsg(msg='OK DeployOp')

    # got a query from the cloud. run it
    def SubmitQuery(self, request, context):
        global the_uploader
        global the_query

        logger.info(f"got a query ops: {request.op_names} video {request.video_name} qid {request.qid}")

        with the_stats_lock:
            if request.qid in the_stats:
                qids = the_stats.keys()
                logger.warning(f"qid exists. existing: {qids}")
                return cam_cloud_pb2.StrMsg(msg=f'FAIL: qid {request.qid} exists. suggested={max(qids)+1}')

        _GraceKillUploader()
        logger.info("stop op workers..")
        _GraceKillOpWorkers()
        logger.info("op workers stopped")

        with the_buf_lock:  # no need to upload anything
            the_send_buf.clear()

        ### gen the list of frame ids to process ###
        try:
            vs = the_video_stores[request.video_name]
            #with vs.lock:
            frame_ids = [img for img in vs.GetFrameIds()
                         if img % request.frameskip == 0]  # downsample.. 1/X
        except Exception as err:
            random.shuffle(frame_ids) # shuffle all ids.
            logger.error(err)

        ll = len(frame_ids)
        #frame_states = '.' * ll

        # set up the new query info
        with the_query_lock:
            '''
            the_query = QueryInfo() # wipe clean the current query info
            the_query.qid = request.qid
            the_query.crop = [int(x) for x in request.crop.split(',')]
            the_query.op_names = request.op_names
            the_query.op_fnames = [os.path.join(the_op_dir, on) for on in the_query.op_names]
            the_query.op_index = 0
            the_query.video_name = request.video_name
            the_query.img_dir = os.path.join(the_img_dirprefix, the_query.video_name) 
            '''

            the_query = QueryInfo(
                qid = request.qid,
                crop = [int(x) for x in request.crop.split(',')],
                op_names = request.op_names,
                op_fnames = [os.path.join(the_op_dir, on) for on in request.op_names],
                op_index = 0,  # next index will be 0
                video_name = request.video_name,
                img_dir = os.path.join(the_img_dirprefix, request.video_name),
                backqueue = [],
                workqueue = [],
                framestates = {}
            )

            # init each frame info. assign frames with random scores.
            # put them to send queue
            for fi in frame_ids:
                the_query.framestates[fi] = '.'
                the_query.workqueue.append(
                    # NB: frame ids already shuffled
                    FrameInfo(
                        video_name=request.video_name,
                        frame_id=fi,
                        cam_score=-1.0,
                        qid=request.qid
                    )
                )

        # init stats 
        with the_stats_lock:
            the_stats[request.qid] = QueryStat(
                    qid=request.qid,
                    video_name=request.video_name,
                    #op_name=request.op_name,
                    crop=request.crop,
                    status='STARTED',
                    n_frames_processed=0,
                    n_frames_sent=0,
                    n_frames_total=ll
                )

        _StartOpWorkerIfDead()
        _StartUploaderIfDead()

        return cam_cloud_pb2.StrMsg(msg=f'OK: recvd query {len(frame_ids)} frames')

    # move all frames in the given range to backqueue
    def DemoteFrames(self, request, context):
        global the_send_buf, the_query

        frame_ids = {}
        unchanged = 0

        for id in request.frame_ids:
            frame_ids[id] = ''

        frames = [] # all frames to demote

        # demote workqueue->{tmp}
        workqueue = []
        with the_query_lock:
            for f in the_query.workqueue:
                if f.frame_id in frame_ids:
                    frames.append(f)
                else:
                    workqueue.append(f)
                    unchanged += 1
            the_query.workqueue = workqueue

        down = len(frames)
        logger.info(f"demote workqueue->tmp down {down} unchanged {unchanged}")

        # demote upload queue->tmp
        unchanged = 0
        send_buf = []
        with the_buf_lock:
            for (score, f) in the_send_buf:
                if f.frame_id in frame_ids:
                    frames.append(f)
                else:
                    send_buf.append((score, f))
                    unchanged += 1
            the_send_buf = send_buf
            heapq.heapify(the_send_buf)

            # nothing to promo from backqueue

            the_query.backqueue += frames
            for f in frames:
                the_query.framestates[f.frame_id] = '-'

        logger.info(f"demote sendbuf --> tmp down {len(frames)-down} unchanged {unchanged}")

        return cam_cloud_pb2.StrMsg(msg=f'OK: {len(frames)} down')

    # moves all frames NOT in the framemap to backqueue
    def PromoteFrames0(self, request, context):
        frame_ids = {}
        up = 0
        unchanged = 0

        for id in request.frame_ids:
            frame_ids[id] = ''

        #print(frame_ids)

        frames = []

        # 1. backqueue-->workqueue.
        with the_query_lock:
            for idx, f in enumerate(the_query.backqueue):
                if f.frame_id in frame_ids:
                    the_query.workqueue = [f] + the_query.workqueue # front
                    up += 1
                else:
                    unchanged += 1
        #if up == 0:
        #    return cam_cloud_pb2.StrMsg(msg=f'OK down:{len(frames)} up:{up} unchanged:{unchanged}')
        logger.info(f"workqueue<--backqueue up {up} unchanged {unchanged}")
        unchanged = 0

        # 2. upload buf --> {tmp}
        with the_buf_lock:
            for idx, (score,f) in enumerate(the_send_buf):
                #print(f.frame_id)
                if not f.frame_id in frame_ids:
                    frames.append(the_send_buf.pop(idx)[1]) # XXX move score over as well ??
                else:
                    unchanged += 1
            heapq.heapify(the_send_buf)

        logger.info(f"upload buf --> tmp down {len(frames)} unchanged {unchanged}. the_send_buf {len(the_send_buf)}")

        #print('frames', frames)

        # 3. workqueue-->{tmp}
        with the_query_lock:
            for idx, f in enumerate(the_query.workqueue):
                if not f.frame_id in frame_ids:
                    frames.append(the_query.workqueue.pop(idx))
                else:
                    unchanged += 1

        #logger.info("workqueue--> tmp down {len(frames)} unchanged {unchanged}")

        # 4. {tmp} --> backqueue
            the_query.backqueue += frames

        return cam_cloud_pb2.StrMsg(msg=f'OK down:{len(frames)} up:{up} unchanged:{unchanged}')

    # moves all frames NOT in the framemap to backqueue
    def PromoteFrames(self, request, context):
        global the_send_buf, the_query

        frame_ids = {}
        up = 0
        unchanged = 0

        for id in request.frame_ids:
            frame_ids[id] = ''

        logger.warning(f"to promote {len(request.frame_ids)} frs")

        #print(frame_ids)

        frames = []

        # _GraceKillOpWorkers()

        # 3. demote workqueue-->{tmp}
        workqueue = []
        with the_query_lock:
            #logger.info(f"before the_query.workqueue {len(the_query.workqueue)}")
            for f in the_query.workqueue:
                if not f.frame_id in frame_ids:
                    #frames.append(the_query.workqueue.pop(idx))
                    frames.append(f)
                    #down += 1
                else:
                    workqueue.append(f)
                    unchanged += 1
                #print(idx, end=' ')
            the_query.workqueue = workqueue
            #logger.info(f"after the_query.workqueue {len(the_query.workqueue)}")

        down = len(frames)
        logger.info(f"demote workqueue->tmp down {down} unchanged {unchanged}")

        # 2. demote upload buf --> {tmp}
        unchanged = 0
        send_buf = []
        with the_buf_lock:
            #for idx, (score,f) in enumerate(the_send_buf):
            for (score, f) in the_send_buf:
                if not f.frame_id in frame_ids:
                    #frames.append(the_send_buf.pop(idx)[1]) # XXX move score over as well ??
                    frames.append(f)  # drop the scores
                else:
                    send_buf.append((score, f))
                    unchanged += 1
            the_send_buf = send_buf
            heapq.heapify(the_send_buf)
        logger.info(f"demote sendbuf --> tmp down {len(frames)-down} unchanged {unchanged}")

        #print('frames', frames)

        # 1. promote backqueue-->workqueue.
        unchanged = 0
        backqueue = []
        with the_query_lock:
            #for idx, f in enumerate(the_query.backqueue):
            for f in the_query.backqueue:
                if f.frame_id in frame_ids:
                    the_query.workqueue = [f] + the_query.workqueue # front
                    up += 1
                else:
                    backqueue.append(f)
                    unchanged += 1
            the_query.backqueue = backqueue
        #if up == 0:
        #    return cam_cloud_pb2.StrMsg(msg=f'OK down:{len(frames)} up:{up} unchanged:{unchanged}')

        # 4. {tmp} --> backqueue
            the_query.backqueue += frames
            for f in frames:
                the_query.framestates[f.frame_id] = '-'

        logger.info(f"promo backqueue->workqueue up {up} unchanged {unchanged}")
        logger.warning(f"after: send_buf {len(send_buf)} workqueue {len(workqueue)} total {len(send_buf)+len(workqueue)}")

        # logger.info(f"workqueue<--backqueue up {up} unchanged {unchanged}")

        # _StartOpWorkerIfDead()

        return cam_cloud_pb2.StrMsg(msg=f'OK down:{len(frames)} up:{up} unchanged:{unchanged}')

    # got a cmd for controlling the CURRENT query
    def ControlQuery(self, request, context):
        global the_query

        logger.info("got a query cmd qid %d cmd %s"
                    %(request.qid, request.command))


        with the_query_lock:
            '''
            if (request.qid != the_query.qid):
                return cam_cloud_pb2.StrMsg(msg=f'FAIL req qid {request.qid} != query qid {the_query.qid}')
            '''
            qid = the_query.qid

        if request.command == "RESET": # meaning clean up all query stats, etc.
            _GraceKillUploader()
            _GraceKillOpWorkers()

            with the_query_lock:
                the_query = None

            with the_stats_lock:
                the_stats.clear()

            with the_buf_lock:  # no need to upload anything
                the_send_buf.clear()

            # do not resume uploader/opworkers
            return cam_cloud_pb2.StrMsg(msg='OK cam reset')

        if qid == -1:
            return cam_cloud_pb2.StrMsg(msg='FAIL: current qid == -1')

        if request.command == "PAUSE":
            # kill uploader/opworkers but retain query state & states
            _GraceKillUploader()
            _GraceKillOpWorkers()
            return cam_cloud_pb2.StrMsg(msg=f'OK query {qid} paused')
        elif request.command == "RESUME":
            _StartUploaderIfDead()
            _StartOpWorkerIfDead()
            return cam_cloud_pb2.StrMsg(msg=f'OK query {qid} resumed')
        elif request.command == "STOP":
            # TODO: implement
            return cam_cloud_pb2.StrMsg(msg='FAIL')
        else:
            logger.error("unknown cmd")
            return cam_cloud_pb2.StrMsg(msg='FAIL')

    def GetStats(self, request, context):
        n_frames_processed = 0
        n_frames_total = 0
        status = "UNKNOWN"
        t0 = time.time()

        # network bw measurement, cf
        # https://stackoverflow.com/questions/15616378/python-network-bandwidth-monitor

        with the_stats_lock:
            if request.qid in the_stats:
                vm = psutil.virtual_memory()
                #logger.critical(f"{vm.used} {vm.total}")
                ret = cam_cloud_pb2.QueryProgress(qid = request.qid,
                            video_name = the_query.video_name, # XXX lock
                            n_frames_processed = the_stats[request.qid].n_frames_processed,
                            n_frames_sent = the_stats[request.qid].n_frames_sent,
                            n_frames_total = the_stats[request.qid].n_frames_total,
                            status = the_stats[request.qid].status,
                            n_bytes_sent = psutil.net_io_counters().bytes_sent,
                            mem_usage_percent = vm.used / vm.total)
            else:
                ret = cam_cloud_pb2.QueryProgress(qid = request.qid,
                                                   status = 'NONEXISTING')

        #logger.info("GetStats qid %d" % (request.qid))
        logger.info("GetStats qid %d %.2f ms" % (request.qid, 1000 * (time.time() - t0)))
        return ret

    # XXX: only return the current query's state. can be extended
    def GetQueryFrameStates(self, request, context):
        t0 = time.time()

        # assumption: we want to respond to cloud asap. deepcopy may be too expensive.
        with the_query_lock:
            if not the_query or the_query.qid < 0:
                return cam_cloud_pb2.FrameMap(frame_ids=[], frame_states="") # empty
            #fs = copy.deepcopy(the_query.framestates) # a snapshot
            s = FrameStatesToFrameMap(the_query.framestates)

        logger.info("GetQueryFrameStates qid %d %.2f ms" % (request.qid, 1000*(time.time()-t0)))
        return s

        '''
        s = [(fid, st) for fid, st in fs.items()]
        sorted(s, key=lambda x: x[0])
        fids = [x[0] for x in s]
        states = [x[1] for x in s]

        return cam_cloud_pb2.FrameMap(frame_ids = fids, frame_states = ''.join(states))
        '''

        # transfer as framemap
        #return FrameStatesToFrameMap(fs)

    '''        
    # OLD 
    # xzl: on receiving a req (webframework -> controller ->), asking for processing footage
    # yolov3 is invoked over gRPC in async way
    # rpc get_videos(google.protobuf.Empty) returns (common.get_videos_resp) {};
    # rpc process_video(common.VideoRequest) returns (google.protobuf.Empty) {};
    def process_video(self, request, context):
        # FIXME
        logger.info("got req: process video")
        print(f"process_video: {request} ")
        video_path = os.path.join(VIDEO_FOLDER, request.video_name)
        temp_name = request.video_name
        v_name = temp_name.split('/')[-1]
        video_folder_name = '.'.join(v_name.split('.')[:-1])

        target_class = request.object_name

        source = cv2.VideoCapture(video_path)
        counter = (request.offset // 30) * 30

        source.set(cv2.CAP_PROP_POS_FRAMES, request.offset)

        score_obj = {}

        with grpc.insecure_channel(YOLO_CHANNEL_ADDRESS) as channel:
            stub = det_yolov3_pb2_grpc.DetYOLOv3Stub(channel)

            while source.isOpened():
                ret, frame = source.read()
                if not ret:
                    break

                if (counter % 30) == 0:
                    
                    op_score = self.op.predict_image(frame, '350,0,720,400')
                    if (op_score <= 0.3):
                        counter += 1
                        continue
                    else:
                        # FIXME move the following computation to
                        # another thread
                        PQueue.put((op_score, frame))
                    
                    logger.warning("xzl: bypass on-cam op... as of now")
                    _height, _width, _chan = frame.shape
                    _img = common_pb2.Image(data=frame.tobytes(),
                                            height=_height,
                                            width=_width,
                                            channel=_chan)

                    # xzl: invoking server side yolo? (sync??)
                    req = det_yolov3_pb2.DetectionRequest(
                        image=_img,
                        name=f'{counter}.jpg',
                        threshold=0.3,
                        targets=[target_class])
                    resp = stub.Detect(req)

                    exist_target = False

                    temp_score_map = {}

                    # xzl: a resp from server: a list of BBs
                    # draw bbox on the image
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

                counter += 1

        source.release()

        # per image json file: recording confidence
        with open(
                os.path.join(STATIC_FOLDER,
                             *[video_folder_name, target_class,
                               'scores.json']), 'w') as fptr:
            fptr.write(json.dumps(score_obj))

        # xzl: does not return results inline
        return empty_pb2.Empty()

    # OLD 
    # xzl: return metadata for all stored videos. w/ video url, w/o image/score url
    # this seems for the client to display video
    def get_videos(self, request, context):
        logger.info("got req: get videos")
        _files = os.listdir(VIDEO_FOLDER)
        video_files = list(
            filter(lambda x: x.split('.')[-1] in ['mp4', 'mkv'], _files))

        _cam = common_pb2.Camera(name=_NAME, address=_ADDRESS)
        video_list = []
        for f in video_files:
            _video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, f))
            video_folder_name = '.'.join(f.split('.')[:-1])
            video_list.append(
                common_pb2.video_metadata(
                    name=f,
                    camera=_cam,
                    video_url=f'{WEB_APP_DNS}/{video_folder_name}/{f}',
                    images_url='',
                    score_file_url='',
                    frames=int(_video.get(cv2.CAP_PROP_FRAME_COUNT))))

            _video.release()

        return common_pb2.get_videos_resp(videos=video_list)
    '''

    # assumption: videos are stored as images in subdir, i.e. 
    # ${the_img_dirprefix}/${video_name}/{images}
    # $video_name may contain fps info, e.g. XX-10FPS_XXX
    # individual image files are numbered, number is img_fname[:-4]  

    # cf: 
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#wkt
    # https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.util.time_util

    # deprecated
    def ListVideos_0(self, request, context):
        video_name_list = [o for o in os.listdir(the_img_dirprefix)
                         if os.path.isdir(os.path.join(the_img_dirprefix, o))]

        video_list = []

        for f in video_name_list:
            video_path = os.path.join(the_img_dirprefix, f)

            # use hint to find FPS from video name. best effort
            m=re.search(r'''\D*(\d+)(FPS|fps)''', f)
            if m:
                fps = int(m.group(1))
            else:
                fps = -1

            # a list of frame nums without file extensions
            frame_name_list = [int(img[:-4]) for img in os.listdir(video_path)]
            minid = min(frame_name_list)
            maxid = max(frame_name_list)
            diff = maxid - minid + 1
            n_missing_frames = diff - len(frame_name_list)
            assert(n_missing_frames >= 0)

            #duration = Duration()
            #duration.seconds = 0
            #duration = 0
            #if fps > 0:
            #    duration.seconds = int((diff) / fps)

            video_list.append(cam_cloud_pb2.VideoMetadata(
                    video_name = f,
                    n_frames = len(frame_name_list),
                    fps = fps,
                    n_missing_frames = n_missing_frames,
                    #start = Timestamp(), # fake one, or Timestamp(),
                    #end = Timestamp(),
                    #duration = duration,
                    start = 0, # fake
                    end = 0,
                    duration = int(diff / fps) if fps > 0 else 0,
                    frame_id_min = minid,
                    frame_id_max = maxid
                ))
        return cam_cloud_pb2.VideoList(videos = video_list)

    def ListVideos(self, request, context):
        video_name_list = [o for o in os.listdir(the_img_dirprefix)
                           if os.path.isdir(os.path.join(the_img_dirprefix, o))]

        video_list = []

        try:
            for f in video_name_list:
                vs = the_video_stores[f]
                #with vs.lock:
                video_list.append(cam_cloud_pb2.VideoMetadata(
                    video_name=f,
                    n_frames=vs.GetNumFrames(),
                    fps=vs.fps,
                    n_missing_frames=vs.n_missing_frames,
                    frame_id_min=vs.minid,
                    frame_id_max=vs.maxid
                ))
        except Exception as err:
            logger.error(err)

        return cam_cloud_pb2.VideoList(videos=video_list)

    def GetCamSpecs(self, request, context):
        uname = platform.uname()
        stros = f"node:{uname.node}"
        strcpu = f"cpu: {psutil.cpu_count()}x {uname.machine}"
        mem = psutil.virtual_memory()
        strmem = f"mem: {int(mem.available/1024/1024)}/{int(mem.total/1024/1024)} MB"
        msg = stros + " " +  strcpu + " " + strmem
        return cam_cloud_pb2.StrMsg(msg=msg)

    # deprecated
    def GetVideoFrame_0(self, request, context):
        # try various heuristics to locate frames...
        video_path = os.path.join(the_img_dirprefix, request.video_name)
        frame_id = request.frame_id
        frame_path = ""
        found = False

        for ext in ['.JPG', '.jpg']:
            for frame_fname in [f'{frame_id:d}', f'{frame_id:06d}', f'{frame_id:07d}', f'{frame_id:08d}']:
                frame_path = os.path.join(video_path, frame_fname + ext)
                if (os.path.isfile(frame_path)):
                    found = True
                    logger.info(f"try {frame_path}... found")
                    break
                else:
                    # print(f"try {frame_path}... not found")
                    pass
            else:
                continue
            break
        if not found:
            return common_pb2.Image()   # nothing

        try:
            f = open(frame_path, 'rb')
            _height, _width, _chan = 0, 0, 0
            return common_pb2.Image(data=f.read(),
                                    height=_height,
                                    width=_width,
                                    channel=_chan)
        except Exception as e:
            logger.error(e)
            return common_pb2.Image()   # nothing

    def GetVideoFrame(self, request, context):
        logger.info(f'GetVideoFrame {request.video_name} {request.frame_id}')
        try:
            #with the_video_stores[request.video_name].lock:
            return the_video_stores[request.video_name].GetFrame(request.frame_id)
        except Exception as err:
            logger.error(err)

    '''
    # xzl: return res of a completed query. with images/score urls. w/o video url
    # for the client to display query results
    def get_video(self, request, context):
        logger.info("got req: get video")
        temp_name = request.video_name
        v_name = temp_name.split('/')[-1]
        video_folder_name = '.'.join(v_name.split('.')[:-1])

        object_name = request.object_name
        images_url = f'{WEB_APP_DNS}/{video_folder_name}/{object_name}/images/'

        score_file_url = f'{WEB_APP_DNS}/{video_folder_name}/{object_name}/scores.json'

        # xzl: just to get frame count
        _video = cv2.VideoCapture(
            os.path.join(VIDEO_FOLDER, request.video_name))
        frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
        _video.release()
        return common_pb2.video_metadata(name=request.video_name,
                                         frames=frames,
                                         video_url="",
                                         score_file_url=score_file_url,
                                         images_url=images_url,
                                         camera=request.camera,
                                         object_name=object_name)
    '''

def build_video_stores():
    video_name_list = [o for o in os.listdir(the_img_dirprefix)
                       if os.path.isdir(os.path.join(the_img_dirprefix, o))]

    try:
        for vn in video_name_list:
            logger.info(f"build videostore... {vn}")
            the_video_stores[vn] = VideoStore(video_name = vn, prefix=the_img_dirprefix)
            logger.info(f"done. {the_video_stores[vn].GetNumFrames()} frames found")

    except Exception as err:
        logger.error(err)

# https://raspberrypi.stackexchange.com/questions/22005/how-to-prevent-python-script-from-running-more-than-once
the_instance_lock = None

def serve():
    global the_instance_lock, the_img_dirprefix

    logger.info('Init camera service')
    try:
        the_instance_lock = zc.lockfile.LockFile('/tmp/diva-cam')
        logger.debug("grabbed diva lock")
    except zc.lockfile.LockError:
        logger.error("cannot lock file. are we running multiple instances?")
        sys.exit(1)

    # guess the local img path
    if platform.uname().machine.startswith('arm'):
        the_img_dirprefix = '/local/jpg'

    logger.critical(f"the_img_dirprefix set to {the_img_dirprefix}")

    build_video_stores()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    diva_cam_servicer = DivaCameraServicer()
    cam_cloud_pb2_grpc.add_DivaCameraServicer_to_server(
        diva_cam_servicer, server)
    server.add_insecure_port(f'[::]:{CAMERA_CHANNEL_PORT}')
    server.start()

    # _StartUploaderIfDead()

    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logger.warning('DivaCloud stop!!!')
        server.stop(0)

if __name__ == '__main__':
    serve()
