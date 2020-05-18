import time
import os
import sys
import json
import cv2
import threading
import heapq
import re
import random

from google.protobuf.timestamp_pb2 import *
from google.protobuf.duration_pb2 import *


# from threading import Thread, Event, Lock
from concurrent import futures
import logging, coloredlogs
from queue import PriorityQueue

import grpc

import keras
import keras.backend as K

import cam_cloud_pb2_grpc
#import det_yolov3_pb2
#import det_yolov3_pb2_grpc
import common_pb2
import cam_cloud_pb2
import server_diva_pb2_grpc # for submitting frames

from google.protobuf import empty_pb2

from variables import CAMERA_CHANNEL_PORT
from variables import DIVA_CHANNEL_ADDRESS # for submitting frames
from variables import  * # xzl

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

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)

# FIXME (xzl: unused?)
# OP_FNAME_PATH = '/home/bryanjw01/workspace/test_data/source/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'

# TODO: use Python3's PriorityQueue
# buf for outgoing frames. only holding image metadata (path, score, etc.)
# out of which msgs for outgoing frames (including metadata) will be assembled 
the_send_buf = []   
the_buf_lock = threading.Lock()
the_cv = threading.Condition(the_buf_lock)


class QueryInfo():
    qid = -1    # current qid
    crop = [-1,-1,-1,-1]
    op_name = ""
    op_fname = ""
    img_dir = ""
    video_name = ""
    run_imgs = []  # maybe put a separate lock over this
    
# for the current query: all frames (metadata such as fullpaths etc) to be processed. 
# workers will pull from here
the_query_lock = threading.Lock()
the_query = QueryInfo() 

# queries ever executed
the_stats = {}
the_stats_lock = threading.Lock()


the_uploader_stop_req = 0

'''
ev_uploader_stop_req = threading.Event()
ev_uploader_stop_req.clear()
ev_uploader_stop_done = threading.Event()
ev_uploader_stop_done.clear()
ev_uploader_resume_req = threading.Event()

ev_worker_stop_req = threading.Event()
ev_worker_stop_req.clear()
ev_worker_stop_done = threading.Event()
'''

PQueue = PriorityQueue()


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
        self.crop = None
        self.op_updated = True
        '''

    '''
    def isRunning(self):
        return self.is_run
    '''
        
    def stop(self):
        K.clear_session() # xzl: will there be race condition?
        self.kill = True
        
    def loadOp(self):
        with the_query_lock:
            op_fname = the_query.op_fname
            assert(op_fname != "")
            
        logger.info("op worker: loading op: %s" %op_fname)
        
        # xzl: tf1 only
        config = tf.ConfigProto() 
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        keras.backend.set_session(tf.Session(config=config))  # global tf session XXX TODO: only once
        
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
        logger.info("op worker: loading op done.")      
        
        return in_h, in_w
    
    def run(self):        
        logger.info("op worker start running")
        clog = ClockLog(5)  # xzl: in Mengwei's util
        
        with the_query_lock:
            qid = the_query.qid
            print (f'{the_query.qid}  {the_query.op_name}')
            assert(qid >= 0)
            
        in_h, in_w = self.loadOp()
        
        while True:
            if self.kill:
                logger.info("worker: got a stop notice. bye!")
                return 
            '''
            if not self.is_run:
                #logger.info("op worker: !is_run. sleep 1 sec...")
                clog.print("op worker: !is_run. sleep...")
                time.sleep(1)
                continue
            '''
            
            # pull work from global buf 
            
            # the worker won't lost a batch of images, as it won't termiante here            
            # imgs = random.sample(self.run_imgs, OP_BATCH_SZ)
            # sample & remove, w/o changing the order of run_imgs
            # https://stackoverflow.com/questions/16477158/random-sample-with-remove-from-list
            batch = OP_BATCH_SZ
            
            with the_query_lock: 
                if len(the_query.run_imgs) < OP_BATCH_SZ:
                    batch = len(the_query.run_imgs)                
                imgs = [the_query.run_imgs.pop(random.randrange(len(the_query.run_imgs))) for _ in range(batch)]
                crop = the_query.crop
                sz_run_imgs = len(the_query.run_imgs)
                
            if batch == 0: # we got nothing, clean up...
                with the_query_lock:
                    the_query.qid = -1
                with the_stats_lock:
                    if the_stats[qid]['status'] != 'COMPLETED': # there may be other op workers            
                        the_stats[qid]['status'] = 'COMPLETED'       
                        # seconds since epoch. protobuf's timestamp types seem too complicated...             
                        the_stats[qid]['ts_comp'] = time.time() # float                                                                                                                       
                logger.info("opworker: nothing in run_imgs. bye!")
                return 
                        
            #TODO: IO shall be seperated thread
            # xzl: fixing this by having multiple op workers
            frames = self.read_images(imgs, in_h, in_w, crop) 
            scores = self.op.predict(frames)
            # print ('Predict scores: ', scores)
            
            # push a scored frame (metadata) to send buf            
            with the_cv: # auto grab send buf lock
                for i in range(batch):
                    heapq.heappush(the_send_buf, 
                        (-scores[i, 1] + random.uniform(0.00001, 0.00002), 
                         #imgs[i].split('/')[-1],    # tie breaker to prevent heapq from comparing the following dict 
                            {
                                'name': imgs[i].split('/')[-1],
                                'cam_score': -scores[i, 1],
                                'local_path': imgs[i],
                                'qid': qid, 
                            }
                        )
                    )
                    sz_send_buf = len(the_send_buf)
                the_cv.notify()
                
            # self.n_frames_processed += OP_BATCH_SZ
            with the_stats_lock:
                the_stats[qid]['n_frames_processed'] += batch                
                            
            clog.print('Op worker: send_buf size: %d frames (nb: OP_BATCH_SZ=%d), remaining : %d' 
                       %(sz_send_buf, OP_BATCH_SZ, sz_run_imgs))
            #logger.info('Buf size: %d, total num: %d' %(sz,total))

# a thread keep uploading images from the_send_buf to the server
def thread_uploader():
    logger.warn('uploader started')
    
    total_frames = 0
    
    #while True:
    try: 
        server_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
        server_stub = server_diva_pb2_grpc.server_divaStub(server_channel)    
        
        logger.warn('uploader: channel connected')
        clog = ClockLog(5)  # xzl: in Mengwei's util
        
        while True: 
            # logger.info("uploader: wait for op workers to produce a frame...")
            '''
            if ev_uploader_stop_req.is_set():
                print("uploader: pausing. close channel..")
                server_channel.close()
                ev_uploader_stop_req.clear()
                ev_uploader_resume_req.clear()            
                ev_uploader_stop_done.set() # signal the main thread
                print("uploader: paused. wait to resume...")
                ev_uploader_resume_req.wait()
                ev_uploader_resume_req.clear()
                print("uploader: resume.")            
                server_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
                server_stub = server_diva_pb2_grpc.server_divaStub(server_channel)
                print("uploader: channel up.")
            '''
                        
            with the_cv: # auto grab lock
                while (len(the_send_buf) == 0 and not the_uploader_stop_req):
                    the_cv.wait()
                if (the_uploader_stop_req):
                    server_channel.close()
                    logger.warn('uploader: got a stop req. bye!')
                    return 
                
                send_frame = heapq.heappop(the_send_buf)[1]
                # print ('uploader: Buf size: ', len(the_send_buf))
            #f = open(os.path.join(the_img_dirprefix, send_frame['name']), 'rb')
            f = open(send_frame['local_path'], 'rb')
            
            # assemble the request ...
            # _height, _width, _chan = frame.shape # TODO: fill in with opencv2
            _height, _width, _chan = 0, 0, 0
            _img = common_pb2.Image(data=f.read(),
                                    height=_height,
                                    width=_width,
                                    channel=_chan)
    
            req = common_pb2.DetFrameRequest(
                image=_img,
                name=send_frame['name'],
                cam_score=send_frame['cam_score'],
                cls = 'XXX', # TODO
                qid = send_frame['qid']
            )
            
            # logger.warn("uploader:about to submit one frame")
            
            # https://stackoverflow.com/a/7663441
            for _ in range (5): 
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
                    total_frames += 1 
                    break
            else: 
                logger.warn("we failed all attempts") 
            # logger.warn("uploader:submitted one frame. server says " + resp.msg)
                                  
            clog.print("uploader: sent %d frames since born" %(total_frames))
    except Exception as err:
        logger.error(err)
        logger.error("uploader: channel seems broken. bye!")
        return

the_uploader = None # threading.Thread(target=thread_uploader)
the_op_worker = None # XXX can be multiple workers
 
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
    
    '''
    img_dir = ''  # the queried video, a subdir under @the_img_dirprefix
    cur_op_name = '' # used to saved a received op
    op_fname = '' # full path of the current op
    video_name = '' 
    #send_buf = []   # buf for outgoing frames. only holding image metadata (path, score, etc.)
    crop = ''
    operator_ = None
    qid = -1  # current query we are running
    '''
    
    def __init__(self):
        cam_cloud_pb2_grpc.DivaCameraServicer.__init__(self)
        
        #self.op_worker = OP_WORKER()  # xzl: only one worker??
        #self.op_worker.start()  # xzl: they will sleep 

        # self.op_worker = OP_WORKER()
        # self.op_worker.setDaemon(True)
        # self.op_worker.start()
        pass

    '''
    # cloud is requesting one "top" frame from send_buf
    # deprecated
    def GetFrame(self, request, context):
        the_buf_lock.acquire()
        if len(the_send_buf) == 0:
            print ('Empty queue, nothing to send')
            the_buf_lock.release()
            return cam_cloud_pb2.Frame(name='', data=b'')
        send_img = heapq.heappop(self.send_buf)[1]
        the_buf_lock.release()
        
        with open(os.path.join(self.img_dir, send_img), 'rb') as f:
            return cam_cloud_pb2.Frame(
                    name=send_img.split('/')[-1],
                    data=f.read())

    # not needed?
    def InitDiva(self, request, context):
        self.img_dir = os.path.join(the_img_dirprefix, request.video_name) 
        with the_buf_lock:
            the_send_buf = []
        return cam_cloud_pb2.StrMsg(msg='OK')
    '''
    
    # propagate the self's query info to the op worker
    # (re-)gen the frame list for op worker. update the worker's working list, 
    # op, crop, etc.
    # NOT resuming the worker 
    '''
    def LoadQuery(self):
        if self.op_worker.isRunning():
            logger.error("bug: must stop worker first")
            sys.exit(1)
        
        # xzl: sample down to 1FPS. listdir can be very slow
        all_imgs = os.listdir(self.img_dir)
        selected_imgs = [img for img in all_imgs if int(img[:-4]) % 10 == 0] # 1 FPS
        
        with the_work_lock:
            the_work_buf.clear() 
            for img in selected_imgs:
                the_work_buf.append(os.path.join(self.img_dir, img))
        
        self.op_worker.load(
                self.img_dir, selected_imgs, self.op_fname, self.crop, self.qid) 
        pass     
    '''

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
        return cam_cloud_pb2.StrMsg(msg='OK')

    # xzl: keep receiving op (a chunk of data) from cloud. 
    def DeployOp(self, request, context):
        self.cur_op_data += request.data
        # print ('XXX', len(request.data))
        return cam_cloud_pb2.StrMsg(msg='OK')
        
    # got a query from the cloud. run it
    def SubmitQuery(self, request, context):
        global the_uploader 
        global the_query
        
        logger.info("got a query op %s video %s qid %d" %(request.op_name, request.video_name, request.qid))
        
        with the_stats_lock:
            if request.qid in the_stats:
                logger.warn("qid exists")    
                return cam_cloud_pb2.StrMsg(msg='FAIL: qid exists')
    
        logger.info("stop op workers..")
        print("stop op workers")
        # stop op workers
        _GraceKillOpWorkers()
        logger.info("op workers stopped")
       
        _GraceKillUploader()
       
        with the_query_lock:            
            the_query = QueryInfo() # wipe clean all old query info
            the_query.qid = request.qid
            the_query.crop = [int(x) for x in request.crop.split(',')]
            the_query.op_name = request.op_name
            the_query.op_fname = os.path.join(the_op_dir, the_query.op_name)
            the_query.video_name = request.video_name
            the_query.img_dir = os.path.join(the_img_dirprefix, the_query.video_name) 
               
        # xzl: sample down to 1FPS. listdir can be very slow
        all_imgs = os.listdir(the_query.img_dir)
        selected_imgs = [img for img in all_imgs if int(img[:-4]) % 10 == 0] # 1 FPS
        
        with the_query_lock:
            the_query.run_imgs.clear() 
            for img in selected_imgs:
                the_query.run_imgs.append(os.path.join(the_query.img_dir, img))
                ll = len(the_query.run_imgs) 
            
        # init stats 
        with the_stats_lock:
            the_stats[request.qid] = {
                    'qid' : request.qid,
                    'video_name' : request.video_name,
                    'op_name' : request.op_name,
                    'crop' : request.crop,
                    'status' : 'STARTED', 
                    'n_frames_processed' : 0,
                    'n_frames_total' : ll
            }

        _StartOpWorkerIfDead()                            
        _StartUploaderIfDead()
            
        return cam_cloud_pb2.StrMsg(msg='OK: recvd query')
                                    
    # got a cmd for controlling an ongoing query. exec it
    def ControlQuery(self, request, context):
        global the_query
        
        logger.info("got a query cmd qid %d cmd %s" 
                    %(request.qid, request.command))
        
        # commands w/o qid 
        if request.command == "RESET": # meaning clean up all query stats, etc.
            _GraceKillUploader()
            _GraceKillOpWorkers()
            
            with the_query_lock:
                the_query = QueryInfo()
                
            with the_stats_lock:
                the_stats.clear()
                             
            # do not resume uploader/opworkers
            return cam_cloud_pb2.StrMsg(msg='OK')  
                
        # commands w/ qid. only can control the current query
        with the_query_lock: 
            if (request.qid != the_query.qid):
                return cam_cloud_pb2.StrMsg(msg=f'FAIL req qid {request.qid} != query qid {the_query.qid}')
                
        if request.command == "PAUSE":
            # kill uploader/opworkers but retain query state & states
            _GraceKillUploader() 
            _GraceKillOpWorkers()        
            return cam_cloud_pb2.StrMsg(msg='OK')
        elif request.command == "RESUME":
            with the_query_lock:
                if the_query.qid < 0:                 
                    return cam_cloud_pb2.StrMsg(msg='FAIL: no active query')
            
            _StartUploaderIfDead()
            _StartOpWorkerIfDead()
            return cam_cloud_pb2.StrMsg(msg='OK') 
        elif request.command == "STOP":
            # TODO: implement
            return cam_cloud_pb2.StrMsg(msg='FAIL')
        else:
            logger.error("unknown cmd")
            return cam_cloud_pb2.StrMsg(msg='FAIL')
    
    def GetQueryProgress(self, request, context):
        logger.info("GetQueryProgress qid %d" %(request.qid))

        n_frames_processed = 0
        n_frames_total = 0
        status = "UNKNOWN"
                                
        with the_stats_lock:
            if request.qid in the_stats:            
                n_frames_processed = the_stats[request.qid]['n_frames_processed']
                n_frames_total = the_stats[request.qid]['n_frames_total']
                status = the_stats[request.qid]['status']
                                        
        return cam_cloud_pb2.QueryProgress(qid = request.qid, 
                    video_name = the_query.video_name, # XXX lock
                    n_frames_processed = n_frames_processed,
                    n_frames_total = n_frames_total,
                    status = status)

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
    
    def ListVideos(selfelf, request, context):
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
            diff = max(frame_name_list) - min(frame_name_list) + 1        
            n_missing_frames = diff - len(frame_name_list)
            assert(n_missing_frames >= 0)
            
            duration = Duration()
            duration.seconds = 0        
            
            if fps > 0:
                duration.seconds = int((diff) / fps)
                                        
            video_list.append(cam_cloud_pb2.VideoMetadata(
                    name = f, 
                    n_frames = len(frame_name_list), 
                    fps = fps,  
                    n_missing_frames = n_missing_frames,
                    start = Timestamp(), # fake one, or Timestamp(), 
                    end = Timestamp(), 
                    duration = duration
                ))
        return cam_cloud_pb2.VideoList(videos = video_list)
                             
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
    
def serve():    
    
    logger.info('Init camera service')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
