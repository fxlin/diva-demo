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
import logging
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

# FIXME (xzl: unused?)
# OP_FNAME_PATH = '/home/bryanjw01/workspace/test_data/source/chaweng-a3d16c61813043a2711ed3f5a646e4eb.hdf5'

# TODO: use Python3's PriorityQueue
# buf for outgoing frames. only holding image metadata (path, score, etc.)
# out of which msgs for outgoing frames (including metadata) will be assembled 
the_send_buf = []   
the_buf_lock = threading.Lock()
the_cv = threading.Condition(the_buf_lock)

# queries ever executed
the_queries = {}
the_queries_lock = threading.Lock()


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

    def reset(self):
        self.is_run = False
        self.batch_size = 16
        
        self.qid = -1
        self.run_imgs = []  # all frames for this worker
        
        self.kill = False
        self.op = None
        self.crop = None
        self.op_updated = True
                
    def __init__(self):
        threading.Thread.__init__(self)
        self.reset()

    # xzl: load the list of frames, op, crop factor to the worker to exec 
    # invoked by the mainthread on init, and whenenver op is updated
    def load(self, img_dir, img_names, op_fname, crop, qid):
        logger.info("load...")
        self.run_imgs = [] # full paths of frames to process
        self.op_fname = op_fname
        self.qid = qid
        for img in img_names:
            self.run_imgs.append(os.path.join(img_dir, img))
        
        with the_queries_lock:
            the_queries[self.qid]['n_frames_total'] = len(self.run_imgs)
            
        self.op_fname = op_fname 
        self.op_updated = True # xzl: tell the worker thread a new op is there        
        self.crop = [int(x) for x in crop.split(',')] #xzl: just convert to int array?        

    def isRunning(self):
        return self.is_run

    def kill(self):
        K.clear_session()
        self.kill = True

    def pause(self):
        self.is_run = False # xzl: change to cond var?

    def resume(self):
        self.is_run = True # xzl: will wake up the worker thread
        
    def run(self):        
        logger.info("op worker start running")
        clog = ClockLog(5)  # xzl: in Mengwei's util
        
        while True:
            if self.kill:
                break
            if not self.is_run:
                #logger.info("op worker: !is_run. sleep 1 sec...")
                clog.print("op worker: !is_run. sleep...")
                time.sleep(1)
                continue
            if self.op_updated:  # xzl: reload op 
                logger.info("op worker: loading op: %s" %self.op_fname)
                
                # xzl: tf1 only
                config = tf.ConfigProto() 
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
                keras.backend.set_session(tf.Session(config=config))
                
                # xzl: workaround in tf2. however, Keras=2.2.4 + TF 2
                # will cause the following issue. 
                # https://github.com/keras-team/keras/issues/13336
                # workaround: using tf1 as of now... 
                '''
                config = tf.compat.v1.ConfigProto() 
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
                '''
                
                self.op = keras.models.load_model(self.op_fname)
                in_h, in_w = self.op.layers[0].input_shape[1:3]
                self.op_updated = False
                logger.info("op worker: loading op done.")       
                     
            # imgs = random.sample(self.run_imgs, OP_BATCH_SZ)
            # sample & remove, w/o changing the order of run_imgs
            # https://stackoverflow.com/questions/16477158/random-sample-with-remove-from-list
            batch = OP_BATCH_SZ 
            if len(self.run_imgs) < OP_BATCH_SZ:
                batch = len(self.run_imgs)                
            imgs = [self.run_imgs.pop(random.randrange(len(self.run_imgs))) for _ in range(batch)]

            #TODO: IO shall be seperated thread
            # xzl: fixing this by having multiple op workers
            frames = self.read_images(imgs, in_h, in_w, crop=self.crop) 
            scores = self.op.predict(frames)
            # print ('Predict scores: ', scores)
            
            # push a scored frame (metadata) to send buf
            
            with the_cv: # auto grab lock
                for i in range(batch):
                    heapq.heappush(the_send_buf, 
                        (-scores[i, 1] + random.uniform(0.00001, 0.00002), 
                         #imgs[i].split('/')[-1],    # tie breaker to prevent heapq from comparing the following dict 
                            {
                                'name': imgs[i].split('/')[-1],
                                'cam_score': -scores[i, 1],
                                'local_path': imgs[i],
                                'qid': self.qid, 
                            }
                        )
                    )
                the_cv.notify()
                
            # self.n_frames_processed += OP_BATCH_SZ
            with the_queries_lock:
                the_queries[self.qid]['n_frames_processed'] += batch
                if len(self.run_imgs) == 0:
                    the_queries[self.qid]['status'] = 'COMPLETED'
            
            # print ('Buf size: ', len(the_send_buf))
            with the_buf_lock:
                sz = len(the_send_buf)
                total = len(self.run_imgs)
                
            clog.print('Op worker: Buf size: %d frames (nb: OP_BATCH_SZ=%d), remaining : %d' 
                       %(sz, OP_BATCH_SZ, total))
            #logger.info('Buf size: %d, total num: %d' %(sz,total))

# a thread keep uploading images from the_send_buf to the server
def thread_uploader():
    logger.info('uploader started')
    
    total_frames = 0
    last_frames = 0
    
    #while True:
    try: 
        server_channel = grpc.insecure_channel(DIVA_CHANNEL_ADDRESS)
        server_stub = server_diva_pb2_grpc.server_divaStub(server_channel)    
        
        logger.info('uploader: channel connected')
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
                cls = 'bike', # TODO
                qid = send_frame['qid']
            )
            
            resp = server_stub.SubmitFrame(req)
            #logger.info("uploader:submitted one frame. server says " + resp.msg)
            
            total_frames += 1            
            clog.print("uploader: sent %d frames since born" %(total_frames))
    except Exception as err:
        logger.error(err)
        print("uploader: channel seems broken. bye!")
        return
                    
def _GraceKillUploader():
    global the_uploader
    global the_uploader_stop_req
            
    assert(the_uploader_stop_req == 0)
    the_uploader_stop_req = 1
    with the_cv:
        the_cv.notify()
    logger.info("_GraceKillUploader: request uplaoder to stop...")
    
    the_uploader.join(3)
    assert(not the_uploader.is_alive())
    the_uploader_stop_req = 0
    logger.info("_GraceKillUploader: uploader stopped...")

def _StartUploaderIfDead():
    global the_uploader
    if not the_uploader.is_alive():
        the_uploader = threading.Thread(target=thread_uploader)     
        the_uploader.start()
        logger.info("_StartUploaderIfDead")

the_uploader = threading.Thread(target=thread_uploader)
                            
class DivaCameraServicer(cam_cloud_pb2_grpc.DivaCameraServicer):
    img_dir = ''  # the queried video, a subdir under @the_img_dirprefix
    cur_op_data = bytearray(b'') # xzl: the buffer for receiving op
    cur_op_name = '' # used to saved a received op
    op_fname = '' # full path of the current op
    video_name = '' 
    #send_buf = []   # buf for outgoing frames. only holding image metadata (path, score, etc.)
    crop = ''
    operator_ = None
    qid = -1  # current query we are running

    def cleaup(self):
        self.cur_op_data = bytearray(b'')
        self.cur_op_name = ""

    def __init__(self):
        cam_cloud_pb2_grpc.DivaCameraServicer.__init__(self)
        
        logger.info("create op worker")
        
        self.op_worker = OP_WORKER()  # xzl: only one worker??
        self.op_worker.start()  # xzl: they will sleep 

        # self.op_worker = OP_WORKER()
        # self.op_worker.setDaemon(True)
        # self.op_worker.start()
        pass

    def KillOp(self):
        self.op_worker.kill()

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

    # propagate the self's query info to the op worker
    # (re-)gen the frame list for op worker. update the worker's working list, 
    # op, crop, etc.
    # NOT resuming the worker 
    def LoadQuery(self):
        if self.op_worker.isRunning():
            logger.error("bug: must stop worker first")
            sys.exit(1)
        
        # xzl: sample down to 1FPS. listdir can be very slow
        all_imgs = os.listdir(self.img_dir)
        selected_imgs = [img for img in all_imgs if int(img[:-4]) % 10 == 0] # 1 FPS
        self.op_worker.load(
                self.img_dir, selected_imgs, self.op_fname, self.crop, self.qid) 
                

    # xzl: cloud told us - all data chunks of an op has been sent. 
    def DeployOpNotify(self, request, context):
        # now save the op to disk
        self.cur_op_name = request.name
        self.op_fname = os.path.join(the_op_dir, self.cur_op_name)
        self.crop = request.crop
        
        with open(self.op_fname, 'wb') as f:
            f.write(self.cur_op_data)
        self.cur_op_data = bytearray(b'')

        # stop op processing 
        if self.op_worker.isRunning():
            self.op_worker.pause()
        
        self.LoadQuery()  # really needed here?         
        self.op_worker.resume() # wake up workers
        
        return cam_cloud_pb2.StrMsg(msg='OK')

    # xzl: keep receiving op (a chunk of data) from cloud. 
    def DeployOp(self, request, context):
        self.cur_op_data += request.data
        # print ('XXX', len(request.data))
        return cam_cloud_pb2.StrMsg(msg='OK')
        
    # got a query from the cloud. run it
    def SubmitQuery(self, request, context):
        global the_uploader 
        
        logger.info("got a query op %s video %s" %(request.op_name, request.video_name))
        
        # op must exists on local disk
        self.cur_op_name = request.op_name
        self.op_fname = os.path.join(the_op_dir, self.cur_op_name)
        self.video_name = request.video_name
        self.img_dir = os.path.join(the_img_dirprefix, request.video_name) 
        self.crop = request.crop
        self.qid = request.qid

        if self.op_worker.isRunning():
            self.op_worker.pause()
            
        with the_buf_lock:
            the_send_buf = []
        
        with the_queries_lock:
#            if request.qid >= len(the_queries):
#                the_queries.extend([None] * (request.qid + 1 - len(the_queries)))
            the_queries[request.qid] = {
                    'qid' : request.qid,
                    'video_name' : request.video_name,
                    'op_name' : request.op_name,
                    'crop' : request.crop,
                    'status' : 'STARTED', 
                    'n_frames_processed' : 0,
                    'n_frames_total' : -1
            }
            
        self.LoadQuery()
        self.op_worker.resume() # wake up workers
        
        _StartUploaderIfDead()
            
        return cam_cloud_pb2.StrMsg(msg='OK')
                                    
    # got a cmd for controlling an ongoing query. exec it
    def ControlQuery(self, request, context):
        
        
        logger.info("got a query cmd qid %d cmd %s" 
                    %(request.qid, request.command))
        
        if request.command == "RESET":
            self.op_worker.pause()
            time.sleep(1)
            self.op_worker.reset()
            
            '''
            assert(not ev_uploader_stop_done.is_set())
            assert(not ev_uploader_stop_req.is_set())
            
            ev_uploader_stop_req.set()
            print("ControlQuery: request uplaoder to stop...")
            
            if (not ev_uploader_stop_done.wait(3)):
                print('tried to stop uploader thread. never heard back. why?')
                sys.exit(1)
            
            ev_uploader_stop_done.clear()
            '''
            _GraceKillUploader()
            
            with the_buf_lock:
                the_send_buf.clear()
            with the_queries_lock:
                the_queries.clear()
            
            '''
            assert(not ev_uploader_resume_req.is_set())
            ev_uploader_resume_req.set() # let the uploader go
            '''            
            the_uploader = threading.Thread(target=thread_uploader)
            the_uploader.start()
            
            time.sleep(1)
            return cam_cloud_pb2.StrMsg(msg='OK')  
                
        # only can control the current query 
        if (request.qid != self.qid):
            return cam_cloud_pb2.StrMsg(msg='FAIL')
                
        if request.command == "PAUSE":
            _GraceKillUploader() # otherwise yolo will keep running
            
            if not self.op_worker.isRunning():
                logger.warn("bug? worker not running")
                return cam_cloud_pb2.StrMsg(msg='FAIL')
            else:
                self.op_worker.pause()
                return cam_cloud_pb2.StrMsg(msg='OK')
        elif request.command == "RESUME":
            if self.op_worker.isRunning():
                logger.warn("bug? worker alredy running")
                return cam_cloud_pb2.StrMsg(msg='FAIL')
            else:
                self.op_worker.resume()
                _StartUploaderIfDead()
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
                                
        with the_queries_lock:
            if request.qid in the_queries:            
                n_frames_processed = the_queries[request.qid]['n_frames_processed']
                n_frames_total = the_queries[request.qid]['n_frames_total']
                status = the_queries[request.qid]['status']
                                        
        return cam_cloud_pb2.QueryProgress(qid = request.qid, 
                    video_name = self.video_name,
                    n_frames_processed = n_frames_processed,
                    n_frames_total = n_frames_total,
                    status = status)
        
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
                    '''
                    op_score = self.op.predict_image(frame, '350,0,720,400')
                    if (op_score <= 0.3):
                        counter += 1
                        continue
                    else:
                        # FIXME move the following computation to
                        # another thread
                        PQueue.put((op_score, frame))
                    '''
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

    # assumption: videos are stored as images in subdir, i.e. 
    # ${the_img_dirprefix}/${video_name}/{images}
    # $video_name may contain fps info, e.g. XX-10FPS_XXX
    # individual image files are numbered, number is img_fname[:-4]  
    
    # cf: 
    # https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/duration
    
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

def serve():    
    
    logger.info('Init camera service')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    diva_cam_servicer = DivaCameraServicer()
    cam_cloud_pb2_grpc.add_DivaCameraServicer_to_server(
        diva_cam_servicer, server)
    server.add_insecure_port(f'[::]:{CAMERA_CHANNEL_PORT}')
    server.start()
     
    _StartUploaderIfDead()
    
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        logger.warning('DivaCloud stop!!!')
        server.stop(0)

if __name__ == '__main__':
    serve()
