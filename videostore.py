import copy
import os
import sys
import shutil
import logging
import threading
import re
import typing

import PIL
import cv2
import numpy as np

import common_pb2

FORMAT = '%(levelname)8s {%(module)s:%(lineno)d} %(threadName)s %(message)s'
#FORMAT = '{%(module)s:%(lineno)d} %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# for a specific video. if backing dir does not exists, cr; otherwise, don't erase anything
class VideoStore():
    def __init__(self, video_name:str, prefix:str):
        self.lock= threading.Lock()
        self.prefix = prefix
        self.video_name = video_name
        self.video_abspath = os.path.join(prefix, video_name)
        self.minid = -1
        self.maxid = -1
        self.n_missing_frames = 0

        if not os.path.isdir(self.prefix):
            raise

        if not os.path.exists(self.video_abspath):
            os.mkdir(self.video_abspath)
            logger.info(f"mkdir {self.video_abspath}")
        elif not os.path.isdir(self.video_abspath):
            logger.error(f"failed to open videostore: {self.video_abspath} exists but not dir")
            #sys.exit(1)
            raise

        # use hint to find FPS from video name. best effort
        m = re.search(r'''\D*(\d+)(FPS|fps)''', video_name)
        if m:
            self.fps = int(m.group(1))
        else:
            self.fps = -1

        # NB: listdir can be very slow
        # a list of frame nums without file extensions. all leading 0s are removed
        self.frame_ids = [int(img[:-4]) for img in os.listdir(self.video_abspath)]

        if len(self.frame_ids) > 0:
            self.minid = min(self.frame_ids)
            self.maxid = max(self.frame_ids)
            diff = self.maxid - self.minid + 1
            self.n_missing_frames = diff - len(self.frame_ids)
            assert (self.n_missing_frames >= 0)

    # return a deepcopy of frame ids
    def GetFrameIds(self):
        with self.lock:
            return copy.deepcopy(self.frame_ids)

    def GetNumFrames(self):
        with self.lock:
            return len(self.frame_ids)

    def CleanStoredFrames(self):
        with self.lock:
            try:
                shutil.rmtree(self.video_abspath)
                os.mkdir(self.video_abspath)
                logger.info(f"cleaned {self.video_abspath}")
            except Exception as e:
                logger.error(e)

    # unlocked. caller must hold self.lock
    def _GetFramePath(self, frame_id:int, must_exist:bool = True) -> str:
        frame_path = None
        found = False
        for ext in ['.JPG', '.jpg']:
            for frame_fname in [f'{frame_id:07d}', f'{frame_id:06d}', f'{frame_id:d}', f'{frame_id:08d}']:
                frame_path = os.path.join(self.video_abspath, frame_fname + ext)
                if os.path.isfile(frame_path):
                    found = True
                    # logger.info(f"try {frame_path}... found")
                    break
                else:
                    # print(f"try {frame_path}... not found")
                    pass
            else:
                continue
            break

        if found:
            return frame_path
        elif not must_exist:
            return os.path.join(self.video_abspath, f'{frame_id:07d}' + '.jpg')
        else:
            return None

    # if found, return that path. if not found, gen a path (one out of multiple legal ones)
    # by design should NOT made public. do so right now for cv2.imread()
    def GetFramePath(self, frame_id: int, must_exist: bool = True) -> str:
        with self.lock:
            return self._GetFramePath(frame_id, must_exist)

    def GetFrame(self, frame_id:int) -> common_pb2.Image:
        #print('to grab lock...')
        with self.lock:
            #print('lock grabbed')
            frame_path = self._GetFramePath(frame_id, must_exist=True)

            if not frame_path:
                return common_pb2.Image()  # nothing

            try:
                f = open(frame_path, 'rb')
                _height, _width, _chan = 0, 0, 0
                #print('returned 2')
                return common_pb2.Image(data=f.read(),
                                        height=_height,
                                        width=_width,
                                        channel=_chan)
            except Exception as e:
                logger.error(e)
                return common_pb2.Image()  # nothing

    # the video dir must exist
    # return: saved abs path
    # res [x,y] crop (left, upper, right, and lower pixel coordinate)
    # cf: https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
    def StoreFrame(self, frame_id:int, img:common_pb2.Image, res = None, crop = None) -> str:
        with self.lock:
            try:
                frame_path = self._GetFramePath(frame_id, must_exist=False)
                with open(frame_path, 'wb') as f:
                    f.write(img.data)
                # write and readback: silly but works...
                if res or crop:
                    im = PIL.Image.open(frame_path)
                if res:
                    im.thumbnail(res)  # 128x128 thumbnail
                if crop:
                    im.crop(crop)
                im.save(frame_path)
                return frame_path
            except Exception as err:
                logger.error(err)
                return None

    def StoreFrameBBoxes(self, frame_id:int, img:common_pb2.Image, elements) -> str:
        img_data = img.data
        raw_img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)

        # render bboxes
        for idx, ele in enumerate(elements):
            x1, y1, x2, y2 = int(ele.x1), int(ele.y1), int(ele.x2), int(ele.y2)
            cv2.rectangle(raw_img, (x1, y1), (x2, y2),
                          (0, 255, 0), 3)
            # print("draw--->", x1, y1, x2, y2)

        with self.lock:
            try:
                frame_path = self._GetFramePath(frame_id, must_exist=False)
                cv2.imwrite(frame_path, raw_img)
                return frame_path
            except Exception as err:
                logger.error(err)
                return None

class VideoLib():
    def __init__(self, prefix:str):
        self.lock = threading.Lock()
        self.prefix = prefix
        self.videos: typing.Dict[str, VideoStore] = {}

        if not os.path.isdir(self.prefix):
            raise

    # return: ref to the video store
    # if the video store exists, do nothing.
    def AddVideoStore(self, video_name:str):
        with self.lock:
            if not video_name in self.videos:
                vs = VideoStore(video_name, self.prefix)
                self.videos[video_name] = vs
            return self.videos[video_name]

    def GetVideoStore(self, video_name:str):
        with self.lock:
            if video_name in self.videos:
                return self.videos[video_name]
            else:
                return None


    def RemoveVideo(self):
        pass

