import unittest
import os
import time

import cv2
import grpc
import server_diva_pb2_grpc
import server_diva_pb2
import cam_cloud_pb2_grpc
import cam_cloud_pb2

from variables import DIVA_CHANNEL_ADDRESS, CONTROLLER_VIDEO_DIRECTORY
from variables import CAMERA_CHANNEL_ADDRESS, CAMERA_CHANNEL_PORT

from sqlalchemy.orm import Session
from models.common import db_session, init_db
from models.frame import Frame, Status
from models.video import Video, VideoStatus
from models.camera import Camera


class TestProcessVideo(unittest.TestCase):
    SOURCE_VIDEO = 'example.mp4'
    SAMPLE_VIDEO = 'temp_video.mp4'
    SAMPLE_FOLDER = os.path.join('video')

    def setUp(self):
        # session = db_session()
        # v = Video(self.SAMPLE_VIDEO, p)
        # session.add(v)
        # session.commit()
        # db_session.remove()
        pass

    def tearDown(self):
        # p = os.path.join(self.SAMPLE_FOLDER, self.SAMPLE_VIDEO)
        # if os.path.exists(p):
        #     os.remove(p)

        # session = db_session()
        # session.query(Video).filter(Video.name == self.SAMPLE_VIDEO).delete()

        # # FIXME test
        # temp = session.query(Video).filter(
        #     Video.name == self.SAMPLE_VIDEO).all()
        # print(temp)
        # print(session.query(Frame).count())
        pass

    def test_camera_controller_integration(self):
        """
        (operator) -> selected images (camera) --> YOLO (controller) 
            --> store video info in DB (controller)
            --> reply to camera (controller) (optional)
        """

        # with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        #     stub = server_diva_pb2_grpc.server_divaStub(channel)

        #     c_ip = CAMERA_CHANNEL_ADDRESS.split(':')[0]
        #     c_port = CAMERA_CHANNEL_ADDRESS.split(':')[1]
        #     temp_image = cv2.imread(os.path.join('test', 'sample_364.jpeg'))
        #     cam_payload = server_diva_pb2.camera_info(camera_ip=c_ip,
        #                                               camera_port=c_port,
        #                                               name='test_camera')
        #     req = server_diva_pb2.frame_from_camera(image=temp_image,
        #                                             confidence_score=0.3,
        #                                             camera=cam_payload,
        #                                             timestamp=time.time(),
        #                                             video_name='example.mp4',
        #                                             offset=7)
        #     resp = stub.detect_object_in_frame(req)
        #     # FIXME check resp payload
        pass

    def test_download_video_from_camera(self):
        """
        select one image and request related video clip (controller)
            --> reply by uploading video clip (camera)
        """

        v_name = 'example.mp4'

        with grpc.insecure_channel(CAMERA_CHANNEL_ADDRESS) as channel:
            stub = cam_cloud_pb2_grpc.DivaCameraStub(channel)
            req = cam_cloud_pb2.VideoRequest(timestamp=int(time.time()),
                                             video_name=v_name,
                                             offset=7)
            resp = stub.DownloadVideo(req)

            v_path = os.path.join('/tmp', v_name)

            with open(v_path, 'w') as fptr:
                fptr.write(resp.video.data.decode())

        # message VideoRequest {
        #     int32 timestamp = 1;
        #     int32 offset = 2;
        #     string video_name = 3;
        # }

        # message VideoResponse {
        #     string msg = 1;
        #     int32 status_code = 2;
        #     common.File video = 3;
        # }

    def test_process_video(self):
        pass

        # session: Session = db_session()
        # # FIXME video id? which video?
        # res = session.query(Frame).join(Video).filter(
        #     Video.name == self.SAMPLE_VIDEO).all()

        # self.assertEqual(
        #     len(res), 0, f'Should not exist any frame associated to video \
        #          {self.SAMPLE_VIDEO} yet.')

        # with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
        #     stub = server_diva_pb2_grpc.server_divaStub(channel)
        #     _ = stub.detect_object_in_video(
        #         server_diva_pb2.object_video_pair(
        #             object_name='motorbike', video_name=self.SAMPLE_VIDEO))

        # begin = time.time()

        # rounds = 0

        # while True:
        #     time.sleep(10)
        #     rounds += 1
        #     print(f'{rounds} rounds of test')
        #     all_frames = session.query(Frame).join(Video).filter(
        #         Video.name == self.SAMPLE_VIDEO).distinct().all()

        #     processed_frames = session.query(Frame).join(Video).filter(
        #         Video.name == self.SAMPLE_VIDEO).filter(
        #             Frame.processing_status ==
        #             Status.Finished).distinct().all()

        #     if len(processed_frames) >= (len(all_frames) //
        #                                  2) and len(all_frames) != 0:
        #         break
        #     elif (time.time() - begin) > 60 * 5:
        #         print(
        #             f"TIMEOUT of processing video {(time.time() - begin)/1000}"
        #         )
        #         break

        # temp_res = session.query(Frame).join(Video).filter(
        #     Video.name == self.SAMPLE_VIDEO).all()
        # self.assertNotEqual(
        #     len(temp_res), 0,
        #     f'Should have some records of frames associated to video {self.SAMPLE_VIDEO} in DB'
        # )

        # db_session.remove()


if __name__ == "__main__":
    init_db()
    unittest.main()