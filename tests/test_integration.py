import unittest
import os
import ffmpeg

import grpc
import server_diva_pb2_grpc
import server_diva_pb2

from variables import DIVA_CHANNEL_ADDRESS

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from models.common import db_session
from models.frame import Frame, Status
from models.video import Video


class TestProcessVideo(unittest.TestCase):
    SOURCE_VIDEO = 'example.mp4'
    SAMPLE_VIDEO = 'temp_video.mp4'
    SAMPLE_FOLDER = os.path.join('video')

    @classmethod
    def setUpClass(cls):
        source_path = os.path.join(cls.SAMPLE_FOLDER, cls.SOURCE_VIDEO)
        p = os.path.join(cls.SAMPLE_FOLDER, cls.SAMPLE_VIDEO)

        source_video = ffmpeg.input(source_path)

        # 00:00:10 - 00:00:20
        # FPS is 30
        FPS = 30
        source_video.trim(start_frame=10 * FPS,
                          end_frame=20 * FPS).output(p).run()

        del source_video

        session = db_session()
        v = Video(cls.SAMPLE_VIDEO, source_path)
        session.add(v)
        session.commit()

        db_session.remove()

    @classmethod
    def tearDownClass(cls):
        p = os.path.join(cls.SAMPLE_FOLDER, cls.SAMPLE_VIDEO)
        if os.path.exists(p):
            os.remove(p)

        session = db_session()
        session.query(Frame).join(Video).filter(
            Video.name == cls.SAMPLE_VIDEO).delete()

        # FIXME test
        temp = session.query(Video).filter(
            Video.name == cls.SAMPLE_VIDEO).all()
        print(temp)

    def test_process_video(self):
        session: Session = db_session()
        # FIXME video id? which video?
        res = session.query(Frame).join(Video).filter(
            Video.name == self.SAMPLE_VIDEO).all()

        self.assertEqual(
            len(res), 0,
            f'Should not exist any frame associated to video {self.SAMPLE_VIDEO} yet.'
        )

        with grpc.insecure_channel(DIVA_CHANNEL_ADDRESS) as channel:
            stub = server_diva_pb2_grpc.server_divaStub(channel)
            response = stub.request_frame_path(
                server_diva_pb2.query_statement(name='request directory'))

        while True:
            sleep(5)
            all_frames = session.query(Frame).join(Video).filter(
                Video.name == self.SAMPLE_VIDEO).distinct().all()

            processed_frames = session.query(Frame).join(Video).filter(
                Video.name == self.SAMPLE_VIDEO).filter(
                    Frame.processing_status ==
                    Status.Finished).distinct().all()

            if len(processed_frames) == len(all_frames):
                break

        temp_res = session.query(Frame).join(Video).filter(
            Video.name == self.SAMPLE_VIDEO).all()
        self.assertNotEqual(
            len(temp_res), 0,
            f'Should have some records of frames associated to video {self.SAMPLE_VIDEO} in DB'
        )

        db_session.remove()


if __name__ == "__main__":
    unittest.main()