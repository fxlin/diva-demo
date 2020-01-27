import unittest
import os
import ffmpeg
import time

import grpc
import server_diva_pb2_grpc
import server_diva_pb2

from variables import DIVA_CHANNEL_ADDRESS

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from models.common import db_session, init_db
from models.frame import Frame, Status
from models.video import Video


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
            _ = stub.detect_object_in_video(
                server_diva_pb2.object_video_pair(
                    object_name='motorbike', video_name=self.SAMPLE_VIDEO))

        begin = time.time()

        rounds = 0

        while True:
            time.sleep(10)
            rounds += 1
            print(f'{rounds} rounds of test')
            all_frames = session.query(Frame).join(Video).filter(
                Video.name == self.SAMPLE_VIDEO).distinct().all()

            processed_frames = session.query(Frame).join(Video).filter(
                Video.name == self.SAMPLE_VIDEO).filter(
                    Frame.processing_status ==
                    Status.Finished).distinct().all()

            if len(processed_frames) == len(all_frames) != 0:
                break
            elif (time.time() - begin) > 60 * 5:
                print(
                    f"TIMEOUT of processing video {(time.time() - begin)/1000}"
                )
                break

        temp_res = session.query(Frame).join(Video).filter(
            Video.name == self.SAMPLE_VIDEO).all()
        self.assertNotEqual(
            len(temp_res), 0,
            f'Should have some records of frames associated to video {self.SAMPLE_VIDEO} in DB'
        )

        db_session.remove()


if __name__ == "__main__":
    init_db()
    unittest.main()