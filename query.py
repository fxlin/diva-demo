#  FIXME : come back and change the hardcoded value of how often to query
#  This file issues queries to the database and returns the difference between finished and either processing or initialized

import os
import sys

from sqlalchemy.orm.exc import MultipleResultsFound
from models.common import db_session
from models.video import Video
from models.frame import Frame, Status
from models.element import Element


def request_frames(video_name):
    session = db_session()
    session.begin()
    processed_frames = session.query(Frame.name).join(Video).filter(Video.name == video_name).filter(Frame.processing_status == Status.Finished).all()
    all_frames = session.query(Frame.name).join(Video).filter(Video.name == video_name).all()
    return processed_frames if len(processed_frames) == len(all_frames) else False


#  Test to see if it works properly
if __name__ == '__main__':
    print(request_frames('example.mp4'))