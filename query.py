#  FIXME : come back and change the hardcoded value of how often to query
#  This file issues queries to the database and returns the difference between finished and either processing or initialized

import os
import sys

from sqlalchemy.orm.exc import MultipleResultsFound
from models.common import db_session, init_db
from models.video import Video
from models.frame import Frame, Status
from models.element import Element


def request_frames(video_id):
    session = db_session()
    session.begin()
    frame_status = [i for i in session.query(Frame.processing_status).filter(Frame.video_id == video_id)]
    name = [i for i in session.query(Frame.name).filter(Frame.video_id == video_id)]
    for status in frame_status:
        if status != Status.Finished:
            db_session.remove()
            return False
    db_session.remove()
    return name


def request_videoID(name):
    session = db_session()
    session.begin()
    video_id = [i for i in db_session.query(Video.id).filter(Video.name == name)][0]
    db_session.remove()
    return video_id if video_id > 0 else False


#  Test to see if it works properly
if __name__ == '__main__':
    init_db()
    print(request_frames(request_videoID('example.mp4')))