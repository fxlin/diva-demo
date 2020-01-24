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
    # init_db()
    session = db_session()
    session.begin()
    status = [i for i in session.query(Frame.processing_status).all()]
    id = [i for i in session.query(Frame.video_id).all()]
    name = [i for i in session.query(Frame.name).all()]
    num_frames = id.count(video_id)
    video_frames = list()
    for i in range(len(status)):
        if id[i] == video_id and status[i] == 3:
            video_frames.append(name[i])
    db_session.remove()
    return video_frames if num_frames == len(video_frames) else False


def request_videoID(name):
    # init_db()
    session = db_session()
    session.begin()
    video_name = [i for i in db_session.query(Video.name).all()]
    db_session.remove()
    return video_name.index(name) + 1 if video_name.index(name) != -1 else False


#  Test to see if it works properly
if __name__ == '__main__':
    init_db()
    print(request_frames(request_videoID('example.mp4')))