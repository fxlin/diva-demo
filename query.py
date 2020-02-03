#  This file issues queries to the database and returns the difference between finished and either processing or initialized

from models.common import db_session
from models.video import Video
from models.frame import Frame, Status
from models.element import Element


def request_frames(video_name):
    processed_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name) \
        .filter(Frame.processing_status == Status.Finished).all()
    processed_frames = [value for (value,) in processed_frames]
    failed_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name) \
        .filter(Frame.processing_status == Status.Failed).all()
    all_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name).all()
    db_session.close()
    print(processed_frames, all_frames, failed_frames)
    return processed_frames if len(processed_frames + failed_frames) == len(all_frames) else False


def request_coordinates(frame_id, time):
    coordinates = db_session.query(Element.box_coordinate).join(Video).filter(Frame.id == frame_id).limit(300).all()
    db_session.close()
    return coordinates


#  Test to see if it works properly
if __name__ == '__main__':
    print(request_frames('example.mp4'))