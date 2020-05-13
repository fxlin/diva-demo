from models.common import db_session
from models.video import Video
from models.frame import Frame, Status
from models.element import Element

def num_frames(video_name):
    processed_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name) \
        .filter(Frame.processing_status == Status.Finished).all()
    all_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name).all()
    db_session.close()
    return len(processed_frames), len(all_frames)

def request_frames(video_name):
    processed_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name) \
        .filter(Frame.processing_status == Status.Finished).all()
    failed_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name) \
        .filter(Frame.processing_status == Status.Failed).all()
    all_frames = db_session.query(Frame.name).join(Video).filter(Video.name == video_name).all()
    db_session.close()
    print('# processed:', len(processed_frames), '# failed', len(failed_frames), '# total:', len(all_frames))
    #  This line is [val for (val,) in processed_frames] extracts only the name of the image
    #  len(processed_frames + failed_frames) == len(all_frames) checks if it is done processing frames
    val = [val for (val,) in processed_frames]
    return val, len(processed_frames + failed_frames) == len(all_frames)



def request_coordinates(video_name):
    coordinates = db_session.query(Element.box_coordinate, Element.frame_id).join(Video).filter(Video.name == video_name).all()
    time = db_session.query(Frame.name, Frame.id).join(Video).filter(Video.name == video_name).all()
    coordinates.sort(key=lambda x: x[1])
    time.sort(key=lambda x: x[1])
    result = {int(t) / 10: coord for (coord, _), (t, _) in zip(coordinates, time)}
    db_session.close()
    return result


#  Test to see if it works properly
if __name__ == '__main__':
    print(request_frames('example.mp4'))