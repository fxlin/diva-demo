'''

our web server using bokeh

display image:
cf: https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/image_url.html

bokeh.server.contexts
https://docs.bokeh.org/en/latest/docs/reference/server/contexts.html

various widgets
https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html#div

range slider (default)
https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html#tabs
custom double ended slider
https://docs.bokeh.org/en/latest/docs/user_guide/extensions_gallery/widget.html

hover tooltip with pic
https://docs.bokeh.org/en/latest/docs/user_guide/tools.html?highlight=custom%20tooltip
https://discourse.bokeh.org/t/hovertool-displaying-image/1198/7

'''
from __future__ import print_function

import copy
import time

import numpy as np

from bokeh.driving import count
from bokeh.layouts import column, gridplot, row
from bokeh.models import Select, Slider, Button, ImageURL, Plot, LinearAxis, Paragraph, Div, Range1d, Text, LabelSet, Title
from bokeh.models import CDSView, IndexFilter, RadioButtonGroup

# palettes https://docs.bokeh.org/en/latest/docs/reference/palettes.html
from bokeh.palettes import RdYlGn10, Spectral6
from bokeh.transform import linear_cmap


from bokeh.plotting import curdoc, figure

from bokeh  .models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from datetime import date
from random import randint

# xzl
import sys
import threading
import logging
from dataclasses import asdict
from functools import partial
from tornado import gen

import grpc
import server_diva_pb2
import server_diva_pb2_grpc

import json
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

#import server.tables as tables
#import server.forms as forms
#import server.control as cloud
#
import control
from control import VideoInfo

#from .control import list_videos_cam
#from .control import query_submit

#from .control import list_videos_cam
#from control import query_submit

from variables import *

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
#logger = logging.getLogger(__name__)
logger = logging.getLogger()

np.random.seed(1)

doc = curdoc()

#######################################

PLOT_IMG_WIDTH = 1000
PLOT_IMG_HEIGHT = 200

# for results
#PER_IMG_WIDTH = 200 # for image spacing
PER_IMG_WIDTH = 300 # for image spacing
PER_IMG_HEIGHT = 100 # only for determining label offset
PER_IMG_SPACING = int(PER_IMG_WIDTH * 1.1)
PLOT_IMG_HEIGHT = PER_IMG_HEIGHT + 20 # some margin for frame info


FULL_IMG_WIDTH = 1280 >> 1
FULL_IMG_HEIGHT = 720 >> 1

PER_PREVIEW_IMG_WIDTH = 128
PER_PREVIEW_IMG_HEIGHT = 72
PER_PREVIEW_IMG_SPACING = int(PER_PREVIEW_IMG_WIDTH * 1.1)
PLOT_PREVIEW_HEIGHT = PER_PREVIEW_IMG_HEIGHT + 20 # some margin for frame info

PER_IMG_720P_HEIGHT = int(720/(1280/PER_IMG_WIDTH)) # 720p image scaled height

N_PREVIEWS = 5
N_RESULTS = 20 # show in table
N_RESULTS_IMG = 5

BUTTON_WIDTH = 100

UPDATE_INTERVAL_MS = 2000   # short interval like 500ms will cause unresponsive UI. XXX opt parsing framemaps
the_interval_ms = UPDATE_INTERVAL_MS

PLACEHOLDER = ['test1', 'test2', 'test3', 'test4', 'test5']

#######################################

MA12, MA26, EMA12, EMA26 = '12-tick Moving Avg', '26-tick Moving Avg', '12-tick EMA', '26-tick EMA'

source = ColumnDataSource(dict(
    time=[], average=[], low=[], high=[], open=[], close=[],
    ma=[], macd=[], macd9=[], macdh=[], color=[]
))

p = figure(plot_height=500, tools="xpan,xwheel_zoom,xbox_zoom,reset", x_axis_type=None, y_axis_location="right")
p.x_range.follow = "end"
p.x_range.follow_interval = 100
p.x_range.range_padding = 0

p.line(x='time', y='average', alpha=0.2, line_width=3, color='navy', source=source)
p.line(x='time', y='ma', alpha=0.8, line_width=2, color='orange', source=source)
p.segment(x0='time', y0='low', x1='time', y1='high', line_width=2, color='black', source=source)
p.segment(x0='time', y0='open', x1='time', y1='close', line_width=8, color='color', source=source)

p2 = figure(plot_height=250, x_range=p.x_range, tools="xpan,xwheel_zoom,xbox_zoom,reset", y_axis_location="right")
p2.line(x='time', y='macd', color='red', source=source)
p2.line(x='time', y='macd9', color='blue', source=source)
p2.segment(x0='time', y0=0, x1='time', y1='macdh', line_width=6, color='black', alpha=0.5, source=source)

mean = Slider(title="mean", value=0, start=-0.01, end=0.01, step=0.001)
stddev = Slider(title="stddev", value=0.04, start=0.01, end=0.1, step=0.01)
mavg = Select(value=MA12, options=[MA12, MA26, EMA12, EMA26])

################
# pause/resume
################

b_pause = Button(label="Pause", button_type="success", width=BUTTON_WIDTH, disabled=True)
b_resume = Button(label="Resume", button_type="success", width=BUTTON_WIDTH, disabled=True)

the_started = False
the_paused = False

the_qid = -1
the_video_name = "chaweng-1_10FPS"
the_video_info = None
the_all_classes = ['car','bus','truck','train','person','bicycle']
the_class = the_all_classes[0]

the_frameskip_choices = [1, 5, 10, 20, 100]
the_frameskip = the_frameskip_choices[0]

the_videos:typing.List[VideoInfo] = None

def cb_resume():
    global the_paused, the_poll_cb
    if the_paused:
        console_write("resume query requested")
        msg = control.query_resume()
        console_append("cam replied:" + msg)
        if msg.startswith("OK"):
            the_paused = False
            the_poll_cb = doc.add_periodic_callback(periodic_callback, 500)
            b_resume.disabled = True
            b_pause.disabled = False
    else:
        logger.error("bug?")

def cb_pause():
    global the_paused, the_poll_cb

    if not the_paused:
        console_write("pause query requested")
        msg = control.query_pause()
        console_append("pause query. cam replied:" + msg)
        if msg.startswith("OK"):
            the_paused = True
            doc.remove_periodic_callback(the_poll_cb)
            b_pause.disabled = True
            b_resume.disabled = False
    else:
        logger.error("bug?")

b_pause.on_click(cb_pause)
b_resume.on_click(cb_resume)

######
# the plot showing progress. eg  n_frames_processed_cam
######

src_cam_init = dict(
    ts=[],
    n_frames_processed_cam=[],
    n_frames_processed_cam_rate =[],
    n_frames_processed_yolo_rate=[],
    n_frames_recv_yolo_rate=[], # means positive results
    n_mbytes_cam_sent_rate=[],
    cam_mem_usage_percent=[]
)

src_cam = ColumnDataSource(src_cam_init)

# tools="xpan,xwheel_zoom,xbox_zoom,reset",
#  y_axis_location="right"
pcam = figure(plot_height=100, plot_width=PLOT_IMG_WIDTH>>3, toolbar_location=None,
              x_axis_type=None, y_axis_type=None, title="PosFr/sec")
pcam.x_range.follow = "end"
pcam.x_range.follow_interval = 100
pcam.x_range.range_padding = 0

# pcam.line(x='ts', y='n_frames_processed_cam', alpha=0.8, line_width=2, color='red', source=src_cam)
pcam.line(x='ts', y='n_frames_recv_yolo_rate', alpha=0.8, line_width=2, color='green', source=src_cam)

xaxis = LinearAxis()
xaxis.minor_tick_line_color = None
pcam.add_layout(xaxis, 'below')

yaxis = LinearAxis()
yaxis.minor_tick_line_color = None
pcam.add_layout(yaxis,'left')

## showing cam processing rate
#  y_axis_location="right"
# x_axis_type=None
pcam1 = figure(plot_height=100, plot_width=PLOT_IMG_WIDTH>>3, toolbar_location=None,
               x_axis_type=None, y_axis_type=None, title="CamProcFr/sec")
pcam1.x_range.follow = "end"
pcam1.x_range.follow_interval = 100
pcam1.x_range.range_padding = 0

xaxis = LinearAxis()
xaxis.minor_tick_line_color = None
pcam1.add_layout(xaxis, 'below')

yaxis = LinearAxis()
yaxis.minor_tick_line_color = None
pcam1.add_layout(yaxis,'left')

pcam1.line(x='ts', y='n_frames_processed_cam_rate', alpha=0.8, line_width=2, color='red', source=src_cam)

#############
## show yolo processing rate
pcam2 = figure(plot_height=100, plot_width=PLOT_IMG_WIDTH>>3, toolbar_location=None,
               x_axis_type=None, y_axis_type=None, title="CloudProcFr/sec")
pcam2.x_range.follow = "end"
pcam2.x_range.follow_interval = 100
pcam2.x_range.range_padding = 0

xaxis = LinearAxis()
xaxis.minor_tick_line_color = None
pcam2.add_layout(xaxis, 'below')

yaxis = LinearAxis()
yaxis.minor_tick_line_color = None
pcam2.add_layout(yaxis,'left')

pcam2.line(x='ts', y='n_frames_processed_yolo_rate', alpha=0.8, line_width=2, color='blue', source=src_cam)

#############
## show cam upload data rate
pcam3 = figure(plot_height=100, plot_width=PLOT_IMG_WIDTH>>3, toolbar_location=None,
               x_axis_type=None, y_axis_type=None, title="CamUpMB/sec")
pcam3.x_range.follow = "end"
pcam3.x_range.follow_interval = 100
pcam3.x_range.range_padding = 0

xaxis = LinearAxis()
xaxis.minor_tick_line_color = None
pcam3.add_layout(xaxis, 'below')

yaxis = LinearAxis()
yaxis.minor_tick_line_color = None
pcam3.add_layout(yaxis,'left')

pcam3.line(x='ts', y='n_mbytes_cam_sent_rate', alpha=0.8, line_width=2, color='red', source=src_cam)

#############
## show cam mem usage
pcam4 = figure(plot_height=100, plot_width=PLOT_IMG_WIDTH>>3, toolbar_location=None,
               x_axis_type=None, y_axis_type=None, title="CamMemPercent")
pcam4.x_range.follow = "end"
pcam4.x_range.follow_interval = 100
pcam4.x_range.range_padding = 0

xaxis = LinearAxis()
xaxis.minor_tick_line_color = None
pcam4.add_layout(xaxis, 'below')

yaxis = LinearAxis()
yaxis.minor_tick_line_color = None
pcam4.add_layout(yaxis,'left')

pcam4.line(x='ts', y='cam_mem_usage_percent', alpha=0.8, line_width=2, color='red', source=src_cam)



# not in use. periodic callback. perhaps simpler
@count()
def update_camsrc(t):
    logger.info(f"update_camsrc checkgin....{t}")
    prog = control.get_latest_query_progress(qid=0)
    if not prog: return

    new_data = dict(
        ts = [prog.ts],
        n_frames_processed_cam = [prog.n_frames_processed_cam],
        #n_frames_recv_yolo = [prog.n_frames_recv_yolo]
        n_frames_recv_yolo=[0]
    )

    print(new_data)
    src_cam.stream(new_data, rollover=300)

# curdoc().add_periodic_callback(update_camsrc, 500)

##############################
# test: using a separate thread to poll progress on demand
# todo: call all progress update from here
##############################

# dirty
the_last_n_frames_processed_cam = -1
the_last_n_bytes_sent_cam = -1
the_last_ts = -1

# update the aggregated stats & framemap
# scheduled by worker thread, executed in the main thread
@gen.coroutine
def cb_update_datasrc(prog, stats, res):  # stats: QueryInfoCloud
    global UPDATE_INTERVAL_MS, the_interval_ms

    logger.info(f">>>>>>>>>>>>> cb_update_datasrc started")

    t0 = time.time()

    global the_last_n_frames_processed_cam, the_last_n_frames_recv_yolo, the_last_n_frames_processed_yolo, the_last_ts, the_start_ts
    global the_last_n_bytes_sent_cam

    if the_last_ts < 0: # once
        n_frames_processed_cam_rate = 0
        n_frames_processed_yolo_rate = 0
        n_frames_recv_yolo_rate = 0
        n_mbytes_cam_sent_rate = 0
        the_start_ts = prog.ts
    else:
        #n_frames_processed_cam_rate = (prog.n_frames_processed_cam - the_last_n_frames_processed_cam) / (prog.ts - the_last_ts)
        n_frames_processed_cam_rate = (stats.n_frames_processed_cam - the_last_n_frames_processed_cam) / (
                    prog.ts - the_last_ts)
        n_frames_processed_yolo_rate = (stats.n_frames_processed_yolo - the_last_n_frames_processed_yolo) / (
                prog.ts - the_last_ts)
        n_frames_recv_yolo_rate = (stats.n_frames_recv_yolo - the_last_n_frames_recv_yolo) / (
            prog.ts - the_last_ts)
        n_mbytes_cam_sent_rate = (stats.n_bytes_sent_cam - the_last_n_bytes_sent_cam) /  (
            prog.ts - the_last_ts) / 1024 / 1024

    the_last_n_frames_processed_cam = stats.n_frames_processed_cam
    the_last_n_frames_processed_yolo = stats.n_frames_processed_yolo
    the_last_n_frames_recv_yolo = stats.n_frames_recv_yolo
    the_last_n_bytes_sent_cam = stats.n_bytes_sent_cam
    the_last_ts = prog.ts

    new_data = dict(
        ts=[prog.ts - the_start_ts],
        n_frames_processed_cam=[stats.n_frames_processed_cam],
        n_frames_processed_cam_rate=[n_frames_processed_cam_rate],
        n_frames_processed_yolo_rate=[n_frames_processed_yolo_rate],
        #n_frames_recv_yolo=[prog.n_frames_recv_yolo]
        n_frames_recv_yolo_rate=[n_frames_recv_yolo_rate],
        n_mbytes_cam_sent_rate=[n_mbytes_cam_sent_rate],
        cam_mem_usage_percent=[stats.cam_mem_usage_percent]
    )
    #print(new_data)
    src_cam.stream(new_data, rollover=300)

    cb_update_span(prog.framestates) # ... expensive

    t1 = time.time()

    cb_update_res(res)

    t2 = time.time()

    logger.info(f"<<<<<< cb_update_datasrc ended {(t1-t0)*1000:.2f} {(t2-t1)*1000:.2f} ms")

    # control the rate of update
    '''
    if t2-t0 > 0.1: # too high
        the_interval_ms *= 1.5
    else:
        the_interval_ms = max(the_interval_ms - 100, UPDATE_INTERVAL_MS)
    '''

ev_progress = threading.Event()

def thread_progress():
    logger.info("started")
    while True:
        ev_progress.wait()
        ev_progress.clear()

        # pull updates from control bkend...
        t0 = time.time()
        prog = control.create_query_progress_snapshot(qid=the_qid)
        stat = control.query_progress(qid=the_qid) # stats
        res = control.query_results(the_qid)
        t1 = time.time()

        if not prog:
            continue

        # update/render on demand
        if prog:
            doc.add_next_tick_callback(partial(cb_update_datasrc, prog, stat, res))
            logger.info(f"pull backend update... {1000*(t1-t0)} ms")
        else:
            logger.info("poll progress... failed")

thread_progress = threading.Thread(target=thread_progress)
thread_progress.start()

the_poll_cb = None

# will be called periodically once query in progress.. from main thread
@count()
def periodic_callback(t):
    global the_poll_cb, the_interval_ms
    ev_progress.set() # async. kick the worker thread, which will pull the query progress from cam
    #cb_update_res(t) # sync. pull & update query res from cloud (not cam). sync
    the_poll_cb = doc.add_timeout_callback(periodic_callback, int(the_interval_ms))
    logger.info(f"schedule next update in {the_interval_ms} ms")

######
# text console
######
#para_log = Paragraph(text="hello world", width=200, height=100)
para_log = Div(text="[log messages]", width=300, height=50, background='lightgrey')

def console_append(msg:str):
    #para_log.text += '<ul>{}</ul>'.format(msg)
    para_log.text += '{}<br>'.format(msg)

def console_clear():
    para_log.text = ''

def console_write(msg:str):
    console_clear()
    console_append(msg)

######
# start a sample query
######

def cb_newquery():
    global the_poll_cb, the_started, the_qid, the_video_info, the_videos
    global data_span, source_single_res, source_results, src_cam
    '''
    the_videos = control.list_videos()    
    for v in the_videos:
        if v.video_name == the_video_name:
            the_video_info = v
            break
    else:
        logger.error(f"cannot find video {the_video_name}")
        raise
    '''

    # abort an ongoing query
    if the_started:
        msg = control.query_reset()
        console_write(f"reset cam requested. cam said {msg}")
        if msg.startswith('OK'):
            the_started = False
            #b_query.label="Query"
            update_b_query_label()
            b_pause.disabled = True
        return

    # new query...
    if not the_video_name or not the_video_info:
        logger.error(f"cannot find video {the_video_name}")
        console_write("cannot run query: no video selected")
        return

    # assumption:
    # video naming: video is named as ${scene}-X_XXfps-...
    #   where ${scene} is like "purdue", "chaweng", ...
    # opname: ops on the camera side are named as ${scene}-0, ${scene}-1, ${scene}-2...
    # op num: 0, 1, 2. .. up to 5

    scene = the_video_name.split('-')[0]
    assert(len(scene) > 0)

    op_names = [f'{scene}-{x}' for x in range(0,6)]  # up to 6 ops

    the_qid = control.query_submit(video_name=the_video_name,
                 op_names=op_names, crop='-1,-1,-1,-1', target_class=the_class, frameskip=the_frameskip)
    # print('qid:', resp)
    console_append(f"new query started. qid {the_qid}")
    #the_poll_cb = doc.add_periodic_callback(periodic_callback, UPDATE_INTERVAL_MS)
    the_poll_cb = doc.add_timeout_callback(periodic_callback, the_interval_ms)

    # clear any res from prev query
    data_span = data_span_init
    source_single_res.data = source_single_res_init
    source_results = source_results_init
    src_cam.data = src_cam_init

    the_started = True
    b_query.label="Abort"

    b_pause.disabled = False

b_query = Button(label="Query", button_type="danger", width=BUTTON_WIDTH*2)
b_query.on_click(cb_newquery)

# will return ??:?? if incomplete info
def frameid_to_vtime(id:int) -> str:
    if the_video_info and the_video_info.fps > 0:
        d = int(id/the_video_info.fps)
        return f'{int(d/3600):02d}:{int((d%3600)/60):02d}:{int(d%60):02d}'
    else:
        return '??:??'

def update_b_query_label():
    v = the_video_name if the_video_name else '??'
    fs = the_frameskip if the_frameskip > 0 else '??'
    c = the_class

    nf = the_video_info.frame_id_max - the_video_info.frame_id_min + 1 - the_video_info.n_missing_frames
    '''
    if the_video_info.fps > 0:
        d = int(nf/the_video_info.fps)
        dstr = f'{int(d/3600)}:{int((d%3600)/60)}:{int(d%60)}'
    else:
        dstr = '??:??'
    '''
    dstr = frameid_to_vtime(nf)
    nf /= fs

    b_query.label=f'Query:{v}/{c}/skip={fs} frames={int(nf)} dur={dstr}'

######
# list videos
######

cds_videos = {}
source_videos = ColumnDataSource(data=cds_videos)

columns = [
        TableColumn(field="video_name", title="Name"),
        TableColumn(field="n_frames", title="Frames"),
        TableColumn(field="n_missing_frames", title="Missing"),
        TableColumn(field="fps", title="fps"),
        TableColumn(field="duration", title="duration"),
        TableColumn(field="frame_id_min", title="min"),
        TableColumn(field="frame_id_max", title="max"),
    ]
table_listvideos = DataTable(source=source_videos, columns=columns,
                             width=400, height=100)

def show_cam_specs():
    console_clear()
    console_append('requested to list videos')
    msg = control.cam_specs()
    console_append(msg)

def cb_listvideos():
    global cds_videos
    global the_videos

    console_clear()
    console_append('requested to list videos')

    the_videos = control.list_videos()
    if len(the_videos) == 0:
        logger.error("nothing!")
        return

    console_append(f'{len(the_videos)} found')

    # convert to dicts
    vds = [asdict(v) for v in the_videos]

    # gen cds
    cds_videos = {}
    for k in vds[0].keys():
        cds_videos[k] = list(x[k] for x in vds)

    print(cds_videos)

    source_videos.data = cds_videos  # commit

b_lv = Button(label="ListVideos", button_type="success", width=BUTTON_WIDTH)
b_lv.on_click(cb_listvideos)

def cb_listvideo_table(attraname, old, new):
    global the_video_name, the_video_info
    # print(f"selected! {attraname} {old} {new}")
    i = source_videos.selected.indices[0]
    the_video_name = the_videos[i].video_name
    the_video_info = the_videos[i]
    console_write(f'selected video {i} {the_video_name}')
    if len(the_videos) >= i + 1:
        load_preview_frames(the_videos[i], N_PREVIEWS)
        update_b_query_label()

source_videos.selected.on_change('indices', cb_listvideo_table)

######
# list queries
######

cds_queries = {}
source_queries = ColumnDataSource(data=cds_queries) # init cds

table_listqueries = DataTable(
    source=source_queries,
    columns = [
        TableColumn(field="qid", title="qid"),
        TableColumn(field="status_cloud", title="Cloud"),
        TableColumn(field="status_cam", title="Cam"),
        TableColumn(field="target_class", title="Class"),
        TableColumn(field="video_name", title="Video"),
        TableColumn(field="n_result_frames", title="#ResFrames"),
    ],
    width=400,
    height=100
)

def cb_listqueries():
    global cds_queries

    console_clear()
    console_append("list queries requested")

    queries = control.list_queries_cloud()
    if len(queries) == 0:
        logger.info("no queries returned")
        console_append("no queries returned ")
        return

    # covert to dicts
    console_append(f"{len(queries)} queries found")

    vds = [asdict(v) for v in queries]

    # gen cds
    cds_queries = {}
    for k in vds[0].keys():
        cds_queries[k] = list(x[k] for x in vds)

    # add one column
    cds_queries['n_result_frames'] = list(len(x['results']) for x in vds)

    # erase columns that cannot be sent to browser
    cds_queries.pop('results', None)
    cds_queries.pop('progress_snapshots', None)

    logger.info(f"got {len(queries)} queries")

    source_queries.data = cds_queries # commit

b_lq = Button(label="ListQueries", button_type="success", width=BUTTON_WIDTH)
b_lq.on_click(cb_listqueries)

######
# preview img
######

# url = "server/static/preview/chaweng-1_10FPS/83999.thumbnail.jpg"
url_logo = "https://static.bokeh.org/logos/logo.png"  # for testing

sourceimg = ColumnDataSource(dict(
    url = [url_logo] * N_PREVIEWS,
    x1  = [i * PER_PREVIEW_IMG_SPACING for i in range(N_PREVIEWS)],
    #y1  = [PER_PREVIEW_IMG_HEIGHT] * N_PREVIEWS,
    y1  = [0] * N_PREVIEWS,
    w1  = [10] * N_PREVIEWS,
    h1  = [10] * N_PREVIEWS,
    #frame_ids = ['preview', '-5', '100000', 'id3', 'id4', 'id5']
    frame_ids = [''] * N_PREVIEWS
))

#xdr = Range1d(start=-100, end=200)
#ydr = Range1d(start=-100, end=200)

# format of the plot
ppreview = Plot(
#    title=None, x_range=xdr, y_range=ydr, plot_width=300, plot_height=300,
    x_range=Range1d(start=0, end=PER_PREVIEW_IMG_SPACING * N_PREVIEWS),
        y_range=Range1d(start=0, end=PLOT_PREVIEW_HEIGHT),
        title=Title(text="Preview"),
        plot_width=PER_PREVIEW_IMG_SPACING * N_PREVIEWS,
        plot_height=int(PLOT_PREVIEW_HEIGHT * 1.5),
        min_border=0, toolbar_location=None
    )


'''
# for layout debugging
yaxis = LinearAxis()
ppreview.add_layout(yaxis, 'left')
xaxis = LinearAxis()
ppreview.add_layout(xaxis, 'above')
'''

#image1 = ImageURL(url="url", x="x1", y="y1", w = 'w1', h='h1', anchor="top_left")
image1 = ImageURL(url="url", x="x1", y="y1", w = 'w1', h='h1', anchor="bottom_left")

# text1 = LabelSet(text="frame_ids", x="x1", y="y1", x_offset=0, y_offset=-50, level='glyph', source=sourceimg)
text1 = Text(text="frame_ids", x="x1", y="y1", x_offset=0, y_offset=-PLOT_PREVIEW_HEIGHT,
             text_color="magenta", text_font='helvetica', text_font_size='10px', text_baseline='bottom')

ppreview.add_glyph(sourceimg, image1) # feed the data
ppreview.add_glyph(sourceimg, text1) # feed the data
#ppreview.add_layout(text1) # feed the data -- nothing shown, why??

def load_preview_frames(v:VideoInfo, n_max:int):
    global the_videolib_preview
    the_videolib_preview.AddVideoStore(v.video_name).CleanStoredFrames()
    ids = control.download_video_preview_frames(v, n_max, res=[PER_PREVIEW_IMG_WIDTH, PER_PREVIEW_IMG_HEIGHT])
    console_write(f'{len(ids)} preview fetched')

    cds_previews = dict(
        url = [the_videolib_preview.GetVideoStore(v.video_name).GetFramePath(id) for id in ids],
        frame_ids = [f'fr:{id} [ {frameid_to_vtime(id)} ]' for id in ids],
        #url = [],
        x1 = [i * PER_PREVIEW_IMG_SPACING for i in range(n_max)],
        #y1 = [PER_PREVIEW_IMG_HEIGHT] * n_max,
        y1=[0] * n_max,
        w1 = [PER_PREVIEW_IMG_WIDTH] * n_max,
        h1 = [PER_PREVIEW_IMG_HEIGHT] * n_max
        #h1 = None,
        #h1 = [100] * n_max
    )

    print(cds_previews)
    sourceimg.data = cds_previews

    '''
    cds_txt = dict(
        text = [f'{id:07d}' for id in ids],
        x1 = [i * 200 for i in range(n_max)],
        y1 = [400] * n_max,
    )
    print(cds_txt)
    sourcetext.data = cds_txt
    '''

######
# show a single image selected
######

source_single_res_init = dict(
    url = [""],
    x1 = [0],
    y1 = [FULL_IMG_HEIGHT],
    w1 = [FULL_IMG_WIDTH],
    h1 = [FULL_IMG_HEIGHT],
)

source_single_res = ColumnDataSource(source_single_res_init)

psingle = Plot(
    x_range=Range1d(start=0, end=FULL_IMG_WIDTH), y_range=Range1d(start=0, end=FULL_IMG_HEIGHT),
    title=None, plot_width=FULL_IMG_WIDTH, plot_height=FULL_IMG_HEIGHT,
    min_border=0, toolbar_location=None)

#image_single = ImageURL(url="url", x="x1", y="y1", h="h1", w="w1", anchor="top_left")
image_single = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="top_left")
psingle.add_glyph(source_single_res, image_single)

def single_img_load(url:str):
    new_data = dict (
        url = [url],
        x1 = [0],
        y1 = [FULL_IMG_HEIGHT],
        w1 = [FULL_IMG_WIDTH],
        h1 = [FULL_IMG_HEIGHT],
    )
    source_single_res.data = new_data

######
# result imgs & table
######

source_results_init = ColumnDataSource(dict(
    url = [url_logo] * N_RESULTS_IMG,
    x1 = [i * PER_IMG_SPACING for i in range(N_RESULTS_IMG)],
    y1 = [PER_IMG_HEIGHT] * N_RESULTS_IMG,
    w1  = [10] * N_RESULTS_IMG,
    h1  = [10] * N_RESULTS_IMG,
    #frame_desc = ['desc1'] * N_RESULTS_IMG,
    frame_desc = [''] * N_RESULTS_IMG,
    frame_ids = [0] * N_RESULTS_IMG,
    scores = [0] * N_RESULTS_IMG,
    n_bboxes = [0] * N_RESULTS_IMG
))

source_results = source_results_init

presults = Plot(
    x_range=Range1d(start=0, end=PER_IMG_SPACING * N_RESULTS_IMG), y_range=Range1d(start=0, end=PLOT_IMG_HEIGHT),
    title=None, plot_width=PER_IMG_SPACING * N_RESULTS_IMG,
    plot_height=int(PLOT_IMG_HEIGHT*1.5),
    min_border=0, toolbar_location=None)

#image_results = ImageURL(url="url", x="x1", y="y1", anchor="top_left")
image_results = ImageURL(url="url", x="x1", y="y1", h="h1", w="w1", anchor="top_left")
txt_results = Text(text="frame_desc", x="x1", y="y1")
                   #x_offset=0, y_offset=PER_IMG_720P_HEIGHT+30)

# only show a small subset of result images
view_res = CDSView(source=source_results, filters=[IndexFilter([x for x in range(N_RESULTS_IMG)])])

presults.add_glyph(source_results, image_results, view=view_res)
presults.add_glyph(source_results, txt_results, view=view_res)

table_results = DataTable(
    source=source_results,
    columns = [
        TableColumn(field="frame_ids", title="Frame"),
        TableColumn(field="scores", title="Score"),
        TableColumn(field="n_bboxes", title="BBoxes"),
    ],
    width=400,
    height=200
)

# get updated query res from cloud.
# to be called in ticks of the main thread. rendering as table, spans, etc.
# sync, as it wont talk to the cam
def cb_update_res(res):
    global the_qid, the_videolib_results, the_current_span

    #logger.info(f">>>>>>>>>>>>> cb_update_res started")
    #t0 = time.time()

    #res = control.query_results(the_qid) # won't contact the cam

    #t1 = time.time()

    # filter res based on currently selected timespan
    # filtering using CDSviwe is not working: filters there can only accept simple logic (based on prescribed JS)
    # cannot handle custom data + logic like this
    if the_current_span != (-1,-1): # a span selected
        newres = [f for f in res if the_current_span[0] <= f.frame_id < the_current_span[1]]
        res = newres

    if res and len(res) > 0:
        # top frames
        n = min(len(res), N_RESULTS)
        ids = [f.frame_id for f in res[0:n]]
        scores = [f.high_confidence for f in res[0:n]]
        n_bboxes = [f.n_bboxes for f in res[0:n]]

        new_data = dict(
            url = [the_videolib_results.GetVideoStore(f'{the_qid}').GetFramePath(id) for id in ids],
            frame_desc = [f'{ids[x]} score:{scores[x]:.2f} boxes:{n_bboxes[x]}' for x in range(n)],
            frame_ids = ids,
            scores = scores,
            n_bboxes = n_bboxes,
            x1=[i * PER_IMG_SPACING for i in range(n)],
            y1=[PER_IMG_HEIGHT] * n,
            w1=[PER_IMG_WIDTH] * n,
            h1=[PLOT_IMG_HEIGHT] * n
        )

        #print('---------> update results:', new_data)
        source_results.data = new_data # commit

    '''
    t2 = time.time()
    logger.info(f"<<<<<< cb_update_res ended {t} {(t1-t0)*1000:.2f} {(t2-t1)*1000:.2f} ms")
    if t2-t0 > 0.1: # too high
        the_interval_ms *= 1.5
    else:
        the_interval_ms = max(the_interval_ms - 100, UPDATE_INTERVAL_MS)
    '''

def cb_results_table(attr, old, new):
    i = source_results.selected.indices[0]
    url = source_results.data['url'][i]
    console_write(f"selected frame {url} to display")
    logger.warning(f"selected frame {url} to display")
    single_img_load(url)
    # global view_res
    # view_res.filters = [IndexFilter([x for x in range(1)])] # must update filter like this

source_results.selected.on_change('indices', cb_results_table)

##############################
# class selector
##############################
class_sel = RadioButtonGroup(labels=the_all_classes, active=0, width=int(BUTTON_WIDTH/2))

def cb_class(attr, old, new):
    global the_class, the_all_classes
    the_class = the_all_classes[class_sel.active]
    #print(the_class)
    console_write(f'chose class = {the_class}')
    update_b_query_label()

#class_sel.on_change('active', lambda  attr, old, new: cb_class())
class_sel.on_change('active', cb_class)

##############################
# frameskip selector
# https://www.programcreek.com/python/example/106842/bokeh.models.Select
##############################
frameskip_sel = Select(value=f"{the_frameskip}", options=[f"{k}" for k in the_frameskip_choices], width=40
                       #title = "frameskip"
                       )

def cb_frameskip(attr, old, new):
    global the_frameskip
    #print(frameskip_sel.value)
    the_frameskip = int(frameskip_sel.value)
    console_write(f'set frameskip = {the_frameskip}')
    update_b_query_label()

frameskip_sel.on_change('value', cb_frameskip)

##############################
# vbar progress tracker
# colors: https://docs.bokeh.org/en/latest/docs/reference/colors.html
##############################
NSPANS = 10
timespans = [f'win{d}' for d in range(NSPANS)]
#timespans = [f'{d:4d}min' for d in range(NSPANS)]
#states = ['.', '-', '1', '2', '3', '4', '5', 's', 'r']
states = ['held', 'queued', 'pass1', 'pass2', 'pass3', 'pass4', 'pass5', 'uploaded', 'positive']
colors = ["lightsteelblue", "white", "gainsboro", "lightgray", "silver", "darkgray", "gray", "cyan", "lime"]
data_span_init = {
    'timespans' : timespans,
    'held': [0] * NSPANS,  # backqueue
    'queued' : [0] * NSPANS,    # .
    'pass1' : [0] * NSPANS,    # ditto
    'pass2' : [0] * NSPANS,    # ditto
    'pass3' : [0] * NSPANS,    # ditto
    'pass4' : [0] * NSPANS,    # ditto
    'pass5' : [0] * NSPANS,    # ditto
    'uploaded' : [0] * NSPANS,    # s
    'positive' : [0] * NSPANS,    # r
    'max_confidence' : [0.0] * NSPANS,  # max of confidence in all 'r' frames
    'avg_confidence' : [0.0] * NSPANS,  # ditto
}

data_span = data_span_init

source_span = ColumnDataSource(data=data_span)

pspan = figure(x_range=timespans, plot_height=200, plot_width=int(PLOT_IMG_WIDTH/2.5),
               # title="Fruit Counts by Year", toolbar_location=None,
               title="Progress per win",
               tools="hover,box_select,tap",
               #tooltips="$name @timespans: @$name")
                tooltips="$name frames: @$name")
pspan.vbar_stack(states, x='timespans', width=0.9, color=colors, source=source_span,
             #legend_label=states
                 )

''' # this adds labeles to the top of bars. not what we need
label_span = LabelSet(text="max_confidence", x="timespans", y="r",
                      x_offset=0, y_offset=-50, source=source_span)
pspan.add_layout(label_span)
'''

pspan.y_range.start = 0
pspan.y_range.end = 10
#pspan.x_range.range_padding = 0.1
pspan.xgrid.grid_line_color = None
pspan.axis.minor_tick_line_color = None
pspan.outline_line_color = None
pspan.legend.location = "top_left"
pspan.legend.orientation = "horizontal"

the_spans = None
the_current_span = (-1, -1) # min, max

###########
# callbacks, etc

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def cb_update_span(framestates:typing.Dict[int, str]):
    global the_spans
    fs_list = [(frame_id, state) for frame_id, state in framestates.items()]
    fs_list.sort(key= lambda x: x[0])
    #print(fs_list[0:10])

    '''
    I do not fully understand how npo.array_split() works, but i) the 0th span seems incorrect (e.g. 20k frs, minid 0, maxid 9999)
    and ii) the split arrays are in sophisticated types. i) causes a bug. bad.           
    '''
    #the_spans = np.array_split(fs_list, NSPANS)    # slice into N spans
    the_spans = list(chunks(fs_list, int(len(fs_list)/NSPANS)+1))
    assert(len(the_spans) == NSPANS)

    new_data = copy.deepcopy(data_span)

    '''
    ss = [id for id, st in the_spans[0]]
    print(ss[0:10], the_spans[0][0:10], len(the_spans[0]), max(the_spans[0], key=lambda x: x[0]))    
    for x in range(1,9999):
        c = ss.count(x)
        if c>0:
            print(x, c)
    '''

    #for x in the_spans[0][0:10]:
    #    print(x[0], sep=' ')

    # new_data['timespans'] = [f'winXXX{d}' for d in range(NSPANS)]

    # sp = [ (frame_id, state) ... ]
    # f is a (frame_id, state) tuple

    for idx, sp in enumerate(the_spans):
        new_data['held'][idx] = sum(1 for f in sp if f[1] == '-')
        new_data['queued'][idx] = sum(1 for f in sp if f[1] == '.')
        new_data['pass1'][idx] = sum(1 for f in sp if f[1] == '1')
        new_data['pass2'][idx] = sum(1 for f in sp if f[1] == '2')
        new_data['pass3'][idx] = sum(1 for f in sp if f[1] == '3')
        new_data['pass4'][idx] = sum(1 for f in sp if f[1] == '4')
        new_data['pass5'][idx] = sum(1 for f in sp if f[1] == '5')
        new_data['uploaded'][idx] = sum(1 for f in sp if f[1] == 's')
        #new_data['r'][idx] = sum(1 for f in sp if f[1] == 'r')
        # a bit hack. for 'r' frames, we store the confidence as its framestate
        new_data['positive'][idx] = sum(1 for f in sp if f[1][0] == '0')

        max_conf_frame, new_data['max_confidence'][idx] = max(sp, key=lambda x: float(x[1]) if x[1][0] == '0' else 0)
        # todo: later...
        #new_data['avg_confidence'][idx] = mean(sp, key=lambda x: float(x[1]) if x[1][0] == '0' or x[1][0] == '1' else 0)

    #print(new_data)
    #pspan.x_range.factors = [f'winXXX{d}' for d in range(NSPANS)] # works

    #factors = [f'winXXX{d}' for d in range(NSPANS)]  # working

    # factors = [f'winXXX{len(sp)}' for sp in the_spans] # not working?
    #factors = [f'winXXX{idx}' for idx, sp in enumerate(the_spans)]  # working
    # factors = [f'winXXX{len(sp)/10000}' for idx, sp in enumerate(the_spans)]  # not working

    #factors = [f'{d+1}min' for d in range(NSPANS)]  #  working
    #factors = [f'3min' for d in range(NSPANS)]  # working
    #factors = [f"{int(len(sp) * the_frameskip/10/60)}min" for sp in the_spans] # not working?


    # update pspan -- XXX inefficeint. should do it only once when query starts
    '''
    factors = []
    offset = 0
    if the_video_info and the_video_info.fps > 0:
        fps = the_video_info.fps
        for sp in the_spans:
            #factors.append(f"{int(len(sp) * the_frameskip/fps/60)}min")
            factors.append(f"{offset}-{offset+int(len(sp) * the_frameskip / fps / 60)}min")
            offset+=int(len(sp) * the_frameskip / fps / 60)
    else: # we have no fps info
        for sp in the_spans:
            factors.append(f"{int(len(sp)/1000)}k")


    pspan.x_range.factors = factors
    '''

    title = f"Each win: {int(len(sp))} frames in "
    if the_video_info and the_video_info.fps > 0:
        fps = the_video_info.fps
        title += f"{int(len(sp) * the_frameskip / fps / 60)} mins"

    pspan.title.text= title

    #print(len(sp), the_frameskip, fps)
    #print(factors)

    source_span.data = new_data

def cb_span(attr, old, new):
    global the_spans, the_current_span
    nsel = len(source_span.selected.indices)
    if nsel == 0 or not the_spans:
        the_current_span = (-1, -1)
        return
    i = source_span.selected.indices[0]
    the_current_span = (int(min(the_spans[i], key=lambda x: x[0])[0]),
                        int(max(the_spans[i], key=lambda x: x[0])[0]))

    logger.warning(f"selected span {i}. fr {the_current_span} frs {len(the_spans[i])}")

source_span.selected.on_change('indices', cb_span)

####
# "promote" "demote" buttons
####

def cb_promo(is_promote:bool):
    global the_spans
    nsel = len(source_span.selected.indices)
    #print(f'{nsel} total selected')

    if is_promote:
        action = "promote"
    else:
        action = "demote"

    i = source_span.selected.indices[0]
    minid = min(the_spans[i], key=lambda x: x[0])[0]
    maxid = max(the_spans[i], key=lambda x: x[0])[0]

    console_write(f"{action} frames {minid} -- {maxid}")
    logger.warning(f"{action} span {i} frs {len(the_spans[i])} {minid} -- {maxid}")
    msg = control.promote_frames(-1, int(minid), int(maxid), is_promote=is_promote)
    console_append("cam replied:" + msg)

b_promo = Button(label="Promo", button_type="success", width=BUTTON_WIDTH)
b_promo.on_click(partial(cb_promo, is_promote=True))

b_demo = Button(label="Demote", button_type="danger", width=BUTTON_WIDTH)
b_demo.on_click(partial(cb_promo, is_promote=False))

###########
# bar graph with only "results"

pspan0 = figure(x_range=timespans, y_range=(0,10), plot_height=200, plot_width=int(PLOT_IMG_WIDTH/2.5),
               # title="Fruit Counts by Year", toolbar_location=None,
                title="Positives per win",
               tools="hover,box_select,tap", tooltips="pos: @positive; max_confidence: @max_confidence")

mapper = linear_cmap(field_name='max_confidence', palette=RdYlGn10, low=1, high=0) # reverse a palaette
pspan0.vbar(x='timespans', top='positive', width=0.9, color=mapper, source=source_span)

##############################

def _create_prices(t):
    last_average = 100 if t==0 else source.data['average'][-1]
    returns = np.asarray(np.random.lognormal(mean.value, stddev.value, 1))
    average =  last_average * np.cumprod(returns)
    high = average * np.exp(abs(np.random.gamma(1, 0.03, size=1)))
    low = average / np.exp(abs(np.random.gamma(1, 0.03, size=1)))
    delta = high - low
    open = low + delta * np.random.uniform(0.05, 0.95, size=1)
    close = low + delta * np.random.uniform(0.05, 0.95, size=1)
    return open[0], high[0], low[0], close[0], average[0]

def _moving_avg(prices, days=10):
    if len(prices) < days: return [100]
    return np.convolve(prices[-days:], np.ones(days, dtype=float), mode="valid") / days

def _ema(prices, days=10):
    if len(prices) < days or days < 2: return [prices[-1]]
    a = 2.0 / (days+1)
    kernel = np.ones(days, dtype=float)
    kernel[1:] = 1 - a
    kernel = a * np.cumprod(kernel)
    # The 0.8647 normalizes out that we stop the EMA after a finite number of terms
    return np.convolve(prices[-days:], kernel, mode="valid") / (0.8647)

# xzl
data1 = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        downloads=[randint(0, 100) for i in range(10)],
        test=[randint(0, 100) for i in range(10)]
    )
source1 = ColumnDataSource(data1)

columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
        TableColumn(field="test", title="xxx"),
    ]
data_table = DataTable(source=source1, columns=columns, width=400, height=280)

@count()
def update(t):
    #print(t)
    if paused:
        return
        
    open, high, low, close, average = _create_prices(t)
    color = "green" if open < close else "red"

    new_data = dict(
        time=[t],
        open=[open],
        high=[high],
        low=[low],
        close=[close],
        average=[average],
        color=[color],
    )

    close = source.data['close'] + [close]
    ma12 = _moving_avg(close[-12:], 12)[0]
    ma26 = _moving_avg(close[-26:], 26)[0]
    ema12 = _ema(close[-12:], 12)[0]
    ema26 = _ema(close[-26:], 26)[0]

    if   mavg.value == MA12:  new_data['ma'] = [ma12]
    elif mavg.value == MA26:  new_data['ma'] = [ma26]
    elif mavg.value == EMA12: new_data['ma'] = [ema12]
    elif mavg.value == EMA26: new_data['ma'] = [ema26]

    macd = ema12 - ema26
    new_data['macd'] = [macd]

    macd_series = source.data['macd'] + [macd]
    macd9 = _ema(macd_series[-26:], 9)[0]
    new_data['macd9'] = [macd9]
    new_data['macdh'] = [macd - macd9]

    source.stream(new_data, 300)

######################## UI layout #####################
#curdoc().add_root(column(row(mean, stddev, mavg), gridplot([[p], [p2]], toolbar_location="left", plot_width=1000)))
#curdoc().add_root(column(row(mean, stddev, mavg, b), 
    #gridplot([[p]], toolbar_location="left", plot_width=1000))
    #)

'''
curdoc().add_root(column(row(mean, stddev, mavg, b, b_lv),
    gridplot([[p]], toolbar_location="left", plot_width=1000), data_table)
    )
'''

'''
doc.add_root(column(row(b, b_lv, b_query),
    gridplot([[pcam, ppreview]], toolbar_location="left", plot_width=1000), table_listvideos)
    )
'''

'''
doc.add_root(column(
    row(para_log, frameskip_sel, class_sel),
    row(table_listvideos, pcam, pcam1, pcam2, table_listqueries),
    row(b_pause, b_resume, b_lv, b_lq, b_query),
    #row(pcam, table_results),
    row(column(pspan, pspan0, row(b_promo, b_demo)), table_results),
    #row(pspan, b_promo, b_demo),
    #pspan0,
    presults,
    psingle,
    ppreview
))
'''

doc.add_root(column(
    row(table_listvideos, ppreview),
    row(frameskip_sel, class_sel),
    row(b_pause, b_resume, b_lv, b_lq, b_query),
    #row(pcam, table_results),
    row(column(pspan, pspan0, row(b_promo, b_demo)), column(table_results, row(pcam, pcam2), row(pcam1, pcam3, pcam4), para_log), psingle, table_listqueries),
    #row(pspan, b_promo, b_demo),
    #pspan0,
    presults
))


# curdoc().add_periodic_callback(update, 500)
doc.title = "📷⚙️"

logger.info("main execed!")

# cb_listvideos()   # can do it here.
show_cam_specs()