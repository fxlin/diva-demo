'''
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

import numpy as np

from bokeh.driving import count
from bokeh.layouts import column, gridplot, row
from bokeh.models import Select, Slider, Button, ImageURL, Plot, LinearAxis, Paragraph, Div, Range1d, Text
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
PER_IMG_WIDTH = 200 # for image spacing
PER_IMG_HEIGHT = 100 # only for determining label offset
PER_IMG_720P_HEIGHT = int(720/(1280/PER_IMG_WIDTH)) # 720p image scaled height

N_PREVIEWS = 5
N_RESULTS = 20
N_RESULTS_IMG = 5

BUTTON_WIDTH = 100

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
the_all_classes = ['car','motorbike','person','bicycle','bus','truck']
the_class = the_all_classes[0]

def cb_resume():
    global the_paused, the_poll_cb
    if the_paused:
        console_write("resume query requested")
        msg = control.query_resume()
        console_append("cam replied:" + msg)
        if msg.startswith("OK"):
            the_paused = False
            the_poll_cb = doc.add_periodic_callback(update_camsrc_request, 500)
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
# plot for n_frames_processed_cam
######

src_cam = ColumnDataSource(dict(
    ts=[], n_frames_processed_cam=[], n_frames_recv_yolo=[]
))

pcam = figure(plot_height=300, plot_width=500, tools="xpan,xwheel_zoom,xbox_zoom,reset", x_axis_type=None, y_axis_location="right")
pcam.x_range.follow = "end"
pcam.x_range.follow_interval = 100
pcam.x_range.range_padding = 0

pcam.line(x='ts', y='n_frames_processed_cam', alpha=0.8, line_width=2, color='red', source=src_cam)
pcam.line(x='ts', y='n_frames_recv_yolo', alpha=0.8, line_width=2, color='blue', source=src_cam)

xaxis = LinearAxis()
pcam.add_layout(xaxis, 'below')

yaxis = LinearAxis()
pcam.add_layout(yaxis,'left')

@count()
def update_camsrc(t):
    logger.info(f"update_camsrc checkgin....{t}")
    prog = control.get_latest_query_progress(qid=0)
    if not prog: return

    new_data = dict(
        ts = [prog.ts],
        n_frames_processed_cam = [prog.n_frames_processed_cam],
        n_frames_recv_yolo = [prog.n_frames_recv_yolo]
    )

    print(new_data)
    src_cam.stream(new_data, rollover=300)

# curdoc().add_periodic_callback(update_camsrc, 500)

##############################
# test: using a separate thread to poll progress on demand
# todo: call all progress update from here
##############################

@gen.coroutine
def cb_update_camsrc(prog):
    #logger.info("update doc...")

    new_data = dict(
        ts=[prog.ts],
        n_frames_processed_cam=[prog.n_frames_processed_cam],
        n_frames_recv_yolo=[prog.n_frames_recv_yolo]
    )
    # print(new_data)
    src_cam.stream(new_data, rollover=300)
    cb_update_span(prog.framestates)

ev_progress = threading.Event()

def thread_progress():
    logger.info("started")
    while True:
        ev_progress.wait()
        ev_progress.clear()
        prog = control.create_query_progress_snapshot(qid=the_qid) # test
        if prog:
            doc.add_next_tick_callback(partial(cb_update_camsrc, prog))
            # logger.info("poll progress.. ok")
        else:
            logger.info("poll progress... failed")

thread_progress = threading.Thread(target=thread_progress)
thread_progress.start()

# will be called periodically once query in progress
@count()
def update_camsrc_request(t):
    #logger.info("check....")
    ev_progress.set()
    cb_update_results(t)

the_poll_cb = None

######
# text console
######
#para_log = Paragraph(text="hello world", width=200, height=100)
para_log = Div(text="[log messages]", width=200, height=50)

def console_append(msg:str):
    para_log.text += '<ul>{}</ul>'.format(msg)

def console_clear():
    para_log.text = ''

def console_write(msg:str):
    console_clear()
    console_append(msg)

######
# start a sample query
######

def cb_newquery():
    global the_poll_cb, the_started, the_qid

    the_qid = control.query_submit(video_name=the_video_name,
                 op_names=['random', 'random', 'random'], crop='-1,-1,-1,-1', target_class=the_class)
    # print('qid:', resp)
    console_append(f"new query started. qid {the_qid}")
    the_poll_cb = doc.add_periodic_callback(update_camsrc_request, 500)

    the_started = True
    b_query.label="Abort"

    b_pause.disabled = False

b_query = Button(label="Query", button_type="danger", width=BUTTON_WIDTH*2)
b_query.on_click(cb_newquery)

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

the_videos:typing.List[VideoInfo] = None

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
    # print(f"selected! {attraname} {old} {new}")
    i = source_videos.selected.indices[0]
    the_video_name = the_videos[i].video_name
    console_write(f'selected video {i} {the_video_name}')
    if len(the_videos) >= i + 1:
        load_preview_frames(the_videos[i], 3)
        b_query.label=f'Query:{the_video_name}'

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
    url = [url_logo, url_logo, url_logo],
    #x1  = [10, 20, 30],
    x1  = [0, 200, 400],
    #y1  = [10, 20, 30],
    y1  = [100, 200, 500],
    w1  = [10, 10, 10],
    h1  = [10, 10, 10],
    frame_ids = ['preview', '-5', '100000']
))

#xdr = Range1d(start=-100, end=200)
#ydr = Range1d(start=-100, end=200)

# format of the plot
pimg = Plot(
#    title=None, x_range=xdr, y_range=ydr, plot_width=300, plot_height=300,
    x_range=Range1d(start=0, end=PLOT_IMG_WIDTH), y_range=Range1d(start=0, end=PLOT_IMG_HEIGHT),
    title=None, plot_width=PLOT_IMG_WIDTH, plot_height=PLOT_IMG_HEIGHT,
    min_border=0, toolbar_location=None)

#image1 = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
image1 = ImageURL(url="url", x="x1", y="y1", anchor="top_left")

'''
sourcetext = ColumnDataSource(dict(
    text = [],
    x1 = [],
    y1 = []
))
'''
# text1 = LabelSet(text="frame_ids", x="x1", y="y1", x_offset=0, y_offset=-50, level='glyph', source=sourceimg)
text1 = Text(text="frame_ids", x="x1", y="y1", x_offset=0, y_offset=PER_IMG_HEIGHT, text_color="#96deb3")

pimg.add_glyph(sourceimg, image1) # feed the data
pimg.add_glyph(sourceimg, text1) # feed the data
#pimg.add_layout(text1) # feed the data -- nothing shown, why??

def load_preview_frames(v:VideoInfo, n_max:int):
    global the_videolib_preview
    the_videolib_preview.AddVideoStore(v.video_name).CleanStoredFrames()
    ids = control.download_video_preview_frames(v, n_max)
    console_write(f'{len(ids)} preview fetched')

    cds_previews = dict(
        url = [the_videolib_preview.GetVideoStore(v.video_name).GetFramePath(id) for id in ids],
        frame_ids = [f'{id}' for id in ids],
        #url = [],
        x1 = [i * PER_IMG_WIDTH for i in range(n_max)],
        y1 = [PLOT_IMG_HEIGHT] * n_max,
        #w1 = None,
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

source_single_res = ColumnDataSource(dict(
    url = [url_logo],
    x1 = [0],
    y1 = [720],
 #   h1 = [100, 100],
  #  w1 = [100, 100],
))

psingle = Plot(
    x_range=Range1d(start=0, end=1280), y_range=Range1d(start=0, end=720),
    title=None, plot_width=1280, plot_height=720,
    min_border=0, toolbar_location=None)

#image_single = ImageURL(url="url", x="x1", y="y1", h="h1", w="w1", anchor="top_left")
image_single = ImageURL(url="url", x="x1", y="y1", anchor="top_left")
psingle.add_glyph(source_single_res, image_single)

def single_img_load(url:str):
    new_data = dict (
        url = [url],
        x1 = [0],
        y1 = [720]
    )
    source_single_res.data = new_data

######
# result imgs & table
######

source_results = ColumnDataSource(dict(
    url = [url_logo, url_logo],
    x1 = [0, 200],
    y1 = [100, 200],
    h1 = [100, 100],
    w1 = [100, 100],
    frame_desc = ['desc1', 'desc2'],
    frame_ids = [1, 2],
    scores = [0, 0],
    n_bboxes = [0, 0]
))

presults = Plot(
    x_range=Range1d(start=0, end=PLOT_IMG_WIDTH), y_range=Range1d(start=0, end=PLOT_IMG_HEIGHT),
    title=None, plot_width=PLOT_IMG_WIDTH, plot_height=PLOT_IMG_HEIGHT,
    min_border=0, toolbar_location=None)

#image_results = ImageURL(url="url", x="x1", y="y1", anchor="top_left")
image_results = ImageURL(url="url", x="x1", y="y1", h="h1", w="w1", anchor="top_left")
txt_results = Text(text="frame_desc", x="x1", y="y1", x_offset=0, y_offset=PER_IMG_720P_HEIGHT+30)

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
    height=400
)

# cb for main thread
# @count
def cb_update_results(t):
    global the_qid, the_videolib_results, the_current_span
    res = control.query_results(the_qid)

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
            frame_desc = [f'{ids[x]}/{scores[x]:.2f}/{n_bboxes[x]}' for x in range(n)],
            frame_ids = ids,
            scores = scores,
            n_bboxes = n_bboxes,
            x1=[i * PER_IMG_WIDTH for i in range(n)],
            y1=[PLOT_IMG_HEIGHT] * n,
            w1=[PER_IMG_WIDTH] * n,
            h1=[PER_IMG_720P_HEIGHT] * n
        )

        #print('---------> update results:', new_data)
        source_results.data = new_data # commit

def cb_results_table(attr, old, new):
    i = source_results.selected.indices[0]
    url = source_results.data['url'][i]
    console_write(f"selected frame {url} to display")
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

#class_sel.on_change('active', lambda  attr, old, new: cb_class())
class_sel.on_change('active', cb_class)


##############################
# vbar progress tracker
##############################
NSPANS = 10
timespans = [f'win{d}' for d in range(NSPANS)]
states = ['.', '-', '1', '2', '3', 's', 'r']
#states = ".-123sr"
colors = ["darkgrey", "black", "lightpink", "hotpink", "deeppink", "cyan", "lime"]
data_span = {
    'timespans' : timespans,
    '.' : [10] * NSPANS,
    '-' : [10] * NSPANS,
    '1' : [20] * NSPANS,    # ditto
    '2' : [20] * NSPANS,    # ditto
    '3' : [20] * NSPANS,    # ditto
    's' : [30] * NSPANS,
    'r' : [30] * NSPANS,
    'max_confidence' : [0.0] * NSPANS,  # max of confidence in all 'r' frames
    'avg_confidence' : [0.0] * NSPANS,  # ditto
}
source_span = ColumnDataSource(data=data_span)

pspan = figure(x_range=timespans, plot_height=250, plot_width=PLOT_IMG_WIDTH>>1,
               # title="Fruit Counts by Year", toolbar_location=None,
               tools="hover,box_select,tap", tooltips="$name @timespans: @$name")
pspan.vbar_stack(states, x='timespans', width=0.9, color=colors, source=source_span,
             #legend_label=states
                 )

pspan.y_range.start = 0
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
def cb_update_span(framestates:typing.Dict[int, str]):
    global the_spans
    fs_list = [(frame_id, state) for frame_id, state in framestates.items()]
    #sorted(fs_list, key=lambda x: x[0])
    fs_list.sort(key= lambda x: x[0])
    the_spans = np.array_split(fs_list, NSPANS)    # slice into N spans
    new_data = copy.deepcopy(data_span)

    # new_data['timespans'] = [f'winXXX{d}' for d in range(NSPANS)]

    # f is a (frame_id, state) tuple
    for idx, sp in enumerate(the_spans):
        new_data['.'][idx] = sum(1 for f in sp if f[1] == '.')
        new_data['-'][idx] = sum(1 for f in sp if f[1] == '-')
        new_data['1'][idx] = sum(1 for f in sp if f[1] == '1')
        new_data['2'][idx] = sum(1 for f in sp if f[1] == '2')
        new_data['3'][idx] = sum(1 for f in sp if f[1] == '3')
        new_data['s'][idx] = sum(1 for f in sp if f[1] == 's')
        #new_data['r'][idx] = sum(1 for f in sp if f[1] == 'r')
        # a bit hack. for 'r' frames, we store the confidence as its framestate
        new_data['r'][idx] = sum(1 for f in sp if f[1][0] == '0')

        max_conf_frame, new_data['max_confidence'][idx] = max(sp, key=lambda x: float(x[1]) if x[1][0] == '0' else 0)
        # todo: later...
        #new_data['avg_confidence'][idx] = mean(sp, key=lambda x: float(x[1]) if x[1][0] == '0' or x[1][0] == '1' else 0)

    #print(new_data)

    source_span.data = new_data

def cb_span(attr, old, new):
    global the_spans, the_current_span
    nsel = len(source_span.selected.indices)
    if nsel == 0:
        the_current_span = (-1, -1)
        return
    i = source_span.selected.indices[0]
    the_current_span = (int(min(the_spans[i], key=lambda x: x[0])[0]),
                        int(max(the_spans[i], key=lambda x: x[0])[0]))

    print(f"selected! {attr} {old} {new} {i} the_current_span {the_current_span}")

source_span.selected.on_change('indices', cb_span)

####
# "promote" "demote" buttons
####

def cb_promo(is_promote:bool):
    global the_spans
    nsel = len(source_span.selected.indices)
    print(f'{nsel} total selected')

    if is_promote:
        action = "promote"
    else:
        action = "demote"

    i = source_span.selected.indices[0]
    minid = min(the_spans[i], key=lambda x: x[0])[0]
    maxid = max(the_spans[i], key=lambda x: x[0])[0]

    console_write(f"{action} frames {minid} -- {maxid}")
    msg = control.promote_frames(-1, int(minid), int(maxid), is_promote=is_promote)
    console_append("cam replied:" + msg)

b_promo = Button(label="Promo", button_type="success", width=BUTTON_WIDTH)
b_promo.on_click(partial(cb_promo, is_promote=True))

b_demo = Button(label="Demote", button_type="success", width=BUTTON_WIDTH)
b_demo.on_click(partial(cb_promo, is_promote=False))

###########
# bar graph with only "results"
pspan0 = figure(x_range=timespans, plot_height=250, plot_width=PLOT_IMG_WIDTH>>1,
               # title="Fruit Counts by Year", toolbar_location=None,
               tools="hover,box_select,tap", tooltips="pos: @r; max_confidence: @max_confidence")

mapper = linear_cmap(field_name='max_confidence', palette=RdYlGn10, low=1, high=0) # reverse a palaette
pspan0.vbar(x='timespans', top='r', width=0.9, color=mapper, source=source_span)

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
    gridplot([[pcam, pimg]], toolbar_location="left", plot_width=1000), table_listvideos)
    )
'''

doc.add_root(column(
    row(para_log, class_sel),
    row(table_listvideos, table_listqueries),
    row(b_pause, b_resume, b_lv, b_lq, b_query),
    row(pcam, table_results),
    row(pspan, b_promo, b_demo),
    pspan0,
    presults,
    psingle,
    pimg
))

# curdoc().add_periodic_callback(update, 500)
doc.title = "📷⚙️"

logger.info("main execed!")
