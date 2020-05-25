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

import numpy as np

from bokeh.driving import count
from bokeh.layouts import column, gridplot, row
from bokeh.models import Select, Slider, Button, ImageURL, Plot, LinearAxis, Paragraph, Div
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

b_pause = Button(label="Pause", button_type="success", width=50, disabled=True)
b_resume = Button(label="Resume", button_type="success", width=50, disabled=True)

the_started = False
the_paused = False

def cb_resume():
    global the_paused
    if the_paused:
        console_append("resume query requested")
        msg = control.query_resume()
        console_append("cam replied:" + msg)
        if msg.startswith("OK"):
            the_paused = False
            b_resume.disabled = True
            b_pause.disabled = False
    else:
        logger.error("bug?")

def cb_pause():
    global the_paused
    if not the_paused:
        console_append("pause query requested")
        msg = control.query_pause()
        console_append("pause query. cam replied:" + msg)
        if msg.startswith("OK"):
            the_paused = True
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

pcam = figure(plot_height=500, tools="xpan,xwheel_zoom,xbox_zoom,reset", x_axis_type=None, y_axis_location="right")
pcam.x_range.follow = "end"
pcam.x_range.follow_interval = 100
pcam.x_range.range_padding = 0

pcam.line(x='ts', y='n_frames_processed_cam', alpha=0.8, line_width=2, color='red', source=src_cam)
pcam.line(x='ts', y='n_frames_recv_yolo', alpha=0.8, line_width=2, color='blue', source=src_cam)

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
##############################

@gen.coroutine
def cb_update_camsrc(prog):
    logger.info("update doc...")

    new_data = dict(
        ts=[prog.ts],
        n_frames_processed_cam=[prog.n_frames_processed_cam],
        n_frames_recv_yolo=[prog.n_frames_recv_yolo]
    )
    print(new_data)
    src_cam.stream(new_data, rollover=300)

ev_progress = threading.Event()

def thread_progress():
    logger.info("started")
    while True:
        ev_progress.wait()
        ev_progress.clear()
        prog = control.create_query_progress_snapshot(qid=0) # test
        if prog:
            doc.add_next_tick_callback(partial(cb_update_camsrc, prog))
            # logger.info("poll progress.. ok")
        else:
            logger.info("poll progress... failed")

thread_progress = threading.Thread(target=thread_progress)
thread_progress.start()

@count()
def update_camsrc_request(t):
    #logger.info("check....")
    ev_progress.set()

the_poll_cb = None

######
# text for log
######
#para_log = Paragraph(text="hello world", width=200, height=100)
para_log = Div(text="[log messages]", width=200, height=100)

def console_append(msg:str):
    para_log.text += '<ul>{}</ul>'.format(msg)

def console_clear():
    para_log.text = ''

######
# start a sample query
######

def cb_newquery():
    global the_poll_cb
    resp=control.query_submit(video_name='chaweng-1_10FPS',
                 op_name='random', crop='-1,-1,-1,-1', target_class='motorbike')
    # print('qid:', resp)
    console_append(f"new query started. qid {resp}")
    # the_poll_cb = doc.add_periodic_callback(update_camsrc_request, 500)

    the_started = True
    b_query.label="Abort"

    b_pause.disabled = False

b_query = Button(label="Query", button_type="danger")
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

def cb_listvideos():
    global cds_videos

    console_clear()
    console_append('requested to list videos')

    videos = control.list_videos()
    if len(videos) == 0:
        logger.error("nothing!")
        return

    console_append(f'{len(videos)} found')

    # convert to dicts
    vds = [asdict(v) for v in videos]

    # gen cds
    cds_videos = {}
    for k in vds[0].keys():
        cds_videos[k] = list(x[k] for x in vds)

    print(cds_videos)

    source_videos.data = cds_videos  # commit

b_lv = Button(label="ListVideos", button_type="success")
b_lv.on_click(cb_listvideos)

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

b_lq = Button(label="ListQueries", button_type="success")
b_lq.on_click(cb_listqueries)

######
# plot shoiwing img
######

url = "server/static/preview/chaweng-1_10FPS/83999.thumbnail.jpg"
# url = "https://static.bokeh.org/logos/logo.png"

sourceimg = ColumnDataSource(dict(
    url = [url, url, url],
    x1  = [10, 20, 30],
    y1  = [10, 20, 30],
    w1  = [10, 10, 10],
    h1  = [10, 10, 10],
))

#xdr = Range1d(start=-100, end=200)
#ydr = Range1d(start=-100, end=200)

# format of the plot
pimg = Plot(
#    title=None, x_range=xdr, y_range=ydr, plot_width=300, plot_height=300,
    title=None, plot_width=300, plot_height=300,
    min_border=0, toolbar_location=None)

xaxis = LinearAxis()
pimg.add_layout(xaxis, 'below')

yaxis = LinearAxis()
pimg.add_layout(yaxis,'left')

image1 = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
pimg.add_glyph(sourceimg, image1) # feed the data
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
    para_log,
    row(table_listvideos, table_listqueries),
    row(b_pause, b_resume, b_lv, b_lq, b_query),
    pcam,
    pimg,
))

# curdoc().add_periodic_callback(update, 500)
doc.title = "OHLC"

logger.info("main execed!")
