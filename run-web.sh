#bokeh serve --show ohlc --allow-websocket-origin=10.10.10.3:5006
PYTHONPATH=$PWD \
bokeh serve server --allow-websocket-origin=10.10.10.3:5006 --log-level info
