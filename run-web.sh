#bokeh serve --show ohlc --allow-websocket-origin=10.10.10.3:5006

# create if non existing 
mkdir -p preview result

PYTHONPATH=$PWD \
bokeh serve server --allow-websocket-origin=10.10.10.3:5006 --log-level info
