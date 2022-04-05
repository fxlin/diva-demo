#bokeh serve --show ohlc --allow-websocket-origin=10.10.10.3:5006

# create if non existing 
mkdir -p preview result


# cf: https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html
# can also set up BOKEH_ALLOW_WS_ORIGIN env var

PYTHONPATH=$PWD \
bokeh serve server \
--allow-websocket-origin=10.10.10.3:5006 \
--allow-websocket-origin=gpusrv14.cs.virginia.edu:5006 \
--log-level info
