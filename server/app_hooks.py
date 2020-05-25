import logging, sys
import control as cloud

FORMAT = '%(asctime)-15s %(levelname)8s %(thread)d %(threadName)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
#logger = logging.getLogger(__name__)
logger = logging.getLogger()

server = None
tps = None

def on_server_loaded(server_context):
    global server, tps
    ''' If present, this function is called when the server first starts. '''
    logger.error("--------- flask server running --------------------")
    server, tps = cloud.grpc_serve()
    pass

def on_server_unloaded(server_context):
    ''' If present, this function is called when the server shuts down. '''
    pass

def on_session_created(session_context):
    ''' If present, this function is called when a session is created. '''
    pass

def on_session_destroyed(session_context):
    ''' If present, this function is called when a session is closed. '''
    pass