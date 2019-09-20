# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import det_yolov3_pb2 as det__yolov3__pb2


class DetYOLOv3Stub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.DetFrame = channel.unary_unary(
        '/DetYOLOv3/DetFrame',
        request_serializer=det__yolov3__pb2.DetFrameRequest.SerializeToString,
        response_deserializer=det__yolov3__pb2.Score.FromString,
        )


class DetYOLOv3Servicer(object):
  # missing associated documentation comment in .proto file
  pass

  def DetFrame(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DetYOLOv3Servicer_to_server(servicer, server):
  rpc_method_handlers = {
      'DetFrame': grpc.unary_unary_rpc_method_handler(
          servicer.DetFrame,
          request_deserializer=det__yolov3__pb2.DetFrameRequest.FromString,
          response_serializer=det__yolov3__pb2.Score.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'DetYOLOv3', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
