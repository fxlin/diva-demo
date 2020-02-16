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
        '/det_yolov3.DetYOLOv3/DetFrame',
        request_serializer=det__yolov3__pb2.DetFrameRequest.SerializeToString,
        response_deserializer=det__yolov3__pb2.Score.FromString,
        )
    self.Detect = channel.unary_unary(
        '/det_yolov3.DetYOLOv3/Detect',
        request_serializer=det__yolov3__pb2.DetectionRequest.SerializeToString,
        response_deserializer=det__yolov3__pb2.DetectionOutput.FromString,
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

  def Detect(self, request, context):
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
      'Detect': grpc.unary_unary_rpc_method_handler(
          servicer.Detect,
          request_deserializer=det__yolov3__pb2.DetectionRequest.FromString,
          response_serializer=det__yolov3__pb2.DetectionOutput.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'det_yolov3.DetYOLOv3', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
