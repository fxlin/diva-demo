# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import cam_cloud_pb2 as cam__cloud__pb2


class DivaCameraStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.InitDiva = channel.unary_unary(
        '/camera.DivaCamera/InitDiva',
        request_serializer=cam__cloud__pb2.InitDivaRequest.SerializeToString,
        response_deserializer=cam__cloud__pb2.StrMsg.FromString,
        )
    self.GetFrame = channel.unary_unary(
        '/camera.DivaCamera/GetFrame',
        request_serializer=cam__cloud__pb2.GetFrameRequest.SerializeToString,
        response_deserializer=cam__cloud__pb2.Frame.FromString,
        )
    self.DeployOp = channel.unary_unary(
        '/camera.DivaCamera/DeployOp',
        request_serializer=cam__cloud__pb2.Chunk.SerializeToString,
        response_deserializer=cam__cloud__pb2.StrMsg.FromString,
        )
    self.DeployOpNotify = channel.unary_unary(
        '/camera.DivaCamera/DeployOpNotify',
        request_serializer=cam__cloud__pb2.DeployOpRequest.SerializeToString,
        response_deserializer=cam__cloud__pb2.StrMsg.FromString,
        )
    self.DownloadVideo = channel.unary_unary(
        '/camera.DivaCamera/DownloadVideo',
        request_serializer=cam__cloud__pb2.DeployOpRequest.SerializeToString,
        response_deserializer=cam__cloud__pb2.VideoResponse.FromString,
        )


class DivaCameraServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def InitDiva(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetFrame(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DeployOp(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DeployOpNotify(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DownloadVideo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DivaCameraServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'InitDiva': grpc.unary_unary_rpc_method_handler(
          servicer.InitDiva,
          request_deserializer=cam__cloud__pb2.InitDivaRequest.FromString,
          response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
      ),
      'GetFrame': grpc.unary_unary_rpc_method_handler(
          servicer.GetFrame,
          request_deserializer=cam__cloud__pb2.GetFrameRequest.FromString,
          response_serializer=cam__cloud__pb2.Frame.SerializeToString,
      ),
      'DeployOp': grpc.unary_unary_rpc_method_handler(
          servicer.DeployOp,
          request_deserializer=cam__cloud__pb2.Chunk.FromString,
          response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
      ),
      'DeployOpNotify': grpc.unary_unary_rpc_method_handler(
          servicer.DeployOpNotify,
          request_deserializer=cam__cloud__pb2.DeployOpRequest.FromString,
          response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
      ),
      'DownloadVideo': grpc.unary_unary_rpc_method_handler(
          servicer.DownloadVideo,
          request_deserializer=cam__cloud__pb2.DeployOpRequest.FromString,
          response_serializer=cam__cloud__pb2.VideoResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'camera.DivaCamera', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
