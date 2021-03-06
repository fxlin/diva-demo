# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import cam_cloud_pb2 as cam__cloud__pb2
import common_pb2 as common__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


class DivaCameraStub(object):
    """Missing associated documentation comment in .proto file"""

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
        self.SubmitQuery = channel.unary_unary(
                '/camera.DivaCamera/SubmitQuery',
                request_serializer=cam__cloud__pb2.QueryRequest.SerializeToString,
                response_deserializer=cam__cloud__pb2.StrMsg.FromString,
                )
        self.ControlQuery = channel.unary_unary(
                '/camera.DivaCamera/ControlQuery',
                request_serializer=cam__cloud__pb2.ControlQueryRequest.SerializeToString,
                response_deserializer=cam__cloud__pb2.StrMsg.FromString,
                )
        self.PromoteFrames = channel.unary_unary(
                '/camera.DivaCamera/PromoteFrames',
                request_serializer=cam__cloud__pb2.FrameMap.SerializeToString,
                response_deserializer=cam__cloud__pb2.StrMsg.FromString,
                )
        self.DemoteFrames = channel.unary_unary(
                '/camera.DivaCamera/DemoteFrames',
                request_serializer=cam__cloud__pb2.FrameMap.SerializeToString,
                response_deserializer=cam__cloud__pb2.StrMsg.FromString,
                )
        self.GetStats = channel.unary_unary(
                '/camera.DivaCamera/GetStats',
                request_serializer=cam__cloud__pb2.ControlQueryRequest.SerializeToString,
                response_deserializer=cam__cloud__pb2.QueryProgress.FromString,
                )
        self.GetQueryFrameStates = channel.unary_unary(
                '/camera.DivaCamera/GetQueryFrameStates',
                request_serializer=cam__cloud__pb2.ControlQueryRequest.SerializeToString,
                response_deserializer=cam__cloud__pb2.FrameMap.FromString,
                )
        self.GetCamSpecs = channel.unary_unary(
                '/camera.DivaCamera/GetCamSpecs',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=cam__cloud__pb2.StrMsg.FromString,
                )
        self.ListVideos = channel.unary_unary(
                '/camera.DivaCamera/ListVideos',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=cam__cloud__pb2.VideoList.FromString,
                )
        self.GetVideoFrame = channel.unary_unary(
                '/camera.DivaCamera/GetVideoFrame',
                request_serializer=cam__cloud__pb2.GetVideoFrameRequest.SerializeToString,
                response_deserializer=common__pb2.Image.FromString,
                )
        self.get_videos = channel.unary_unary(
                '/camera.DivaCamera/get_videos',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=common__pb2.get_videos_resp.FromString,
                )
        self.get_video = channel.unary_unary(
                '/camera.DivaCamera/get_video',
                request_serializer=common__pb2.VideoRequest.SerializeToString,
                response_deserializer=common__pb2.video_metadata.FromString,
                )
        self.process_video = channel.unary_unary(
                '/camera.DivaCamera/process_video',
                request_serializer=common__pb2.VideoRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class DivaCameraServicer(object):
    """Missing associated documentation comment in .proto file"""

    def InitDiva(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFrame(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeployOp(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeployOpNotify(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DownloadVideo(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitQuery(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ControlQuery(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PromoteFrames(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DemoteFrames(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetStats(self, request, context):
        """xzl: only ControlQueryRequest.qid is used
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetQueryFrameStates(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetCamSpecs(self, request, context):
        """return a string that describs the hw sepcs of the cam
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListVideos(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetVideoFrame(self, request, context):
        """rpc ListQueries(google.protobuf.Empty) returns (QueryList) {};
        xzl: return encoded image data eg jpg
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_videos(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_video(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def process_video(self, request, context):
        """Missing associated documentation comment in .proto file"""
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
            'SubmitQuery': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitQuery,
                    request_deserializer=cam__cloud__pb2.QueryRequest.FromString,
                    response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
            ),
            'ControlQuery': grpc.unary_unary_rpc_method_handler(
                    servicer.ControlQuery,
                    request_deserializer=cam__cloud__pb2.ControlQueryRequest.FromString,
                    response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
            ),
            'PromoteFrames': grpc.unary_unary_rpc_method_handler(
                    servicer.PromoteFrames,
                    request_deserializer=cam__cloud__pb2.FrameMap.FromString,
                    response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
            ),
            'DemoteFrames': grpc.unary_unary_rpc_method_handler(
                    servicer.DemoteFrames,
                    request_deserializer=cam__cloud__pb2.FrameMap.FromString,
                    response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
            ),
            'GetStats': grpc.unary_unary_rpc_method_handler(
                    servicer.GetStats,
                    request_deserializer=cam__cloud__pb2.ControlQueryRequest.FromString,
                    response_serializer=cam__cloud__pb2.QueryProgress.SerializeToString,
            ),
            'GetQueryFrameStates': grpc.unary_unary_rpc_method_handler(
                    servicer.GetQueryFrameStates,
                    request_deserializer=cam__cloud__pb2.ControlQueryRequest.FromString,
                    response_serializer=cam__cloud__pb2.FrameMap.SerializeToString,
            ),
            'GetCamSpecs': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCamSpecs,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=cam__cloud__pb2.StrMsg.SerializeToString,
            ),
            'ListVideos': grpc.unary_unary_rpc_method_handler(
                    servicer.ListVideos,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=cam__cloud__pb2.VideoList.SerializeToString,
            ),
            'GetVideoFrame': grpc.unary_unary_rpc_method_handler(
                    servicer.GetVideoFrame,
                    request_deserializer=cam__cloud__pb2.GetVideoFrameRequest.FromString,
                    response_serializer=common__pb2.Image.SerializeToString,
            ),
            'get_videos': grpc.unary_unary_rpc_method_handler(
                    servicer.get_videos,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=common__pb2.get_videos_resp.SerializeToString,
            ),
            'get_video': grpc.unary_unary_rpc_method_handler(
                    servicer.get_video,
                    request_deserializer=common__pb2.VideoRequest.FromString,
                    response_serializer=common__pb2.video_metadata.SerializeToString,
            ),
            'process_video': grpc.unary_unary_rpc_method_handler(
                    servicer.process_video,
                    request_deserializer=common__pb2.VideoRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'camera.DivaCamera', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DivaCamera(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def InitDiva(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/InitDiva',
            cam__cloud__pb2.InitDivaRequest.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/GetFrame',
            cam__cloud__pb2.GetFrameRequest.SerializeToString,
            cam__cloud__pb2.Frame.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeployOp(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/DeployOp',
            cam__cloud__pb2.Chunk.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeployOpNotify(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/DeployOpNotify',
            cam__cloud__pb2.DeployOpRequest.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DownloadVideo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/DownloadVideo',
            cam__cloud__pb2.DeployOpRequest.SerializeToString,
            cam__cloud__pb2.VideoResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitQuery(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/SubmitQuery',
            cam__cloud__pb2.QueryRequest.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ControlQuery(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/ControlQuery',
            cam__cloud__pb2.ControlQueryRequest.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PromoteFrames(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/PromoteFrames',
            cam__cloud__pb2.FrameMap.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DemoteFrames(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/DemoteFrames',
            cam__cloud__pb2.FrameMap.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetStats(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/GetStats',
            cam__cloud__pb2.ControlQueryRequest.SerializeToString,
            cam__cloud__pb2.QueryProgress.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetQueryFrameStates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/GetQueryFrameStates',
            cam__cloud__pb2.ControlQueryRequest.SerializeToString,
            cam__cloud__pb2.FrameMap.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetCamSpecs(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/GetCamSpecs',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            cam__cloud__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListVideos(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/ListVideos',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            cam__cloud__pb2.VideoList.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetVideoFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/GetVideoFrame',
            cam__cloud__pb2.GetVideoFrameRequest.SerializeToString,
            common__pb2.Image.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_videos(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/get_videos',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            common__pb2.get_videos_resp.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_video(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/get_video',
            common__pb2.VideoRequest.SerializeToString,
            common__pb2.video_metadata.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def process_video(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/camera.DivaCamera/process_video',
            common__pb2.VideoRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
