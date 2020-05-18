# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import common_pb2 as common__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import server_diva_pb2 as server__diva__pb2


class server_divaStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.detect_object_in_video = channel.unary_unary(
                '/server_diva.server_diva/detect_object_in_video',
                request_serializer=server__diva__pb2.object_video_pair.SerializeToString,
                response_deserializer=server__diva__pb2.detection_result.FromString,
                )
        self.register_camera = channel.unary_unary(
                '/server_diva.server_diva/register_camera',
                request_serializer=server__diva__pb2.camera_info.SerializeToString,
                response_deserializer=server__diva__pb2.response.FromString,
                )
        self.detect_object_in_frame = channel.unary_unary(
                '/server_diva.server_diva/detect_object_in_frame',
                request_serializer=server__diva__pb2.frame_from_camera.SerializeToString,
                response_deserializer=server__diva__pb2.response.FromString,
                )
        self.process_video = channel.unary_unary(
                '/server_diva.server_diva/process_video',
                request_serializer=common__pb2.VideoRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.get_videos = channel.unary_unary(
                '/server_diva.server_diva/get_videos',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=common__pb2.get_videos_resp.FromString,
                )
        self.get_video = channel.unary_unary(
                '/server_diva.server_diva/get_video',
                request_serializer=common__pb2.VideoRequest.SerializeToString,
                response_deserializer=common__pb2.video_metadata.FromString,
                )
        self.SubmitFrame = channel.unary_unary(
                '/server_diva.server_diva/SubmitFrame',
                request_serializer=common__pb2.DetFrameRequest.SerializeToString,
                response_deserializer=server__diva__pb2.StrMsg.FromString,
                )


class server_divaServicer(object):
    """Missing associated documentation comment in .proto file"""

    def detect_object_in_video(self, request, context):
        """deprecated
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def register_camera(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def detect_object_in_frame(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def process_video(self, request, context):
        """maintaining
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

    def SubmitFrame(self, request, context):
        """xzl
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_server_divaServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'detect_object_in_video': grpc.unary_unary_rpc_method_handler(
                    servicer.detect_object_in_video,
                    request_deserializer=server__diva__pb2.object_video_pair.FromString,
                    response_serializer=server__diva__pb2.detection_result.SerializeToString,
            ),
            'register_camera': grpc.unary_unary_rpc_method_handler(
                    servicer.register_camera,
                    request_deserializer=server__diva__pb2.camera_info.FromString,
                    response_serializer=server__diva__pb2.response.SerializeToString,
            ),
            'detect_object_in_frame': grpc.unary_unary_rpc_method_handler(
                    servicer.detect_object_in_frame,
                    request_deserializer=server__diva__pb2.frame_from_camera.FromString,
                    response_serializer=server__diva__pb2.response.SerializeToString,
            ),
            'process_video': grpc.unary_unary_rpc_method_handler(
                    servicer.process_video,
                    request_deserializer=common__pb2.VideoRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
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
            'SubmitFrame': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitFrame,
                    request_deserializer=common__pb2.DetFrameRequest.FromString,
                    response_serializer=server__diva__pb2.StrMsg.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'server_diva.server_diva', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class server_diva(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def detect_object_in_video(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/detect_object_in_video',
            server__diva__pb2.object_video_pair.SerializeToString,
            server__diva__pb2.detection_result.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def register_camera(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/register_camera',
            server__diva__pb2.camera_info.SerializeToString,
            server__diva__pb2.response.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def detect_object_in_frame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/detect_object_in_frame',
            server__diva__pb2.frame_from_camera.SerializeToString,
            server__diva__pb2.response.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/process_video',
            common__pb2.VideoRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/get_videos',
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
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/get_video',
            common__pb2.VideoRequest.SerializeToString,
            common__pb2.video_metadata.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/server_diva.server_diva/SubmitFrame',
            common__pb2.DetFrameRequest.SerializeToString,
            server__diva__pb2.StrMsg.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
