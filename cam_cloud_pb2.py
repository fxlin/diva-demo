# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cam_cloud.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='cam_cloud.proto',
  package='camera',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0f\x63\x61m_cloud.proto\x12\x06\x63\x61mera\x1a\x1bgoogle/protobuf/empty.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/duration.proto\x1a\x0c\x63ommon.proto\"E\n\x0cVideoRequest\x12\x11\n\ttimestamp\x18\x01 \x01(\x05\x12\x0e\n\x06offset\x18\x02 \x01(\x05\x12\x12\n\nvideo_name\x18\x03 \x01(\t\"N\n\rVideoResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x1b\n\x05video\x18\x03 \x01(\x0b\x32\x0c.common.File\"-\n\x0f\x44\x65ployOpRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x63rop\x18\x02 \x01(\t\"x\n\x0cQueryRequest\x12\x12\n\nvideo_name\x18\x01 \x01(\t\x12\x10\n\x08op_names\x18\x02 \x03(\t\x12\x0c\n\x04\x63rop\x18\x03 \x01(\t\x12\x0b\n\x03qid\x18\x04 \x01(\x05\x12\x14\n\x0ctarget_class\x18\x05 \x01(\t\x12\x11\n\tframeskip\x18\x06 \x01(\x05\"3\n\x13\x43ontrolQueryRequest\x12\x0b\n\x03qid\x18\x01 \x01(\x05\x12\x0f\n\x07\x63ommand\x18\x02 \x01(\t\"\x9c\x01\n\rQueryProgress\x12\x0b\n\x03qid\x18\x01 \x01(\x05\x12\x12\n\nvideo_name\x18\x02 \x01(\t\x12\x1a\n\x12n_frames_processed\x18\x03 \x01(\x03\x12\x16\n\x0en_frames_total\x18\x04 \x01(\x03\x12\x0e\n\x06status\x18\x05 \x01(\t\x12\x0f\n\x07ts_comp\x18\x06 \x01(\x02\x12\x15\n\rn_frames_sent\x18\x07 \x01(\x03\"#\n\x0fInitDivaRequest\x12\x10\n\x08img_path\x18\x01 \x01(\t\"\x1f\n\x0fGetFrameRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"<\n\x14GetVideoFrameRequest\x12\x12\n\nvideo_name\x18\x01 \x01(\t\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\x03\"#\n\x05\x46rame\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\x15\n\x06StrMsg\x12\x0b\n\x03msg\x18\x01 \x01(\t\"\x15\n\x05\x43hunk\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"2\n\tVideoList\x12%\n\x06videos\x18\x01 \x03(\x0b\x32\x15.camera.VideoMetadata\"\xb6\x01\n\rVideoMetadata\x12\x12\n\nvideo_name\x18\x01 \x01(\t\x12\x10\n\x08n_frames\x18\x02 \x01(\x03\x12\x18\n\x10n_missing_frames\x18\x03 \x01(\x03\x12\x0b\n\x03\x66ps\x18\x04 \x01(\x05\x12\r\n\x05start\x18\x05 \x01(\x02\x12\x0b\n\x03\x65nd\x18\x06 \x01(\x02\x12\x10\n\x08\x64uration\x18\x07 \x01(\x02\x12\x14\n\x0c\x66rame_id_min\x18\x08 \x01(\x03\x12\x14\n\x0c\x66rame_id_max\x18\t \x01(\x03\"3\n\x08\x46rameMap\x12\x11\n\tframe_ids\x18\x01 \x03(\x03\x12\x14\n\x0c\x66rame_states\x18\x02 \x01(\t2\xc9\x07\n\nDivaCamera\x12\x35\n\x08InitDiva\x12\x17.camera.InitDivaRequest\x1a\x0e.camera.StrMsg\"\x00\x12\x34\n\x08GetFrame\x12\x17.camera.GetFrameRequest\x1a\r.camera.Frame\"\x00\x12+\n\x08\x44\x65ployOp\x12\r.camera.Chunk\x1a\x0e.camera.StrMsg\"\x00\x12;\n\x0e\x44\x65ployOpNotify\x12\x17.camera.DeployOpRequest\x1a\x0e.camera.StrMsg\"\x00\x12\x41\n\rDownloadVideo\x12\x17.camera.DeployOpRequest\x1a\x15.camera.VideoResponse\"\x00\x12\x35\n\x0bSubmitQuery\x12\x14.camera.QueryRequest\x1a\x0e.camera.StrMsg\"\x00\x12=\n\x0c\x43ontrolQuery\x12\x1b.camera.ControlQueryRequest\x1a\x0e.camera.StrMsg\"\x00\x12\x33\n\rPromoteFrames\x12\x10.camera.FrameMap\x1a\x0e.camera.StrMsg\"\x00\x12\x32\n\x0c\x44\x65moteFrames\x12\x10.camera.FrameMap\x1a\x0e.camera.StrMsg\"\x00\x12@\n\x08GetStats\x12\x1b.camera.ControlQueryRequest\x1a\x15.camera.QueryProgress\"\x00\x12\x46\n\x13GetQueryFrameStates\x12\x1b.camera.ControlQueryRequest\x1a\x10.camera.FrameMap\"\x00\x12\x39\n\nListVideos\x12\x16.google.protobuf.Empty\x1a\x11.camera.VideoList\"\x00\x12>\n\rGetVideoFrame\x12\x1c.camera.GetVideoFrameRequest\x1a\r.common.Image\"\x00\x12?\n\nget_videos\x12\x16.google.protobuf.Empty\x1a\x17.common.get_videos_resp\"\x00\x12;\n\tget_video\x12\x14.common.VideoRequest\x1a\x16.common.video_metadata\"\x00\x12?\n\rprocess_video\x12\x14.common.VideoRequest\x1a\x16.google.protobuf.Empty\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,google_dot_protobuf_dot_duration__pb2.DESCRIPTOR,common__pb2.DESCRIPTOR,])




_VIDEOREQUEST = _descriptor.Descriptor(
  name='VideoRequest',
  full_name='camera.VideoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='camera.VideoRequest.timestamp', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offset', full_name='camera.VideoRequest.offset', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_name', full_name='camera.VideoRequest.video_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=135,
  serialized_end=204,
)


_VIDEORESPONSE = _descriptor.Descriptor(
  name='VideoResponse',
  full_name='camera.VideoResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='camera.VideoResponse.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status_code', full_name='camera.VideoResponse.status_code', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video', full_name='camera.VideoResponse.video', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=206,
  serialized_end=284,
)


_DEPLOYOPREQUEST = _descriptor.Descriptor(
  name='DeployOpRequest',
  full_name='camera.DeployOpRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='camera.DeployOpRequest.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop', full_name='camera.DeployOpRequest.crop', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=286,
  serialized_end=331,
)


_QUERYREQUEST = _descriptor.Descriptor(
  name='QueryRequest',
  full_name='camera.QueryRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_name', full_name='camera.QueryRequest.video_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='op_names', full_name='camera.QueryRequest.op_names', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crop', full_name='camera.QueryRequest.crop', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='qid', full_name='camera.QueryRequest.qid', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='target_class', full_name='camera.QueryRequest.target_class', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frameskip', full_name='camera.QueryRequest.frameskip', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=333,
  serialized_end=453,
)


_CONTROLQUERYREQUEST = _descriptor.Descriptor(
  name='ControlQueryRequest',
  full_name='camera.ControlQueryRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='qid', full_name='camera.ControlQueryRequest.qid', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='command', full_name='camera.ControlQueryRequest.command', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=455,
  serialized_end=506,
)


_QUERYPROGRESS = _descriptor.Descriptor(
  name='QueryProgress',
  full_name='camera.QueryProgress',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='qid', full_name='camera.QueryProgress.qid', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_name', full_name='camera.QueryProgress.video_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_frames_processed', full_name='camera.QueryProgress.n_frames_processed', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_frames_total', full_name='camera.QueryProgress.n_frames_total', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='camera.QueryProgress.status', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ts_comp', full_name='camera.QueryProgress.ts_comp', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_frames_sent', full_name='camera.QueryProgress.n_frames_sent', index=6,
      number=7, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=509,
  serialized_end=665,
)


_INITDIVAREQUEST = _descriptor.Descriptor(
  name='InitDivaRequest',
  full_name='camera.InitDivaRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='img_path', full_name='camera.InitDivaRequest.img_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=667,
  serialized_end=702,
)


_GETFRAMEREQUEST = _descriptor.Descriptor(
  name='GetFrameRequest',
  full_name='camera.GetFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='camera.GetFrameRequest.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=704,
  serialized_end=735,
)


_GETVIDEOFRAMEREQUEST = _descriptor.Descriptor(
  name='GetVideoFrameRequest',
  full_name='camera.GetVideoFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_name', full_name='camera.GetVideoFrameRequest.video_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='camera.GetVideoFrameRequest.frame_id', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=737,
  serialized_end=797,
)


_FRAME = _descriptor.Descriptor(
  name='Frame',
  full_name='camera.Frame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='camera.Frame.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='camera.Frame.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=799,
  serialized_end=834,
)


_STRMSG = _descriptor.Descriptor(
  name='StrMsg',
  full_name='camera.StrMsg',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='camera.StrMsg.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=836,
  serialized_end=857,
)


_CHUNK = _descriptor.Descriptor(
  name='Chunk',
  full_name='camera.Chunk',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='camera.Chunk.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=859,
  serialized_end=880,
)


_VIDEOLIST = _descriptor.Descriptor(
  name='VideoList',
  full_name='camera.VideoList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='videos', full_name='camera.VideoList.videos', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=882,
  serialized_end=932,
)


_VIDEOMETADATA = _descriptor.Descriptor(
  name='VideoMetadata',
  full_name='camera.VideoMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_name', full_name='camera.VideoMetadata.video_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_frames', full_name='camera.VideoMetadata.n_frames', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_missing_frames', full_name='camera.VideoMetadata.n_missing_frames', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fps', full_name='camera.VideoMetadata.fps', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='camera.VideoMetadata.start', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='camera.VideoMetadata.end', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='duration', full_name='camera.VideoMetadata.duration', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_id_min', full_name='camera.VideoMetadata.frame_id_min', index=7,
      number=8, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_id_max', full_name='camera.VideoMetadata.frame_id_max', index=8,
      number=9, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=935,
  serialized_end=1117,
)


_FRAMEMAP = _descriptor.Descriptor(
  name='FrameMap',
  full_name='camera.FrameMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_ids', full_name='camera.FrameMap.frame_ids', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_states', full_name='camera.FrameMap.frame_states', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1119,
  serialized_end=1170,
)

_VIDEORESPONSE.fields_by_name['video'].message_type = common__pb2._FILE
_VIDEOLIST.fields_by_name['videos'].message_type = _VIDEOMETADATA
DESCRIPTOR.message_types_by_name['VideoRequest'] = _VIDEOREQUEST
DESCRIPTOR.message_types_by_name['VideoResponse'] = _VIDEORESPONSE
DESCRIPTOR.message_types_by_name['DeployOpRequest'] = _DEPLOYOPREQUEST
DESCRIPTOR.message_types_by_name['QueryRequest'] = _QUERYREQUEST
DESCRIPTOR.message_types_by_name['ControlQueryRequest'] = _CONTROLQUERYREQUEST
DESCRIPTOR.message_types_by_name['QueryProgress'] = _QUERYPROGRESS
DESCRIPTOR.message_types_by_name['InitDivaRequest'] = _INITDIVAREQUEST
DESCRIPTOR.message_types_by_name['GetFrameRequest'] = _GETFRAMEREQUEST
DESCRIPTOR.message_types_by_name['GetVideoFrameRequest'] = _GETVIDEOFRAMEREQUEST
DESCRIPTOR.message_types_by_name['Frame'] = _FRAME
DESCRIPTOR.message_types_by_name['StrMsg'] = _STRMSG
DESCRIPTOR.message_types_by_name['Chunk'] = _CHUNK
DESCRIPTOR.message_types_by_name['VideoList'] = _VIDEOLIST
DESCRIPTOR.message_types_by_name['VideoMetadata'] = _VIDEOMETADATA
DESCRIPTOR.message_types_by_name['FrameMap'] = _FRAMEMAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VideoRequest = _reflection.GeneratedProtocolMessageType('VideoRequest', (_message.Message,), {
  'DESCRIPTOR' : _VIDEOREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.VideoRequest)
  })
_sym_db.RegisterMessage(VideoRequest)

VideoResponse = _reflection.GeneratedProtocolMessageType('VideoResponse', (_message.Message,), {
  'DESCRIPTOR' : _VIDEORESPONSE,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.VideoResponse)
  })
_sym_db.RegisterMessage(VideoResponse)

DeployOpRequest = _reflection.GeneratedProtocolMessageType('DeployOpRequest', (_message.Message,), {
  'DESCRIPTOR' : _DEPLOYOPREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.DeployOpRequest)
  })
_sym_db.RegisterMessage(DeployOpRequest)

QueryRequest = _reflection.GeneratedProtocolMessageType('QueryRequest', (_message.Message,), {
  'DESCRIPTOR' : _QUERYREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.QueryRequest)
  })
_sym_db.RegisterMessage(QueryRequest)

ControlQueryRequest = _reflection.GeneratedProtocolMessageType('ControlQueryRequest', (_message.Message,), {
  'DESCRIPTOR' : _CONTROLQUERYREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.ControlQueryRequest)
  })
_sym_db.RegisterMessage(ControlQueryRequest)

QueryProgress = _reflection.GeneratedProtocolMessageType('QueryProgress', (_message.Message,), {
  'DESCRIPTOR' : _QUERYPROGRESS,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.QueryProgress)
  })
_sym_db.RegisterMessage(QueryProgress)

InitDivaRequest = _reflection.GeneratedProtocolMessageType('InitDivaRequest', (_message.Message,), {
  'DESCRIPTOR' : _INITDIVAREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.InitDivaRequest)
  })
_sym_db.RegisterMessage(InitDivaRequest)

GetFrameRequest = _reflection.GeneratedProtocolMessageType('GetFrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETFRAMEREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.GetFrameRequest)
  })
_sym_db.RegisterMessage(GetFrameRequest)

GetVideoFrameRequest = _reflection.GeneratedProtocolMessageType('GetVideoFrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETVIDEOFRAMEREQUEST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.GetVideoFrameRequest)
  })
_sym_db.RegisterMessage(GetVideoFrameRequest)

Frame = _reflection.GeneratedProtocolMessageType('Frame', (_message.Message,), {
  'DESCRIPTOR' : _FRAME,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.Frame)
  })
_sym_db.RegisterMessage(Frame)

StrMsg = _reflection.GeneratedProtocolMessageType('StrMsg', (_message.Message,), {
  'DESCRIPTOR' : _STRMSG,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.StrMsg)
  })
_sym_db.RegisterMessage(StrMsg)

Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), {
  'DESCRIPTOR' : _CHUNK,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.Chunk)
  })
_sym_db.RegisterMessage(Chunk)

VideoList = _reflection.GeneratedProtocolMessageType('VideoList', (_message.Message,), {
  'DESCRIPTOR' : _VIDEOLIST,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.VideoList)
  })
_sym_db.RegisterMessage(VideoList)

VideoMetadata = _reflection.GeneratedProtocolMessageType('VideoMetadata', (_message.Message,), {
  'DESCRIPTOR' : _VIDEOMETADATA,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.VideoMetadata)
  })
_sym_db.RegisterMessage(VideoMetadata)

FrameMap = _reflection.GeneratedProtocolMessageType('FrameMap', (_message.Message,), {
  'DESCRIPTOR' : _FRAMEMAP,
  '__module__' : 'cam_cloud_pb2'
  # @@protoc_insertion_point(class_scope:camera.FrameMap)
  })
_sym_db.RegisterMessage(FrameMap)



_DIVACAMERA = _descriptor.ServiceDescriptor(
  name='DivaCamera',
  full_name='camera.DivaCamera',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=1173,
  serialized_end=2142,
  methods=[
  _descriptor.MethodDescriptor(
    name='InitDiva',
    full_name='camera.DivaCamera.InitDiva',
    index=0,
    containing_service=None,
    input_type=_INITDIVAREQUEST,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetFrame',
    full_name='camera.DivaCamera.GetFrame',
    index=1,
    containing_service=None,
    input_type=_GETFRAMEREQUEST,
    output_type=_FRAME,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DeployOp',
    full_name='camera.DivaCamera.DeployOp',
    index=2,
    containing_service=None,
    input_type=_CHUNK,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DeployOpNotify',
    full_name='camera.DivaCamera.DeployOpNotify',
    index=3,
    containing_service=None,
    input_type=_DEPLOYOPREQUEST,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DownloadVideo',
    full_name='camera.DivaCamera.DownloadVideo',
    index=4,
    containing_service=None,
    input_type=_DEPLOYOPREQUEST,
    output_type=_VIDEORESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SubmitQuery',
    full_name='camera.DivaCamera.SubmitQuery',
    index=5,
    containing_service=None,
    input_type=_QUERYREQUEST,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ControlQuery',
    full_name='camera.DivaCamera.ControlQuery',
    index=6,
    containing_service=None,
    input_type=_CONTROLQUERYREQUEST,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='PromoteFrames',
    full_name='camera.DivaCamera.PromoteFrames',
    index=7,
    containing_service=None,
    input_type=_FRAMEMAP,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='DemoteFrames',
    full_name='camera.DivaCamera.DemoteFrames',
    index=8,
    containing_service=None,
    input_type=_FRAMEMAP,
    output_type=_STRMSG,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetStats',
    full_name='camera.DivaCamera.GetStats',
    index=9,
    containing_service=None,
    input_type=_CONTROLQUERYREQUEST,
    output_type=_QUERYPROGRESS,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetQueryFrameStates',
    full_name='camera.DivaCamera.GetQueryFrameStates',
    index=10,
    containing_service=None,
    input_type=_CONTROLQUERYREQUEST,
    output_type=_FRAMEMAP,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ListVideos',
    full_name='camera.DivaCamera.ListVideos',
    index=11,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_VIDEOLIST,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetVideoFrame',
    full_name='camera.DivaCamera.GetVideoFrame',
    index=12,
    containing_service=None,
    input_type=_GETVIDEOFRAMEREQUEST,
    output_type=common__pb2._IMAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='get_videos',
    full_name='camera.DivaCamera.get_videos',
    index=13,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=common__pb2._GET_VIDEOS_RESP,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='get_video',
    full_name='camera.DivaCamera.get_video',
    index=14,
    containing_service=None,
    input_type=common__pb2._VIDEOREQUEST,
    output_type=common__pb2._VIDEO_METADATA,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='process_video',
    full_name='camera.DivaCamera.process_video',
    index=15,
    containing_service=None,
    input_type=common__pb2._VIDEOREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DIVACAMERA)

DESCRIPTOR.services_by_name['DivaCamera'] = _DIVACAMERA

# @@protoc_insertion_point(module_scope)
