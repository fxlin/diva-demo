# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: server_diva.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='server_diva.proto',
  package='server_diva',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x11server_diva.proto\x12\x0bserver_diva\x1a\x1bgoogle/protobuf/empty.proto\x1a\x0c\x63ommon.proto\"X\n\x10\x64\x65tection_result\x12(\n\x06retval\x18\x01 \x01(\x0b\x32\x16.google.protobuf.EmptyH\x00\x12\x0f\n\x05\x65rror\x18\x02 \x01(\tH\x00\x42\t\n\x07payload\"\xac\x01\n\x11\x66rame_from_camera\x12\x1c\n\x05image\x18\x01 \x01(\x0b\x32\r.common.Image\x12\x18\n\x10\x63onfidence_score\x18\x02 \x01(\x02\x12(\n\x06\x63\x61mera\x18\x03 \x01(\x0b\x32\x18.server_diva.camera_info\x12\x11\n\ttimestamp\x18\x04 \x01(\x05\x12\x12\n\nvideo_name\x18\x05 \x01(\t\x12\x0e\n\x06offset\x18\x06 \x01(\x05\"C\n\x0b\x63\x61mera_info\x12\x11\n\tcamera_ip\x18\x01 \x01(\t\x12\x13\n\x0b\x63\x61mera_port\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\"0\n\x08response\x12\x13\n\x0bstatus_code\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"<\n\x11object_video_pair\x12\x13\n\x0bobject_name\x18\x03 \x01(\t\x12\x12\n\nvideo_name\x18\x04 \x01(\t2\x83\x03\n\x0bserver_diva\x12Y\n\x16\x64\x65tect_object_in_video\x12\x1e.server_diva.object_video_pair\x1a\x1d.server_diva.detection_result\"\x00\x12\x44\n\x0fregister_camera\x12\x18.server_diva.camera_info\x1a\x15.server_diva.response\"\x00\x12Q\n\x16\x64\x65tect_object_in_frame\x12\x1e.server_diva.frame_from_camera\x1a\x15.server_diva.response\"\x00\x12?\n\rprocess_video\x12\x14.common.VideoRequest\x1a\x16.google.protobuf.Empty\"\x00\x12?\n\nget_videos\x12\x16.google.protobuf.Empty\x1a\x17.common.get_videos_resp\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,common__pb2.DESCRIPTOR,])




_DETECTION_RESULT = _descriptor.Descriptor(
  name='detection_result',
  full_name='server_diva.detection_result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='retval', full_name='server_diva.detection_result.retval', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error', full_name='server_diva.detection_result.error', index=1,
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
    _descriptor.OneofDescriptor(
      name='payload', full_name='server_diva.detection_result.payload',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=77,
  serialized_end=165,
)


_FRAME_FROM_CAMERA = _descriptor.Descriptor(
  name='frame_from_camera',
  full_name='server_diva.frame_from_camera',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='server_diva.frame_from_camera.image', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence_score', full_name='server_diva.frame_from_camera.confidence_score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='camera', full_name='server_diva.frame_from_camera.camera', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='server_diva.frame_from_camera.timestamp', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_name', full_name='server_diva.frame_from_camera.video_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offset', full_name='server_diva.frame_from_camera.offset', index=5,
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
  serialized_start=168,
  serialized_end=340,
)


_CAMERA_INFO = _descriptor.Descriptor(
  name='camera_info',
  full_name='server_diva.camera_info',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera_ip', full_name='server_diva.camera_info.camera_ip', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='camera_port', full_name='server_diva.camera_info.camera_port', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='server_diva.camera_info.name', index=2,
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
  serialized_start=342,
  serialized_end=409,
)


_RESPONSE = _descriptor.Descriptor(
  name='response',
  full_name='server_diva.response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status_code', full_name='server_diva.response.status_code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='server_diva.response.message', index=1,
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
  serialized_start=411,
  serialized_end=459,
)


_OBJECT_VIDEO_PAIR = _descriptor.Descriptor(
  name='object_video_pair',
  full_name='server_diva.object_video_pair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='object_name', full_name='server_diva.object_video_pair.object_name', index=0,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_name', full_name='server_diva.object_video_pair.video_name', index=1,
      number=4, type=9, cpp_type=9, label=1,
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
  serialized_start=461,
  serialized_end=521,
)

_DETECTION_RESULT.fields_by_name['retval'].message_type = google_dot_protobuf_dot_empty__pb2._EMPTY
_DETECTION_RESULT.oneofs_by_name['payload'].fields.append(
  _DETECTION_RESULT.fields_by_name['retval'])
_DETECTION_RESULT.fields_by_name['retval'].containing_oneof = _DETECTION_RESULT.oneofs_by_name['payload']
_DETECTION_RESULT.oneofs_by_name['payload'].fields.append(
  _DETECTION_RESULT.fields_by_name['error'])
_DETECTION_RESULT.fields_by_name['error'].containing_oneof = _DETECTION_RESULT.oneofs_by_name['payload']
_FRAME_FROM_CAMERA.fields_by_name['image'].message_type = common__pb2._IMAGE
_FRAME_FROM_CAMERA.fields_by_name['camera'].message_type = _CAMERA_INFO
DESCRIPTOR.message_types_by_name['detection_result'] = _DETECTION_RESULT
DESCRIPTOR.message_types_by_name['frame_from_camera'] = _FRAME_FROM_CAMERA
DESCRIPTOR.message_types_by_name['camera_info'] = _CAMERA_INFO
DESCRIPTOR.message_types_by_name['response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['object_video_pair'] = _OBJECT_VIDEO_PAIR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

detection_result = _reflection.GeneratedProtocolMessageType('detection_result', (_message.Message,), {
  'DESCRIPTOR' : _DETECTION_RESULT,
  '__module__' : 'server_diva_pb2'
  # @@protoc_insertion_point(class_scope:server_diva.detection_result)
  })
_sym_db.RegisterMessage(detection_result)

frame_from_camera = _reflection.GeneratedProtocolMessageType('frame_from_camera', (_message.Message,), {
  'DESCRIPTOR' : _FRAME_FROM_CAMERA,
  '__module__' : 'server_diva_pb2'
  # @@protoc_insertion_point(class_scope:server_diva.frame_from_camera)
  })
_sym_db.RegisterMessage(frame_from_camera)

camera_info = _reflection.GeneratedProtocolMessageType('camera_info', (_message.Message,), {
  'DESCRIPTOR' : _CAMERA_INFO,
  '__module__' : 'server_diva_pb2'
  # @@protoc_insertion_point(class_scope:server_diva.camera_info)
  })
_sym_db.RegisterMessage(camera_info)

response = _reflection.GeneratedProtocolMessageType('response', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSE,
  '__module__' : 'server_diva_pb2'
  # @@protoc_insertion_point(class_scope:server_diva.response)
  })
_sym_db.RegisterMessage(response)

object_video_pair = _reflection.GeneratedProtocolMessageType('object_video_pair', (_message.Message,), {
  'DESCRIPTOR' : _OBJECT_VIDEO_PAIR,
  '__module__' : 'server_diva_pb2'
  # @@protoc_insertion_point(class_scope:server_diva.object_video_pair)
  })
_sym_db.RegisterMessage(object_video_pair)



_SERVER_DIVA = _descriptor.ServiceDescriptor(
  name='server_diva',
  full_name='server_diva.server_diva',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=524,
  serialized_end=911,
  methods=[
  _descriptor.MethodDescriptor(
    name='detect_object_in_video',
    full_name='server_diva.server_diva.detect_object_in_video',
    index=0,
    containing_service=None,
    input_type=_OBJECT_VIDEO_PAIR,
    output_type=_DETECTION_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='register_camera',
    full_name='server_diva.server_diva.register_camera',
    index=1,
    containing_service=None,
    input_type=_CAMERA_INFO,
    output_type=_RESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='detect_object_in_frame',
    full_name='server_diva.server_diva.detect_object_in_frame',
    index=2,
    containing_service=None,
    input_type=_FRAME_FROM_CAMERA,
    output_type=_RESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='process_video',
    full_name='server_diva.server_diva.process_video',
    index=3,
    containing_service=None,
    input_type=common__pb2._VIDEOREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='get_videos',
    full_name='server_diva.server_diva.get_videos',
    index=4,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=common__pb2._GET_VIDEOS_RESP,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SERVER_DIVA)

DESCRIPTOR.services_by_name['server_diva'] = _SERVER_DIVA

# @@protoc_insertion_point(module_scope)
