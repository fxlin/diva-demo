# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='common.proto',
  package='common',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x0c\x63ommon.proto\x12\x06\x63ommon\"E\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0f\n\x07\x63hannel\x18\x04 \x01(\x05\"|\n\x0f\x44\x65tFrameRequest\x12\x1c\n\x05image\x18\x01 \x01(\x0b\x32\r.common.Image\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0b\n\x03\x63ls\x18\x03 \x01(\t\x12\x11\n\tcam_score\x18\x04 \x01(\x02\x12\x10\n\x08nn_score\x18\x05 \x01(\x02\x12\x0b\n\x03qid\x18\x06 \x01(\x05\"\x14\n\x04\x46ile\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"z\n\x0cVideoRequest\x12\x11\n\ttimestamp\x18\x01 \x01(\x05\x12\x0e\n\x06offset\x18\x02 \x01(\x05\x12\x12\n\nvideo_name\x18\x03 \x01(\t\x12\x13\n\x0bobject_name\x18\x04 \x01(\t\x12\x1e\n\x06\x63\x61mera\x18\x05 \x01(\x0b\x32\x0e.common.Camera\"\'\n\x06\x43\x61mera\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07\x61\x64\x64ress\x18\x02 \x01(\t\"9\n\x0fget_videos_resp\x12&\n\x06videos\x18\x01 \x03(\x0b\x32\x16.common.video_metadata\"\xa2\x01\n\x0evideo_metadata\x12\x0e\n\x06\x66rames\x18\x01 \x01(\x05\x12\x16\n\x0escore_file_url\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x1e\n\x06\x63\x61mera\x18\x04 \x01(\x0b\x32\x0e.common.Camera\x12\x11\n\tvideo_url\x18\x05 \x01(\t\x12\x12\n\nimages_url\x18\x06 \x01(\t\x12\x13\n\x0bobject_name\x18\x07 \x01(\t\"K\n\x11\x44\x65tFrameRequestv2\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0b\n\x03\x63ls\x18\x03 \x01(\t\x12\r\n\x05score\x18\x04 \x01(\x05\x62\x06proto3'
)




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='common.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='common.Image.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='common.Image.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='common.Image.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel', full_name='common.Image.channel', index=3,
      number=4, type=5, cpp_type=1, label=1,
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
  serialized_start=24,
  serialized_end=93,
)


_DETFRAMEREQUEST = _descriptor.Descriptor(
  name='DetFrameRequest',
  full_name='common.DetFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='common.DetFrameRequest.image', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='common.DetFrameRequest.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cls', full_name='common.DetFrameRequest.cls', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cam_score', full_name='common.DetFrameRequest.cam_score', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nn_score', full_name='common.DetFrameRequest.nn_score', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='qid', full_name='common.DetFrameRequest.qid', index=5,
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
  serialized_start=95,
  serialized_end=219,
)


_FILE = _descriptor.Descriptor(
  name='File',
  full_name='common.File',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='common.File.data', index=0,
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
  serialized_start=221,
  serialized_end=241,
)


_VIDEOREQUEST = _descriptor.Descriptor(
  name='VideoRequest',
  full_name='common.VideoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='common.VideoRequest.timestamp', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='offset', full_name='common.VideoRequest.offset', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_name', full_name='common.VideoRequest.video_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_name', full_name='common.VideoRequest.object_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='camera', full_name='common.VideoRequest.camera', index=4,
      number=5, type=11, cpp_type=10, label=1,
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
  serialized_start=243,
  serialized_end=365,
)


_CAMERA = _descriptor.Descriptor(
  name='Camera',
  full_name='common.Camera',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='common.Camera.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='common.Camera.address', index=1,
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
  serialized_start=367,
  serialized_end=406,
)


_GET_VIDEOS_RESP = _descriptor.Descriptor(
  name='get_videos_resp',
  full_name='common.get_videos_resp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='videos', full_name='common.get_videos_resp.videos', index=0,
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
  serialized_start=408,
  serialized_end=465,
)


_VIDEO_METADATA = _descriptor.Descriptor(
  name='video_metadata',
  full_name='common.video_metadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frames', full_name='common.video_metadata.frames', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score_file_url', full_name='common.video_metadata.score_file_url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='common.video_metadata.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='camera', full_name='common.video_metadata.camera', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_url', full_name='common.video_metadata.video_url', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='images_url', full_name='common.video_metadata.images_url', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_name', full_name='common.video_metadata.object_name', index=6,
      number=7, type=9, cpp_type=9, label=1,
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
  serialized_start=468,
  serialized_end=630,
)


_DETFRAMEREQUESTV2 = _descriptor.Descriptor(
  name='DetFrameRequestv2',
  full_name='common.DetFrameRequestv2',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='common.DetFrameRequestv2.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='common.DetFrameRequestv2.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cls', full_name='common.DetFrameRequestv2.cls', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='common.DetFrameRequestv2.score', index=3,
      number=4, type=5, cpp_type=1, label=1,
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
  serialized_start=632,
  serialized_end=707,
)

_DETFRAMEREQUEST.fields_by_name['image'].message_type = _IMAGE
_VIDEOREQUEST.fields_by_name['camera'].message_type = _CAMERA
_GET_VIDEOS_RESP.fields_by_name['videos'].message_type = _VIDEO_METADATA
_VIDEO_METADATA.fields_by_name['camera'].message_type = _CAMERA
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['DetFrameRequest'] = _DETFRAMEREQUEST
DESCRIPTOR.message_types_by_name['File'] = _FILE
DESCRIPTOR.message_types_by_name['VideoRequest'] = _VIDEOREQUEST
DESCRIPTOR.message_types_by_name['Camera'] = _CAMERA
DESCRIPTOR.message_types_by_name['get_videos_resp'] = _GET_VIDEOS_RESP
DESCRIPTOR.message_types_by_name['video_metadata'] = _VIDEO_METADATA
DESCRIPTOR.message_types_by_name['DetFrameRequestv2'] = _DETFRAMEREQUESTV2
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.Image)
  })
_sym_db.RegisterMessage(Image)

DetFrameRequest = _reflection.GeneratedProtocolMessageType('DetFrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETFRAMEREQUEST,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.DetFrameRequest)
  })
_sym_db.RegisterMessage(DetFrameRequest)

File = _reflection.GeneratedProtocolMessageType('File', (_message.Message,), {
  'DESCRIPTOR' : _FILE,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.File)
  })
_sym_db.RegisterMessage(File)

VideoRequest = _reflection.GeneratedProtocolMessageType('VideoRequest', (_message.Message,), {
  'DESCRIPTOR' : _VIDEOREQUEST,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.VideoRequest)
  })
_sym_db.RegisterMessage(VideoRequest)

Camera = _reflection.GeneratedProtocolMessageType('Camera', (_message.Message,), {
  'DESCRIPTOR' : _CAMERA,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.Camera)
  })
_sym_db.RegisterMessage(Camera)

get_videos_resp = _reflection.GeneratedProtocolMessageType('get_videos_resp', (_message.Message,), {
  'DESCRIPTOR' : _GET_VIDEOS_RESP,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.get_videos_resp)
  })
_sym_db.RegisterMessage(get_videos_resp)

video_metadata = _reflection.GeneratedProtocolMessageType('video_metadata', (_message.Message,), {
  'DESCRIPTOR' : _VIDEO_METADATA,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.video_metadata)
  })
_sym_db.RegisterMessage(video_metadata)

DetFrameRequestv2 = _reflection.GeneratedProtocolMessageType('DetFrameRequestv2', (_message.Message,), {
  'DESCRIPTOR' : _DETFRAMEREQUESTV2,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:common.DetFrameRequestv2)
  })
_sym_db.RegisterMessage(DetFrameRequestv2)


# @@protoc_insertion_point(module_scope)
