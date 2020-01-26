# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: det-yolov3.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='det-yolov3.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x10\x64\x65t-yolov3.proto\":\n\x0f\x44\x65tFrameRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0b\n\x03\x63ls\x18\x03 \x01(\t\"\x14\n\x05Score\x12\x0b\n\x03res\x18\x01 \x01(\t23\n\tDetYOLOv3\x12&\n\x08\x44\x65tFrame\x12\x10.DetFrameRequest\x1a\x06.Score\"\x00\x62\x06proto3')
)




_DETFRAMEREQUEST = _descriptor.Descriptor(
  name='DetFrameRequest',
  full_name='DetFrameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='DetFrameRequest.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='DetFrameRequest.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cls', full_name='DetFrameRequest.cls', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=20,
  serialized_end=78,
)


_SCORE = _descriptor.Descriptor(
  name='Score',
  full_name='Score',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='res', full_name='Score.res', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=80,
  serialized_end=100,
)

DESCRIPTOR.message_types_by_name['DetFrameRequest'] = _DETFRAMEREQUEST
DESCRIPTOR.message_types_by_name['Score'] = _SCORE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DetFrameRequest = _reflection.GeneratedProtocolMessageType('DetFrameRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETFRAMEREQUEST,
  '__module__' : 'det_yolov3_pb2'
  # @@protoc_insertion_point(class_scope:DetFrameRequest)
  })
_sym_db.RegisterMessage(DetFrameRequest)

Score = _reflection.GeneratedProtocolMessageType('Score', (_message.Message,), {
  'DESCRIPTOR' : _SCORE,
  '__module__' : 'det_yolov3_pb2'
  # @@protoc_insertion_point(class_scope:Score)
  })
_sym_db.RegisterMessage(Score)



_DETYOLOV3 = _descriptor.ServiceDescriptor(
  name='DetYOLOv3',
  full_name='DetYOLOv3',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=102,
  serialized_end=153,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetFrame',
    full_name='DetYOLOv3.DetFrame',
    index=0,
    containing_service=None,
    input_type=_DETFRAMEREQUEST,
    output_type=_SCORE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DETYOLOV3)

DESCRIPTOR.services_by_name['DetYOLOv3'] = _DETYOLOV3

# @@protoc_insertion_point(module_scope)
