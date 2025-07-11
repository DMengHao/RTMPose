import argparse
import logging
import os
import sys
import torch
import onnx
from onnx import shape_inference
from pathlib import Path
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
config.set_flag(trt.BuilderFlag.FP16)

with open("/17_l_copy.onnx", 'rb') as f:
    parser = trt.OnnxParser(network, logger)
    parser.parse(f.read())

# 添加 profile
input_name = network.get_input(0).name
profile = builder.create_optimization_profile()
profile.set_shape(input_name, (1, 3, 256, 192), (8, 3, 256, 192), (16, 3, 256, 192))
config.add_optimization_profile(profile)

engine = builder.build_engine(network, config)
with open("/home/hhkj/dmh/RZG/rtmlib-main/model/17_l_copy.engine", "wb") as f:
    f.write(engine.serialize())

