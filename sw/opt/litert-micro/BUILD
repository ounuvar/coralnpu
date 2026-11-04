# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Copy interface for DMA tiling (#11). Runs on the DMA engine when CORALNPU_DMA_TILING is
# defined, and as memcpy otherwise (functional tests on npusim). Header only.
cc_library(
    name = "dma_tile",
    hdrs = ["dma_tile.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = ["//fpga:dma"],
)

cc_library(
    name = "memory_util",
    hdrs = ["memory_util.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:compatibility",
    ],
)

# Idea 2 (#9): PostprocessAcc uses the fused requantization instructions
# (vrequant.vv, vrqclamp.vx). Off by default; the instructions run on npusim
# and Spike, and on RTL only once the hardware is built with them.
#   bazel test --//sw/opt/litert-micro:fused_requant ...
bool_flag(
    name = "fused_requant",
    build_setting_default = False,
)

config_setting(
    name = "fused_requant_enabled",
    flag_values = {":fused_requant": "True"},
)

cc_library(
    name = "accumulator_util",
    hdrs = ["accumulator_util.h"],
    defines = select({
        ":fused_requant_enabled": ["CORALNPU_FUSED_REQUANT=1"],
        "//conditions:default": [],
    }),
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:compatibility",
    ],
)

# Scalar reference model for the fused requantization instruction (#9).
# Portable: builds for the host (golden tests) and for CoralNPU (kernel tests).
cc_library(
    name = "requant_ref",
    hdrs = ["requant_ref.h"],
)

cc_library(
    name = "depthwise_conv",
    srcs = [
        "depthwise_conv.cc",
    ],
    hdrs = ["depthwise_conv.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":accumulator_util",
        ":memory_util",
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

# Configuration S (#26): ConvPerChannel dispatches 1x1 and 3x3 convs to RVV
# kernels instead of the scalar TFLite reference. Off by default, so the
# default build matches baseline-v0.
#   bazel test --//sw/opt/litert-micro:rvv_conv ...
bool_flag(
    name = "rvv_conv",
    build_setting_default = False,
)

config_setting(
    name = "rvv_conv_enabled",
    flag_values = {":rvv_conv": "True"},
)

cc_library(
    name = "conv",
    srcs = [
        "conv.cc",
    ],
    hdrs = ["conv.h"],
    local_defines = select({
        ":rvv_conv_enabled": ["CORALNPU_RVV_CONV=1"],
        "//conditions:default": [],
    }),
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":accumulator_util",
        ":memory_util",
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/kernels/internal:common",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

# Matrix-engine (VME/Zvt) kernels for idea 5 (#12).
cc_library(
    name = "vme_util",
    hdrs = ["vme_util.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
)

cc_library(
    name = "conv_vme",
    srcs = ["conv_vme.cc"],
    hdrs = ["conv_vme.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":accumulator_util",
        ":vme_util",
        "@tflite_micro//tensorflow/lite/kernels/internal:common",
        "@tflite_micro//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "fully_connected_vme",
    srcs = ["fully_connected_vme.cc"],
    hdrs = ["fully_connected_vme.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":conv_vme",
        "@tflite_micro//tensorflow/lite/kernels/internal:common",
        "@tflite_micro//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        ":accumulator_util",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

cc_library(
    name = "logistic",
    srcs = ["logistic.cc"],
    hdrs = ["logistic.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
    deps = [
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops",
    ],
)
