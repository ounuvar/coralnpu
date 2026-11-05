# Copyright 2026 Google LLC
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

load("@coralnpu_hw//rules:coralflow_test.bzl", "coralflow_test")
load("@coralnpu_hw//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")
load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

coralnpu_v2_binary(
    name = "max_pool_test",
    srcs = ["max_pool_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:pooling",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

coralnpu_v2_binary(
    name = "conv_test",
    srcs = ["conv_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

coralflow_test(
    name = "max_pool_sim_test",
    srcs = ["max_pool_test.py"],
    elf = ":max_pool_test.elf",
    highmem = True,
    main = "max_pool_test.py",
    targets = [
        "npusim",
        "nexus_fpga",
    ],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)

coralflow_test(
    name = "conv_sim_test",
    srcs = ["conv_test.py"],
    elf = ":conv_test.elf",
    highmem = True,
    main = "conv_test.py",
    targets = [
        "npusim",
        "nexus_fpga",
    ],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)

# Golden tests for the requantization reference model (#9), run on the host.
cc_test(
    name = "requant_ref_test",
    srcs = ["requant_ref_test.cc"],
    deps = [
        "//sw/opt/litert-micro:requant_ref",
        "@com_google_googletest//:gtest_main",
        "@gemmlowp//:fixedpoint",
        "@tflite_micro//tensorflow/lite/kernels/internal:common",
    ],
)

coralnpu_v2_binary(
    name = "conv_vme_test",
    srcs = ["conv_vme_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv_vme",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

# npusim (MPACT models Zvt) checks correctness quickly; the verilator target
# runs the same ELF on the 1 MB VmeCoreMiniHighmemAxi RTL for cycle counts.
coralflow_test(
    name = "conv_vme_sim_test",
    srcs = ["conv_vme_test.py"],
    elf = ":conv_vme_test.elf",
    hdl_toplevel = "VmeCoreMiniHighmemAxi",
    highmem = True,
    main = "conv_vme_test.py",
    targets = [
        "npusim",
        "verilator",
    ],
    deps = [
        # The RTL rule adds numpy itself; npusim gets it from the fixture.
        "@bazel_tools//tools/python/runfiles",
    ],
)

coralnpu_v2_binary(
    name = "fully_connected_vme_test",
    srcs = ["fully_connected_vme_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:fully_connected_vme",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

coralflow_test(
    name = "fully_connected_vme_sim_test",
    srcs = ["fully_connected_vme_test.py"],
    elf = ":fully_connected_vme_test.elf",
    hdl_toplevel = "VmeCoreMiniHighmemAxi",
    highmem = True,
    main = "fully_connected_vme_test.py",
    targets = [
        "npusim",
        "verilator",
    ],
    deps = [
        # The RTL rule adds numpy itself; npusim gets it from the fixture.
        "@bazel_tools//tools/python/runfiles",
    ],
)

coralnpu_v2_binary(
    name = "mobilenet_kernel_bench",
    srcs = ["mobilenet_kernel_bench.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

# Baseline per-layer conv and depthwise cycles for MobileNet (#3), on the RVV
# core without the matrix engine. npusim checks bit-exactness against the
# TFLite reference on every layer; the verilator target gives the cycle counts.
# Pick layers with --test_env=MOBILENET_KERNEL_LAYERS=0,1,4 (model op indices)
# and skip the reference run with --test_env=MOBILENET_KERNEL_SKIP_REF=1.
coralflow_test(
    name = "mobilenet_kernel_bench",
    srcs = ["mobilenet_kernel_bench.py"],
    elf = ":mobilenet_kernel_bench.elf",
    highmem = True,
    main = "mobilenet_kernel_bench.py",
    tags = ["manual"],
    targets = [
        "npusim",
        "verilator",
    ],
    deps = [
        # The RTL rule adds numpy itself; npusim gets it from the fixture.
        "@bazel_tools//tools/python/runfiles",
    ],
)

# Tile-sized conv benchmark for the DMA tiling study (#11). One MobileNet layer 17 tile; the
# _local variant keeps its data in DTCM, the _ext variant keeps it in external memory.
coralnpu_v2_binary(
    name = "ext_mem_tile_bench_local",
    srcs = ["ext_mem_tile_bench.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
    ],
)

coralnpu_v2_binary(
    name = "ext_mem_tile_bench_ext",
    srcs = ["ext_mem_tile_bench.cc"],
    copts = ["-DTILE_IN_EXTMEM"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
    ],
)

coralnpu_v2_binary(
    name = "ext_mem_tile_bench_dma4",
    srcs = ["ext_mem_tile_bench.cc"],
    copts = [
        "-DTILE_DMA",
        "-DDMA_BEAT_LOG2=2",
    ],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//fpga:dma",
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
    ],
)

coralnpu_v2_binary(
    name = "ext_mem_tile_bench_dma16",
    srcs = ["ext_mem_tile_bench.cc"],
    copts = [
        "-DTILE_DMA",
        "-DDMA_BEAT_LOG2=4",
    ],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//fpga:dma",
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
    ],
)

# The same tile, all data in DTCM, also run through the TFLite Micro reference kernel and compared
# byte for byte (#11: the optimized kernel the tiled variants use is bit-exact on this layer shape).
coralnpu_v2_binary(
    name = "ext_mem_tile_bench_ref",
    srcs = ["ext_mem_tile_bench.cc"],
    copts = ["-DTILE_CHECK_REFERENCE"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

# Four-row runs for the double buffering study (#11): the same four rows through the kernel with
# all data in DTCM (the reference output), copied with the DMA one transfer at a time (_dma), and
# copied with the DMA overlapped with compute (_db). 4 and 16 are the DMA beat width in bytes.
[
    coralnpu_v2_binary(
        name = "ext_mem_tile_bench_%s_r4" % variant,
        srcs = ["ext_mem_tile_bench.cc"],
        copts = ["-DTILE_ROWS=4"] + flags,
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [
            "//sw/opt/litert-micro:conv",
            "//sw/utils",
        ] + ([] if variant == "local" else ["//fpga:dma"]),
    )
    for variant, flags in [
        ("local", []),
        (
            "dma4",
            [
                "-DTILE_DMA",
                "-DDMA_BEAT_LOG2=2",
            ],
        ),
        (
            "dma16",
            [
                "-DTILE_DMA",
                "-DDMA_BEAT_LOG2=4",
            ],
        ),
        (
            "db4",
            [
                "-DTILE_DMA",
                "-DTILE_DOUBLE_BUFFER",
                "-DDMA_BEAT_LOG2=2",
            ],
        ),
        (
            "db16",
            [
                "-DTILE_DMA",
                "-DTILE_DOUBLE_BUFFER",
                "-DDMA_BEAT_LOG2=4",
            ],
        ),
    ]
]

coralnpu_v2_binary(
    name = "mobilenet_conv_bench",
    srcs = ["mobilenet_conv_bench.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:conv_vme",
        "//sw/utils",
    ],
)

# Per-layer baseline vs matrix-engine cycles for MobileNet's conv layers (#12).
# npusim checks bit-exactness on every layer; the verilator target gives the
# cycle counts. RTL runs are long; pick layers with
# --test_env=MOBILENET_CONV_LAYERS=0,2,5 (model op indices).
coralflow_test(
    name = "mobilenet_conv_bench",
    srcs = ["mobilenet_conv_bench.py"],
    elf = ":mobilenet_conv_bench.elf",
    hdl_toplevel = "VmeCoreMiniHighmemAxi",
    highmem = True,
    main = "mobilenet_conv_bench.py",
    tags = ["manual"],
    targets = [
        "npusim",
        "verilator",
    ],
    deps = [
        # The RTL rule adds numpy itself; npusim gets it from the fixture.
        "@bazel_tools//tools/python/runfiles",
    ],
)

coralnpu_v2_binary(
    name = "mobilenet_engine_select_test",
    srcs = ["mobilenet_engine_select_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/utils",
        "//tests/cocotb/tutorial/tfmicro:mobilenet_v1_025_224_int8_dummy_lib",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
    ],
)

# Full MobileNet with the per-layer vector/matrix choice (#12): all-vector,
# per-layer and all-matrix runs must produce identical tensors. Meaningful only
# with --//sw/opt/litert-micro:rvv_conv --//sw/opt/litert-micro:conv_engine_select.
# npusim gives instruction counts; the verilator target gives cycles.
coralflow_test(
    name = "mobilenet_engine_select_test",
    srcs = ["mobilenet_engine_select_test.py"],
    elf = ":mobilenet_engine_select_test.elf",
    hdl_toplevel = "VmeCoreMiniHighmemAxi",
    highmem = True,
    main = "mobilenet_engine_select_test.py",
    tags = ["manual"],
    targets = [
        "npusim",
        "verilator",
    ],
    deps = [
        # The RTL rule adds numpy itself; npusim gets it from the fixture.
        "@bazel_tools//tools/python/runfiles",
    ],
)

# ISS check for the fused requantization instructions (#9, step 3): the same
# ELF runs on npusim (MPACT) and on Spike.
coralnpu_v2_binary(
    name = "requant_iss_test",
    srcs = ["requant_iss_test.cc"],
    deps = [
        "//sw/opt/litert-micro:accumulator_util",
        "//sw/opt/litert-micro:requant_ref",
        "//sw/utils",
        "@tflite_micro//tensorflow/lite/kernels/internal:common",
    ],
)

coralflow_test(
    name = "requant_iss_sim_test",
    srcs = ["requant_iss_test.py"],
    elf = ":requant_iss_test.elf",
    main = "requant_iss_test.py",
    targets = ["npusim"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
    ],
)

cc_test(
    name = "requant_iss_spike_test",
    srcs = ["requant_iss_spike_test.cc"],
    data = [":requant_iss_test.elf"],
    deps = [
        "//sw/coralnpu_sim:spike_simulator",
        "//tests/verilator_sim:elf",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_googletest//:gtest_main",
    ],
)
