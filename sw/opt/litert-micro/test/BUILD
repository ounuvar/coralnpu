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

# Tiled versus untiled pointwise conv (#11), on npusim with the memcpy copy path. Run it with and
# without --//sw/opt/litert-micro:rvv_conv to cover the RVV and the reference kernel. Variants: copies
# at Start (default) or at Wait, each with and without prefetch (conv_tiling_test_* libraries).
[
    coralnpu_v2_binary(
        name = "conv_tiled_test" + variant,
        srcs = ["conv_tiled_test.cc"],
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [lib],
    )
    for variant, lib in [
        ("", "//sw/opt/litert-micro:conv"),
        ("_wait", "//sw/opt/litert-micro:conv_tiling_test_wait"),
        ("_prefetch", "//sw/opt/litert-micro:conv_tiling_test_prefetch"),
        ("_prefetch_wait", "//sw/opt/litert-micro:conv_tiling_test_prefetch_wait"),
        ("_r4", "//sw/opt/litert-micro:conv_tiling_test_r4"),
        ("_spread", "//sw/opt/litert-micro:conv_tiling_test_spread"),
        ("_spread_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_wait"),
        ("_spread_r4", "//sw/opt/litert-micro:conv_tiling_test_spread_r4"),
        ("_spread_r4_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_r4_wait"),
        ("_spread_balance", "//sw/opt/litert-micro:conv_tiling_test_spread_balance"),
        ("_spread_balance_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_balance_wait"),
    ]
]

[
    coralflow_test(
        name = "conv_tiled_sim_test" + variant,
        srcs = ["conv_tiled_test.py"],
        elf = ":conv_tiled_test%s.elf" % variant,
        highmem = True,
        main = "conv_tiled_test.py",
        targets = ["npusim"],
        deps = [
            "@bazel_tools//tools/python/runfiles",
        ],
    )
    for variant in [
        "",
        "_wait",
        "_prefetch",
        "_prefetch_wait",
        "_r4",
        "_spread",
        "_spread_wait",
        "_spread_r4",
        "_spread_r4_wait",
        "_spread_balance",
        "_spread_balance_wait",
    ]
]

# Whole pointwise layers through the real dispatcher on the SoC testharness (#11), all with the RVV
# kernel: everything in DTCM (_local, the ceiling), data in external memory without tiling (_ext),
# the DMA-tiled path (_tiled), with prefetch of the next chunk's start-up data (_tiled_pf), and the
# last two with per-phase cycle counters (_stats). Layers l28 and l30 are measured at full size; l17
# and l2 at two pixel counts each, to separate per-layer from per-pixel cost.
[
    coralnpu_v2_binary(
        name = "conv_layer_bench_%s_p%d_%s" % (layer, pixels, variant),
        srcs = ["conv_layer_bench.cc"],
        copts = [
                    "-DLAYER_PIXELS=%d" % pixels,
                    "-DIN_D=%d" % in_d,
                    "-DOUT_D=%d" % out_d,
                ] + ([] if variant == "local" else ["-DPLACE_EXT"]) +
                (["-DBENCH_STATS"] if variant.endswith("_stats") else []),
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [
            "//sw/opt/litert-micro:conv_rvv_tiled_dma" + {
                "tiled": "",
                "tiled_stats": "_stats",
                "tiled_pf": "_prefetch",
                "tiled_pf_stats": "_prefetch_stats",
                "tiled_pf_spread_stats": "_prefetch_spread_stats",
                "tiled_pf_spread_balance_stats": "_prefetch_spread_balance_stats",
            }[variant] if variant.startswith("tiled") else "//sw/opt/litert-micro:conv_rvv",
            "//sw/utils",
        ],
    )
    for layer, pixels, in_d, out_d, variants in [
        (
            "l30",
            49,
            256,
            256,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
                "tiled_stats",
                "tiled_pf_stats",
                "tiled_pf_spread_stats",
                "tiled_pf_spread_balance_stats",
            ],
        ),
        (
            "l28",
            49,
            128,
            256,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
                "tiled_stats",
                "tiled_pf_stats",
                "tiled_pf_spread_stats",
                "tiled_pf_spread_balance_stats",
            ],
        ),
        (
            "l17",
            33,
            128,
            128,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
                "tiled_pf_spread_stats",
                "tiled_pf_spread_balance_stats",
            ],
        ),
        (
            "l17",
            99,
            128,
            128,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
                "tiled_stats",
                "tiled_pf_stats",
                "tiled_pf_spread_stats",
                "tiled_pf_spread_balance_stats",
            ],
        ),
        (
            "l2",
            128,
            8,
            16,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
            ],
        ),
        (
            "l2",
            384,
            8,
            16,
            [
                "local",
                "ext",
                "tiled",
                "tiled_pf",
            ],
        ),
    ]
    for variant in variants
]

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

# Host tests for the pointwise tile planner (#11).
cc_test(
    name = "conv_tile_plan_test",
    srcs = ["conv_tile_plan_test.cc"],
    deps = [
        "//sw/opt/litert-micro:conv_tile_plan",
        "@com_google_googletest//:gtest_main",
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
        "//sw/opt/litert-micro:dma_tile",
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
        "//sw/opt/litert-micro:dma_tile",
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
        "//sw/opt/litert-micro:dma_tile",
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
        "//sw/opt/litert-micro:dma_tile",
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
        "//sw/opt/litert-micro:dma_tile",
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
            "//sw/opt/litert-micro:dma_tile",
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
        "//sw/opt/litert-micro:dma_tile",
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
