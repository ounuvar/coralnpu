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
load("@rules_python//python:defs.bzl", "py_library")

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
        ("_spread", "//sw/opt/litert-micro:conv_tiling_test_spread"),
        ("_spread_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_wait"),
        ("_spread_merge", "//sw/opt/litert-micro:conv_tiling_test_spread_merge"),
        ("_spread_merge_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_merge_wait"),
        ("_spread_merge_overlap", "//sw/opt/litert-micro:conv_tiling_test_spread_merge_overlap"),
        ("_spread_merge_overlap_wait", "//sw/opt/litert-micro:conv_tiling_test_spread_merge_overlap_wait"),
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
        "_spread",
        "_spread_wait",
        "_spread_merge",
        "_spread_merge_wait",
        "_spread_merge_overlap",
        "_spread_merge_overlap_wait",
    ]
]

# Row-tiled versus untiled spatial conv (#11), on npusim with the memcpy copy path: copies at Start
# (default) or at Wait. Run with and without --//sw/opt/litert-micro:rvv_conv.
[
    coralnpu_v2_binary(
        name = "conv_row_tiled_test" + variant,
        srcs = ["conv_row_tiled_test.cc"],
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [lib],
    )
    for variant, lib in [
        ("", "//sw/opt/litert-micro:conv"),
        ("_wait", "//sw/opt/litert-micro:conv_tiling_test_wait"),
    ]
]

[
    coralflow_test(
        name = "conv_row_tiled_sim_test" + variant,
        srcs = ["conv_row_tiled_test.py"],
        elf = ":conv_row_tiled_test%s.elf" % variant,
        highmem = True,
        main = "conv_row_tiled_test.py",
        targets = ["npusim"],
        deps = [
            "@bazel_tools//tools/python/runfiles",
        ],
    )
    for variant in [
        "",
        "_wait",
    ]
]

# MobileNet layer 0 (3x3, stride 2, 3 to 8 channels, SAME) through the dispatcher on the SoC
# testharness (#11): the top 56 rows of the 224x224 input, the same geometry as the whole layer with
# a quarter of the rows (a full layer takes too long to simulate), so cycles are extrapolated by
# four. _local is the ceiling, _ext untiled with data in external memory, _tiled_stats the row-tiled
# path with per-phase counters.
[
    coralnpu_v2_binary(
        name = "conv_layer_bench_l0_h56_" + variant,
        srcs = ["conv_layer_bench.cc"],
        copts = [
                    "-DLAYER_PIXELS=%d" % (56 * 224),
                    "-DLAYER_H=56",
                    "-DLAYER_W=224",
                    "-DFILTER=3",
                    "-DSTRIDE=2",
                    "-DIN_D=3",
                    "-DOUT_D=8",
                ] + ([] if variant == "local" else ["-DPLACE_EXT"]) +
                (["-DBENCH_STATS"] if variant.endswith("_stats") else []),
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [
            "//sw/opt/litert-micro:conv_rvv_tiled_dma_prefetch_spread_merge_overlap_stats" if variant.startswith("tiled") else "//sw/opt/litert-micro:conv_rvv",
            "//sw/utils",
        ],
    )
    for variant in [
        "local",
        "ext",
        "tiled_stats",
    ]
]

# Row-tiled versus untiled depthwise conv (#11), on npusim with the memcpy copy path: copies at
# Start (default) or at Wait (the depthwise_conv_wait library).
# Full MobileNet against TFLite (#11): every conv and depthwise layer of the interpreter run is
# compared with TFLite's reference kernel on the same input. With --//sw/opt/litert-micro:conv_tiling
# the optimized layers are DMA-tiled; the output checksum then matches the build without it.
coralnpu_v2_binary(
    name = "mobilenet_vs_tflite_test",
    srcs = ["mobilenet_vs_tflite_test.cc"],
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

coralflow_test(
    name = "mobilenet_vs_tflite_test",
    srcs = ["mobilenet_vs_tflite_test.py"],
    elf = ":mobilenet_vs_tflite_test.elf",
    highmem = True,
    main = "mobilenet_vs_tflite_test.py",
    tags = ["manual"],
    targets = ["npusim"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
    ],
)

# The tiled paths against TFLite (#11): pointwise, 3x3 row and depthwise tiles at several budgets,
# every byte equal to TFLite's value except #23 rounding ties. Run with and without
# --//sw/opt/litert-micro:rvv_conv; without it the conv tiles must match TFLite exactly.
coralnpu_v2_binary(
    name = "tiled_vs_tflite_test",
    srcs = ["tiled_vs_tflite_test.cc"],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/opt/litert-micro:requant_ref",
        "@tflite_micro//tensorflow/lite/kernels/internal:reference_base",
    ],
)

coralflow_test(
    name = "tiled_vs_tflite_sim_test",
    srcs = ["tiled_vs_tflite_test.py"],
    elf = ":tiled_vs_tflite_test.elf",
    highmem = True,
    main = "tiled_vs_tflite_test.py",
    targets = ["npusim"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
    ],
)

[
    coralnpu_v2_binary(
        name = "depthwise_row_tiled_test" + variant,
        srcs = ["depthwise_row_tiled_test.cc"],
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [lib],
    )
    for variant, lib in [
        ("", "//sw/opt/litert-micro:depthwise_conv"),
        ("_wait", "//sw/opt/litert-micro:depthwise_conv_wait"),
    ]
]

[
    coralflow_test(
        name = "depthwise_row_tiled_sim_test" + variant,
        srcs = ["depthwise_row_tiled_test.py"],
        elf = ":depthwise_row_tiled_test%s.elf" % variant,
        highmem = True,
        main = "depthwise_row_tiled_test.py",
        targets = ["npusim"],
        deps = [
            "@bazel_tools//tools/python/runfiles",
        ],
    )
    for variant in [
        "",
        "_wait",
    ]
]

# Depthwise layers through the real kernel entry on the SoC testharness (#11), the same three ways
# as the layer 0 benchmark: _local (ceiling), _ext (untiled, data in external memory) and
# _tiled_stats (row-tiled path with per-phase counters). dw1_h28 is the top 28 rows of layer 1
# (112x112x8, stride 1, the largest accumulators per row), dw6_h28 the top 28 rows of layer 6
# (56x56x32, stride 2) and dw13 layer 13 whole (14x14x128, stride 1, as layers 15 to 23).
[
    coralnpu_v2_binary(
        name = "depthwise_layer_bench_%s_%s" % (layer, variant),
        srcs = ["depthwise_layer_bench.cc"],
        copts = [
                    "-DDW_H=%d" % h,
                    "-DDW_W=%d" % w,
                    "-DDW_C=%d" % c,
                    "-DDW_STRIDE=%d" % stride,
                ] + ([] if variant == "local" else ["-DPLACE_EXT"]) +
                (["-DBENCH_STATS"] if variant.endswith("_stats") else []),
        dtcm_size_kbytes = 1024,
        itcm_size_kbytes = 1024,
        deps = [
            "//sw/opt/litert-micro:depthwise_conv_tiled_dma_stats" if variant.startswith("tiled") else "//sw/opt/litert-micro:depthwise_conv",
            "//sw/utils",
        ],
    )
    for layer, h, w, c, stride in [
        ("dw1_h28", 28, 112, 8, 1),
        ("dw6_h28", 28, 56, 32, 2),
        ("dw13", 14, 14, 128, 1),
    ]
    for variant in [
        "local",
        "ext",
        "tiled_stats",
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
                "tiled_pf_spread_merge_stats": "_prefetch_spread_merge_stats",
                "tiled_pf_spread_merge_overlap_stats": "_prefetch_spread_merge_overlap_stats",
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
                "tiled_pf_spread_merge_stats",
                "tiled_pf_spread_merge_overlap_stats",
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
                "tiled_pf_spread_merge_stats",
                "tiled_pf_spread_merge_overlap_stats",
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
                "tiled_pf_spread_merge_stats",
                "tiled_pf_spread_merge_overlap_stats",
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
                "tiled_pf_spread_merge_stats",
                "tiled_pf_spread_merge_overlap_stats",
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
        (
            "l5",
            256,
            16,
            32,
            [
                "local",
                "ext",
                "tiled_pf_spread_merge_overlap_stats",
            ],
        ),
        (
            "l7",
            256,
            32,
            32,
            [
                "local",
                "ext",
                "tiled_pf_spread_merge_overlap_stats",
            ],
        ),
        (
            "l10",
            196,
            32,
            64,
            [
                "local",
                "ext",
                "tiled_pf_spread_merge_overlap_stats",
            ],
        ),
        (
            "l12",
            196,
            64,
            64,
            [
                "local",
                "ext",
                "tiled_pf_spread_merge_overlap_stats",
            ],
        ),
        (
            "l15",
            196,
            64,
            128,
            [
                "local",
                "ext",
                "tiled_pf_spread_merge_overlap_stats",
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

# Host tests for the copy interface (#11): beat width, descriptor flags, chain bookkeeping.
[
    cc_test(
        name = "dma_tile_test" + variant,
        srcs = ["dma_tile_test.cc"],
        copts = copts,
        deps = [
            "//sw/opt/litert-micro:dma_tile_memcpy",
            "@com_google_googletest//:gtest_main",
        ],
    )
    for variant, copts in [
        ("", []),
        (
            "_wait",
            ["-DCORALNPU_COPY_AT_WAIT=1"],
        ),
    ]
]

# Host tests for the pointwise tile planner (#11).
cc_test(
    name = "conv_tile_plan_test",
    srcs = ["conv_tile_plan_test.cc"],
    deps = [
        "//sw/opt/litert-micro:conv_tile_plan",
        "@com_google_googletest//:gtest_main",
    ],
)

# Host tests for the row planner of DMA-tiled spatial conv (#11).
cc_test(
    name = "conv_row_plan_test",
    srcs = ["conv_row_plan_test.cc"],
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

# Host copies of TFLite's int8 conv and fully connected reference kernels (#63).
py_library(
    name = "tflite_int8_ref",
    srcs = ["tflite_int8_ref.py"],
    imports = ["."],
)

# npusim (MPACT models Zvt) checks correctness quickly; the verilator target
# runs the same ELF on the 1 MB VmeCoreMiniHighmemAxi RTL for cycle counts.
# The verilator target compares against tflite_int8_ref instead of running the
# scalar reference on the core (#63); npusim runs both and checks they agree.
coralflow_test(
    name = "conv_vme_sim_test",
    # Verilator on mario takes about 250 s (#63).
    size = "large",
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
        ":tflite_int8_ref",
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
    # Verilator on mario takes about 1,100 s, mostly weight packing for the 1024x200 case (#63).
    size = "enormous",
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
        ":tflite_int8_ref",
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
        "//sw/utils:perf_counters",
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
        name = "ext_mem_tile_bench_%s_r4%s" % (variant, width_suffix),
        srcs = ["ext_mem_tile_bench.cc"],
        copts = ["-DTILE_ROWS=4"] + flags + width_flags,
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
    for width_suffix, width_flags in [
        ("", []),
        # 1-pixel tiles: the short smoke test (_w1) of the same variants.
        (
            "_w1",
            ["-DTILE_WIDTH=1"],
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
        "//sw/utils:perf_counters",
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
        "//sw/utils:perf_counters",
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

# #59: the same benches on cores with the memory stall counters. Without
# --//sw/utils:read_perf_counters the ELFs are unchanged and the cycle counts must
# equal the targets above; with it each run also prints PERF lines.
# mobilenet_engine_select_test_perf_rvv runs the full model on the RVV core
# without the matrix engine, for baseline profiles (MOBILENET_ENGINE_RUNS=all_vector).
[
    coralflow_test(
        name = name + suffix,
        srcs = [name + ".py"],
        elf = ":" + name + ".elf",
        hdl_toplevel = toplevel,
        highmem = True,
        main = name + ".py",
        tags = ["manual"],
        targets = ["verilator"],
        deps = ["@bazel_tools//tools/python/runfiles"],
    )
    for name, suffix, toplevel in [
        ("mobilenet_kernel_bench", "_perf", "RvvCoreMiniPerfHighmemAxi"),
        ("mobilenet_conv_bench", "_perf", "VmeCoreMiniPerfHighmemAxi"),
        ("mobilenet_engine_select_test", "_perf", "VmeCoreMiniPerfHighmemAxi"),
        ("mobilenet_engine_select_test", "_perf_rvv", "RvvCoreMiniPerfHighmemAxi"),
    ]
]

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
    hdl_toplevel = "RvvCoreMiniRequantAxi",
    main = "requant_iss_test.py",
    # verilator: the RTL with REQUANT_ON (#9, steps 5-6). uvm: the same RTL
    # co-simulated against Spike and MPACT.
    targets = [
        "npusim",
        "verilator",
        "uvm",
    ],
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
