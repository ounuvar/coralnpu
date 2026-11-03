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

# The matrix engine runs on npusim (MPACT models Zvt); the RTL VME core has
# too little memory for these layer sizes.
coralflow_test(
    name = "conv_vme_sim_test",
    srcs = ["conv_vme_test.py"],
    elf = ":conv_vme_test.elf",
    highmem = True,
    main = "conv_vme_test.py",
    targets = ["npusim"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
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
    highmem = True,
    main = "fully_connected_vme_test.py",
    targets = ["npusim"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)
