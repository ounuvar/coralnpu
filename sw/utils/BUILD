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
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
)

# #59: read the memory stall counters (sw/utils/perf_counters.h). Only for
# cores built with enablePerfCounters; off by default.
bool_flag(
    name = "perf_counters",
    build_setting_default = False,
)

config_setting(
    name = "perf_counters_enabled",
    flag_values = {":perf_counters": "True"},
)

cc_library(
    name = "perf_counters",
    hdrs = ["perf_counters.h"],
    defines = select({
        ":perf_counters_enabled": ["CORALNPU_PERF_COUNTERS=1"],
        "//conditions:default": [],
    }),
    target_compatible_with = ["//platforms/cpu:coralnpu_v2"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
)
