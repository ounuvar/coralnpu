# Copyright 2023 Google LLC

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "systemc",
    cache_entries = {
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_STANDARD": "17",
        "BUILD_SHARED_LIBS": "False",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    generate_args = [
        "-G Ninja",
    ],
    install = True,
    lib_source = ":all_srcs",
    # sc_cor_qt.cpp weak-references the ASan fiber hooks and skips them when
    # NULL. GNU ld leaves missing weak refs as NULL; Apple's ld64 needs -U.
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,-U,___sanitizer_start_switch_fiber",
            "-Wl,-U,___sanitizer_finish_switch_fiber",
        ],
        "//conditions:default": [],
    }),
    out_static_libs = ["libsystemc.a"],
    targets = ["systemc"],
    visibility = ["//visibility:public"],
)
