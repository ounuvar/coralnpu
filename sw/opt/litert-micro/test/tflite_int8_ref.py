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
"""Host copies of TFLite's int8 reference kernels (#63).

Same integer math as tflite::reference_integer_ops::ConvPerChannel and
FullyConnected with MultiplyByQuantizedMultiplier (rounding-doubling high
multiply, then rounding right shift that rounds ties away from zero). The RTL
targets of the matrix-engine tests compare against these instead of running
the scalar reference on the simulated core; the npusim targets still run the C
reference and check that these agree with it.
"""

import numpy as np

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _wrap_int32(x):
    return ((x.astype(np.int64) - _INT32_MIN) % (1 << 32)) + _INT32_MIN


def saturating_rounding_doubling_high_mul(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    n = ab + nudge
    # C++ integer division truncates toward zero.
    q = np.where(n >= 0, n >> 31, -((-n) >> 31))
    overflow = (a == _INT32_MIN) & (b == _INT32_MIN)
    return np.where(overflow, _INT32_MAX, q)


def rounding_divide_by_pot(x, exponent):
    x = np.asarray(x, dtype=np.int64)
    exponent = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> exponent) + (remainder > threshold)


def multiply_by_quantized_multiplier(x, multiplier, shift):
    shift = np.asarray(shift, dtype=np.int64)
    left_shift = np.maximum(shift, 0)
    right_shift = np.maximum(-shift, 0)
    x = _wrap_int32(np.asarray(x, dtype=np.int64) << left_shift)
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x, multiplier), right_shift
    )


def conv_per_channel(
    input_data, filter_data, bias, multiplier, shift, output_shape, stride,
    dilation, padding, input_offset, output_offset, act_min, act_max
):
    """input_data NHWC int8, filter_data [out, fh, fw, in] int8."""
    batches, in_h, in_w, _ = input_data.shape
    out_depth, filt_h, filt_w, _ = filter_data.shape
    _, out_h, out_w, _ = output_shape
    stride_h, stride_w = stride
    dil_h, dil_w = dilation
    pad_h, pad_w = padding
    x = input_data.astype(np.int64) + input_offset
    acc = np.zeros((batches, out_h, out_w, out_depth), dtype=np.int64)
    oy = np.arange(out_h) * stride_h - pad_h
    ox = np.arange(out_w) * stride_w - pad_w
    for fy in range(filt_h):
        iy = oy + fy * dil_h
        valid_y = (iy >= 0) & (iy < in_h)
        for fx in range(filt_w):
            ix = ox + fx * dil_w
            valid_x = (ix >= 0) & (ix < in_w)
            patch = x[:, np.clip(iy, 0, in_h - 1)][:, :,
                                                   np.clip(ix, 0, in_w - 1)]
            # Positions outside the image are skipped, as in TFLite.
            patch = patch * (
                valid_y[None, :, None, None] & valid_x[None, None, :, None]
            )
            acc += patch @ filter_data[:, fy, fx, :].astype(np.int64).T
    acc += np.asarray(bias, dtype=np.int64)
    acc = multiply_by_quantized_multiplier(acc, multiplier, shift)
    return np.clip(acc + output_offset, act_min, act_max).astype(np.int8)


def fully_connected(
    input_data, filter_data, bias, multiplier, shift, input_offset,
    output_offset, act_min, act_max
):
    """input_data [batches, depth], filter_data [out, depth] int8; weights_offset 0."""
    acc = (input_data.astype(np.int64) +
           input_offset) @ filter_data.astype(np.int64).T
    acc += np.asarray(bias, dtype=np.int64)
    acc = multiply_by_quantized_multiplier(acc, multiplier, shift)
    return np.clip(acc + output_offset, act_min, act_max).astype(np.int8)
