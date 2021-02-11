/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h"

#include "tensorflow/lite/experimental/microfrontend/lib/bits.h"

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut) {
  if (x <= 2) {
    return lut[x];
  }

  const int16_t interval = MostSignificantBit32(x);
  lut += 4 * interval - 6;

  const int16_t frac =
      ((interval < 11) ? (x << (11 - interval)) : (x >> (interval - 11))) &
      0x3FF;

  int32_t result = ((int32_t)lut[2] * frac) >> 5;
  result += (int32_t)((uint32_t)lut[1] << 5);
  result *= frac;
  result = (result + (1 << 14)) >> 15;
  result += lut[0];
  return (int16_t)result;
}

uint32_t PcanShrink(const uint32_t x) {
  if (x < (2 << kPcanSnrBits)) {
    return (x * x) >> (2 + 2 * kPcanSnrBits - kPcanOutputBits);
  } else {
    return (x >> (kPcanSnrBits - kPcanOutputBits)) - (1 << kPcanOutputBits);
  }
}

void PcanGainControlApply(struct PcanGainControlState* state,
                          uint32_t* signal) {
  int i;
  for (i = 0; i < state->num_channels; ++i) {
    const uint32_t gain =
        WideDynamicFunction(state->noise_estimate[i], state->gain_lut);
    const uint32_t snr = ((uint64_t)signal[i] * gain) >> state->snr_shift;
    signal[i] = PcanShrink(snr);
  }
}
