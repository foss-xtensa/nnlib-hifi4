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
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h"

#include <stdio.h>

void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config) {
  config->smoothing_bits = 10;
  config->even_smoothing = 0.025;
  config->odd_smoothing = 0.06;
  config->min_signal_remaining = 0.05;
}

int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels) {
  state->smoothing_bits = config->smoothing_bits;
  state->odd_smoothing = config->odd_smoothing * (1 << kNoiseReductionBits);
  state->even_smoothing = config->even_smoothing * (1 << kNoiseReductionBits);
  state->min_signal_remaining =
      config->min_signal_remaining * (1 << kNoiseReductionBits);
  state->num_channels = num_channels;
  state->estimate = calloc(state->num_channels, sizeof(*state->estimate));
  if (state->estimate == NULL) {
    fprintf(stderr, "Failed to alloc estimate buffer\n");
    return 0;
  }
  return 1;
}

void NoiseReductionFreeStateContents(struct NoiseReductionState* state) {
  free(state->estimate);
}
