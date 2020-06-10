/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "tensorflow/lite/experimental/microfrontend/lib/fft.h"

#include <string.h>

#define FIXED_POINT 16
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"

void FftCompute(struct FftState* state, const int16_t* input,
                int input_scale_shift) {
  const size_t input_size = state->input_size;
  const size_t fft_size = state->fft_size;

  int16_t* fft_input = state->input;
  // First, scale the input by the given shift.
  int i;
  for (i = 0; i < input_size; ++i) {
    fft_input[i] = static_cast<int16_t>(static_cast<uint16_t>(input[i])
                                        << input_scale_shift);
  }
  // Zero out whatever else remains in the top part of the input.
  for (; i < fft_size; ++i) {
    fft_input[i] = 0;
  }

  // Apply the FFT.
  kiss_fftr(
      reinterpret_cast<const kiss_fftr_cfg>(state->scratch),
      state->input,
      reinterpret_cast<kiss_fft_cpx*>(state->output));
}

void FftInit(struct FftState* state) {
  // All the initialization is done in FftPopulateState()
}

void FftReset(struct FftState* state) {
  memset(state->input, 0, state->fft_size * sizeof(*state->input));
  memset(state->output, 0, (state->fft_size / 2 + 1) * sizeof(*state->output));
}
