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
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {

// This is the most generic TfLiteRegistration. The actual supported types may
// still be target dependent. The only requirement is that every implementation
// (reference or optimized) must define this function.
TfLiteRegistration Register_FULLY_CONNECTED();

#if defined(CMSIS_NN) || defined(ARDUINO)
// The Arduino is a special case where we use the CMSIS kernels, but because of
// the current approach to building for Arduino, we do not support -DCMSIS_NN as
// part of the build. As a result, we use defined(ARDUINO) as proxy for the
// CMSIS kernels for this one special case.

// Returns a TfLiteRegistration struct for cmsis-nn kernel variant that only
// supports int8.
TfLiteRegistration Register_FULLY_CONNECTED_INT8();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.

inline TfLiteRegistration Register_FULLY_CONNECTED_INT8() {
  return Register_FULLY_CONNECTED();
}

#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
