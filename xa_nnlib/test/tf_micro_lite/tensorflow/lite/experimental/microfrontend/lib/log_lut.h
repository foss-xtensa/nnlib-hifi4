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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Number of segments in the log lookup table. The table will be kLogSegments+1
// in length (with some padding).
#define kLogSegments 128
#define kLogSegmentsLog2 7

// Scale used by lookup table.
#define kLogScale 65536
#define kLogScaleLog2 16
#define kLogCoeff 45426

extern const uint16_t kLogLut[];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_
