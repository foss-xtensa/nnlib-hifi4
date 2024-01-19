/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H
#define ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H

#include "OperationsUtils.h"

#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace android {
namespace nn {

// The implementations in tflite/kernels/internal/ take a Dims<4> object
// even if the original tensors were not 4D.
inline tflite::Dims<4> convertShapeToDims(const Shape& shape) {
  nnAssert(shape.dimensions.size() <= 4);
  tflite::Dims<4> dims;

  // The dimensions are reversed in Dims<4>.
  for (int i = 0; i < 4; ++i) {
    int src = static_cast<int>(shape.dimensions.size()) - i - 1;
    if (src >= 0) {
      dims.sizes[i] = static_cast<int>(getSizeOfDimension(shape, src));
    } else {
      dims.sizes[i] = 1;
    }
  }

  dims.strides[0] = 1;
  for (int i = 1; i<4; i++) {
    dims.strides[i] = dims.strides[i-1] * dims.sizes[i-1];
  }
  return dims;
}

} // nn
} // android

#endif // ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H
