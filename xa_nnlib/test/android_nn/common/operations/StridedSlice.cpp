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
 * Copyright (C) 2018 The Android Open Source Project
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

// Contains the implementation of the operations.

#define LOG_TAG "Operations"

#ifndef HIFI_BUILD
#include "Operations.h"
#else
#include "xa_nnlib_ann_api.h"
#endif //HiFi_BUILD

#include "CpuOperationUtils.h"

#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"

namespace android {
namespace nn {

bool stridedSliceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* beginData, const int32_t* endData,
                         const int32_t* stridesData,
                         int32_t beginMask, int32_t endMask, int32_t shrinkAxisMask,
                         uint8_t* outputData, const Shape& outputShape) {
    // This Op only supports 1-4D cases and since we use the reference 4D
    // implementation, the 1-3D tensors are mapped to 4D.
    const int kMaxDim = 4;

    std::vector<int> starts;
    std::vector<int> stops;
    std::vector<int> strides;

    int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(inputShape));
    for (int32_t idx = numInputDims - 1; idx >= 0; --idx) {
      int32_t dim = static_cast<int32_t>(getSizeOfDimension(inputShape, idx));
      int32_t stride = stridesData[idx];
      // stride value has to be non-zero
      NN_OPS_CHECK(stride != 0);
      bool positiveStride = stride > 0;

      int32_t begin = beginMask & (1 << idx)
              ? positiveStride ? 0 : dim - 1
              : ClampedIndex(beginData[idx], dim, positiveStride);
      int32_t end = endMask & (1 << idx)
              ? positiveStride ? dim : -1
              : ClampedIndex(endData[idx], dim, positiveStride);

      starts.emplace_back(begin);
      stops.emplace_back(end);
      strides.emplace_back(stride);
    }

    for (int i = numInputDims; i < kMaxDim; i++) {
      starts.emplace_back(0);
      stops.emplace_back(1);
      strides.emplace_back(1);
    }

    beginMask = ReverseMaskBits(beginMask, numInputDims);
    endMask = ReverseMaskBits(endMask, numInputDims);
    shrinkAxisMask = ReverseMaskBits(shrinkAxisMask, numInputDims);

    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::reference_ops::StridedSlice(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                beginMask, endMask, shrinkAxisMask,
                starts, stops, strides,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::reference_ops::StridedSlice(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                beginMask, endMask, shrinkAxisMask,
                starts, stops, strides,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }

    return true;
}

} // namespace nn
} // namespace android
