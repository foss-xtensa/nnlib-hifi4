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

#ifndef HIFI_BUILD
#include "Operations.h"
#else
#include "xa_nnlib_ann_api.h"
#endif //HIFI_BUILD

#include "CpuOperationUtils.h"

#ifndef HIFI_BUILD
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#else
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#endif //HiFi_BUILD

namespace android {
namespace nn {

bool concatenationFloat32(const std::vector<const float*>& inputDataPtrs,
                          const std::vector<Shape>& inputShapes, int32_t axis,
                          float* outputData, const Shape& outputShape) {
    int num_inputs = inputShapes.size();
    std::vector<tflite::Dims<4>*> inputDimsPtr(num_inputs);
    std::vector<tflite::Dims<4> > inputDims(num_inputs);
    for (int i=0; i<num_inputs; i++) {
        inputDims[i] = convertShapeToDims(inputShapes[i]);
        inputDimsPtr[i] = &inputDims[i];
    }

#ifndef HIFI_BUILD
    tflite::reference_ops::Concatenation<tflite::FusedActivationFunctionType::kNone, float>(
            getNumberOfDimensions(outputShape) - axis - 1,
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::Concatenation<tflite::FusedActivationFunctionType::kNone, float>(
            getNumberOfDimensions(outputShape) - axis - 1,
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,
            outputData, convertShapeToDims(outputShape));
#endif //HiFi_BUILD

    return true;
}

bool concatenationQuant8(const std::vector<const uint8_t*>& inputDataPtrs,
                         const std::vector<Shape>& inputShapes, int32_t axis,
                         uint8_t* outputData, const Shape& outputShape) {
    int num_inputs = inputShapes.size();
    std::vector<tflite::Dims<4>*> inputDimsPtr(num_inputs);
    std::vector<tflite::Dims<4> > inputDims(num_inputs);
    for (int i=0; i<num_inputs; i++) {
        inputDims[i] = convertShapeToDims(inputShapes[i]);
        inputDimsPtr[i] = &inputDims[i];
    }

#ifndef HIFI_BUILD
    tflite::reference_ops::Concatenation<tflite::FusedActivationFunctionType::kNone, uint8_t>(
            getNumberOfDimensions(outputShape) - axis - 1,
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::Concatenation<tflite::FusedActivationFunctionType::kNone, uint8_t>(
            getNumberOfDimensions(outputShape) - axis - 1,
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,
            outputData, convertShapeToDims(outputShape));
#endif //HiFi_BUILD

    return true;
}
}  // namespace nn
}  // namespace android
