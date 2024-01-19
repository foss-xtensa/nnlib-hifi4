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
#endif //HiFi_BUILD

#include "CpuOperationUtils.h"

#ifndef HIFI_BUILD
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#else
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#endif //HiFi_BUILD

namespace android {
namespace nn {

bool l2normFloat32(const float* inputData, const Shape& inputShape,
                   float* outputData, const Shape& outputShape) {
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    tflite::reference_ops::L2Normalization<tflite::FusedActivationFunctionType::kNone>(
            inputData, convertShapeToDims(inputShape),
            outputData, convertShapeToDims(outputShape));
#else
    int i, ret;
    int32_t batches, height, width, depth;
    batches = (int32_t)getSizeOfDimension(inputShape, 0);
    height  = (int32_t)getSizeOfDimension(inputShape, 1);
    width   = (int32_t)getSizeOfDimension(inputShape, 2);
    depth   = (int32_t)getSizeOfDimension(inputShape, 3);
    for(i = 0; i < batches*height*width; i++)
    {
        ret = xa_nn_l2_norm_f32(outputData + i*depth, inputData + i*depth, depth);
        if(ret)
            return false;
    }
#endif

    return true;
}

bool l2normQuant8(const uint8_t* inputData, const Shape& inputShape,
                  uint8_t* outputData, const Shape& outputShape) {
#ifndef HIFI_BUILD
    tflite::reference_ops::L2Normalization(
            inputData, convertShapeToDims(inputShape),
            inputShape.offset,
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::L2Normalization(
            inputData, convertShapeToDims(inputShape),
            inputShape.offset,
            outputData, convertShapeToDims(outputShape));
#endif //HiFi_BUILD

    return true;
}

bool localResponseNormFloat32(const float* inputData, const Shape& inputShape,
                              int32_t radius, float bias, float alpha, float beta,
                              float* outputData, const Shape& outputShape) {
#ifndef HIFI_BUILD
    tflite::reference_ops::LocalResponseNormalization(
            inputData, convertShapeToDims(inputShape),
            radius, bias, alpha, beta,
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::LocalResponseNormalization(
            inputData, convertShapeToDims(inputShape),
            radius, bias, alpha, beta,
            outputData, convertShapeToDims(outputShape));
#endif //HiFi_BUILD

    return true;
}
}  // namespace nn
}  // namespace android
