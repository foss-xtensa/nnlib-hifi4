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

#ifndef HIFI_BUILD
// executionMutex is used to protect concurrent access of non-threadsafe resources
// like gemmlowp::GemmContext.
// std::mutex is safe for pthreads on Android.
static std::mutex executionMutex;
#endif //HIFI_BUILD

bool fullyConnectedFloat32(const float* inputData, const Shape& inputShape,
                           const float* weightsData, const Shape& weightsShape,
                           const float* biasData, const Shape& biasShape,
                           int32_t activation,
                           float* outputData, const Shape& outputShape) {
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    // b/80425683, optimized implementation produces incorrect results when the
    // number of input elements is the squre of batch_size.
    uint32_t batch_size = getSizeOfDimension(outputShape, 0);
    uint32_t input_n_elements = getNumberOfElements(inputShape);
    if (batch_size * batch_size == input_n_elements) {
        tflite::reference_ops::FullyConnected(
                inputData, convertShapeToDims(inputShape),
                weightsData, convertShapeToDims(weightsShape),
                biasData, convertShapeToDims(biasShape),
                output_activation_min, output_activation_max,
                outputData, convertShapeToDims(outputShape));
    } else {
        tflite::reference_ops::FullyConnected(
                inputData, convertShapeToDims(inputShape),
                weightsData, convertShapeToDims(weightsShape),
                biasData, convertShapeToDims(biasShape),
                output_activation_min, output_activation_max,
                outputData, convertShapeToDims(outputShape));
    }
    return true;
#else
    WORD32 i, batches, output_depth, weight_depth, out_dim_count, weight_dim_count;
    WORD32 ret, out_length;
    out_dim_count = outputShape.dimensions.size();
    batches = 1;
    for(i = out_dim_count-2; i >= 0; i--)
    {
        batches *= outputShape.dimensions[i];
    }
    output_depth = outputShape.dimensions[out_dim_count-1];
    weight_dim_count = weightsShape.dimensions.size();
    weight_depth = weightsShape.dimensions[weight_dim_count-1];

    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_fully_connected_f32(
            &outputData[i*output_depth],
            weightsData,
            &inputData[i*weight_depth],
            biasData,
            weight_depth,
            output_depth);
        if(ret != 0)
            return false;
    }
    out_length = batches*output_depth;
    for(i = 0; i < out_length; i++)
    {
        outputData[i] = outputData[i]<output_activation_min?output_activation_min:outputData[i];
        outputData[i] = outputData[i]>output_activation_max?output_activation_max:outputData[i];
    }
    return true;
#endif
}

bool fullyConnectedQuant8(const uint8_t* inputData, const Shape& inputShape,
                          const uint8_t* weightsData, const Shape& weightsShape,
                          const int32_t* biasData, const Shape& biasShape,
                          int32_t activation,
                          uint8_t* outputData, const Shape& outputShape) {
    int32_t inputOffset = -inputShape.offset;
    int32_t weightsOffset = -weightsShape.offset;
    int32_t outputOffset = outputShape.offset;

    float real_multiplier = 0.0;
    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    if (!GetQuantizedConvolutionMultipler(inputShape, weightsShape, biasShape,
                                          outputShape, &real_multiplier) ||
            !QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier,
                                              &output_shift)) {
        return false;
    }
    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_BUILD
    static gemmlowp::GemmContext gemm_context;

    // Prevent concurrent executions that access gemm_context.
    std::unique_lock<std::mutex> lock(executionMutex);
    // Alow gemmlowp automatically decide how many threads to use.
    gemm_context.set_max_num_threads(0);
#endif //HIFI_BUILD

#ifndef HIFI_NNLIB_OPT
    static gemmlowp::GemmContext gemm_context;
    tflite::reference_ops::FullyConnected(
            inputData, convertShapeToDims(inputShape), inputOffset,
            weightsData, convertShapeToDims(weightsShape), weightsOffset,
            biasData, convertShapeToDims(biasShape),
            outputOffset, output_multiplier, output_shift,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape), &gemm_context);
#else
    WORD32 i, batches, output_depth, weight_depth, out_dim_count, weight_dim_count;
    WORD32 ret, out_length;
    out_dim_count = outputShape.dimensions.size();
    batches = 1;
    for(i = out_dim_count-2; i >= 0; i--)
    {
        batches *= outputShape.dimensions[i];
    }
    output_depth = outputShape.dimensions[out_dim_count-1];
    weight_dim_count = weightsShape.dimensions.size();
    weight_depth = weightsShape.dimensions[weight_dim_count-1];

    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_fully_connected_asym8xasym8_asym8(
            &outputData[i*output_depth],
            weightsData,
            &inputData[i*weight_depth],
            biasData,
            weight_depth,
            output_depth,
            inputOffset,
            weightsOffset,
            output_multiplier,
            -output_shift,
            outputOffset);
        if(ret != 0)
            return false;
    }
    out_length = batches*output_depth;
    for(i = 0; i < out_length; i++)
    {
        outputData[i] = outputData[i]<output_activation_min?output_activation_min:outputData[i];
        outputData[i] = outputData[i]>output_activation_max?output_activation_max:outputData[i];
    }
    return true;
#endif

    return true;
}
}  // namespace nn
}  // namespace android
