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

#include "tensorflow/contrib/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/depthwiseconv_uint8.h"

namespace android {
namespace nn {

#define ANDROID_NN_DEPTHWISE_CONV_PARAMETERS                                    \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;

bool depthwiseConvFloat32(const float* inputData, const Shape& inputShape,
                          const float* filterData, const Shape& filterShape,
                          const float* biasData, const Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                          float* outputData, const Shape& outputShape) {
#else                      
                          float* outputData, const Shape& outputShape, void* p_scratch) {
#endif

    ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    (void) height;
    (void) width;
    (void) outWidth;
    (void) outHeight;
    (void) filterWidth;
    (void) filterHeight;
    tflite::reference_ops::DepthwiseConv(
            inputData, convertShapeToDims(inputShape),
            filterData, convertShapeToDims(filterShape),
            biasData, convertShapeToDims(biasShape),
            stride_width, stride_height,
            paddingWidth, paddingHeight, depth_multiplier,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));

    return true;
#else
    int32_t ret, batches = (int)getSizeOfDimension(outputShape, 0), i;
    int32_t outDepth = (int)getSizeOfDimension(outputShape, 3);
    int32_t inDepth = (int)getSizeOfDimension(inputShape, 3);
    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_conv2d_depthwise_f32(&outputData[i*outHeight*outWidth*outDepth],
                                         filterData,
                                         &inputData[i*height*width*inDepth],
                                         biasData,
                                         height,
                                         width,
                                         inDepth,
                                         filterHeight,
                                         filterWidth,
                                         depth_multiplier,
                                         stride_width,
                                         stride_height,
                                         paddingWidth,
                                         paddingHeight,
                                         outHeight,
                                         outWidth,
                                         0,
                                         0,
                                         p_scratch);
        if(ret != 0)
            return false;
    }

    int out_length;
    out_length = batches*outHeight*outWidth*outDepth;
    for(i = 0; i < out_length; i++)
    {
        outputData[i] = outputData[i]<output_activation_min?output_activation_min:outputData[i];
        outputData[i] = outputData[i]>output_activation_max?output_activation_max:outputData[i];
    }
    return true;
#endif
}


bool depthwiseConvQuant8(const uint8_t* inputData, const Shape& inputShape,
                         const uint8_t* filterData, const Shape& filterShape,
                         const int32_t* biasData, const Shape& biasShape,
                         int32_t padding_left, int32_t padding_right,
                         int32_t padding_top, int32_t padding_bottom,
                         int32_t stride_width, int32_t stride_height,
                         int32_t depth_multiplier, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                         uint8_t* outputData, const Shape& outputShape) {
#else
                         uint8_t* outputData, const Shape& outputShape, void *p_scratch) {
#endif

    ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

    float real_multiplier = 0.0;
    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;


    if (!GetQuantizedConvolutionMultipler(inputShape, filterShape, biasShape,
                                          outputShape, &real_multiplier) ||
            !QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier,
                                              &output_shift)) {
        return false;
    }
    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

    uint32_t inputOffset = -inputShape.offset;
    uint32_t filterOffset = -filterShape.offset;
    uint32_t outputOffset = outputShape.offset;

#ifndef HIFI_NNLIB_OPT
    tflite::reference_ops::DepthwiseConv(
            inputData, convertShapeToDims(inputShape), inputOffset,
            filterData, convertShapeToDims(filterShape), filterOffset,
            biasData, convertShapeToDims(biasShape),
            stride_width, stride_height,
            paddingWidth, paddingHeight, depth_multiplier,
            outputOffset, output_multiplier, output_shift,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));

    return true;
#else
    int32_t ret, batches = (int)getSizeOfDimension(outputShape, 0), i;
    int32_t outDepth = (int)getSizeOfDimension(outputShape, 3);
    int32_t inDepth = (int)getSizeOfDimension(inputShape, 3);
    /* output_shift is negated because it is always right shift in the
    tensorflow version used, this may need to be modified when switching
    to newer version of tensorflow/ANN */
    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_conv2d_depthwise_asym8xasym8(&outputData[i*outHeight*outWidth*outDepth],
                                                 filterData,
                                                 &inputData[i*height*width*inDepth],
                                                 biasData,
                                                 height,
                                                 width,
                                                 inDepth,
                                                 filterHeight,
                                                 filterWidth,
                                                 depth_multiplier,
                                                 stride_width,
                                                 stride_height,
                                                 paddingWidth,
                                                 paddingHeight,
                                                 outHeight,
                                                 outWidth,
                                                 inputOffset,
                                                 filterOffset,
                                                 output_multiplier,
                                                 -output_shift,
                                                 outputOffset,
                                                 0,
                                                 0,
                                                 p_scratch);
        if(ret != 0)
            return false;
    }

    int out_length;
    out_length = batches*outHeight*outWidth*outDepth;
    for(i = 0; i < out_length; i++)
    {
        outputData[i] = outputData[i]<output_activation_min?output_activation_min:outputData[i];
        outputData[i] = outputData[i]>output_activation_max?output_activation_max:outputData[i];
    }
    return true;
#endif
}

#undef ANDROID_NN_DEPTHWISE_CONV_PARAMETERS
}  // namespace nn
}  // namespace android
