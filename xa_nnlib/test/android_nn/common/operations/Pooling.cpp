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

#define ANDROID_NN_POOLING_PARAMETERS                                           \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;

#define ANDROID_NN_POOLING_PARAMETERS_HIFI_BUILD                                \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2); 

bool averagePoolFloat32(const float* inputData, const Shape& inputShape,
                        int32_t padding_left, int32_t padding_right,
                        int32_t padding_top, int32_t padding_bottom,
                        int32_t stride_width, int32_t stride_height,
                        int32_t filter_width, int32_t filter_height, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                        float* outputData, const Shape& outputShape) {
#else
                        float* outputData, const Shape& outputShape, void* p_scratch) {
#endif

    
    float output_activation_min, output_activation_max;
    
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);


#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    ANDROID_NN_POOLING_PARAMETERS
    tflite::reference_ops::AveragePool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#else
//    uint32_t out_channels = getSizeOfDimension(outputShape, 3);                 
    ANDROID_NN_POOLING_PARAMETERS_HIFI_BUILD
    int input_channels = getSizeOfDimension(inputShape, 3);
    int err_f, itr;
    int batch_size = (int)getSizeOfDimension(outputShape, 0);
    const float *ptr_tmp_in;
    float *ptr_tmp_out;


    for(itr=0; itr<batch_size; itr++)
    {
       ptr_tmp_out = &outputData[outHeight*outWidth*input_channels*itr]; 
       ptr_tmp_in  = &inputData[height*width*input_channels*itr];

        err_f =  xa_nn_avgpool_f32(ptr_tmp_out,
                ptr_tmp_in,
                height,
                width,
                input_channels,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_top,
                outHeight,
                outWidth,
                0,
                0,
                p_scratch);
    }

    int total_elements = outHeight * outWidth * input_channels * batch_size;
    err_f = xa_nn_vec_activation_min_max_f32_f32(outputData,
                                         (const float *)outputData,
                                         output_activation_min,
                                         output_activation_max,
                                         total_elements
                                        );
    (void)err_f; /* Unused return error */
#endif

    return true;
}

bool averagePoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                       int32_t padding_left, int32_t padding_right,
                       int32_t padding_top, int32_t padding_bottom,
                       int32_t stride_width, int32_t stride_height,
                       int32_t filter_width, int32_t filter_height, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                       uint8_t* outputData, const Shape& outputShape) {
#else
                       uint8_t* outputData, const Shape& outputShape, void* p_scratch) {
#endif


    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_NNLIB_OPT
    ANDROID_NN_POOLING_PARAMETERS
    tflite::reference_ops::AveragePool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#else
    ANDROID_NN_POOLING_PARAMETERS_HIFI_BUILD
    int input_channels = getSizeOfDimension(inputShape, 3);
    int err_f, itr;
    int batch_size = (int)getSizeOfDimension(outputShape, 0);
    uint8_t *ptr_tmp_out;
    const uint8_t *ptr_tmp_in;  


    for(itr=0; itr<batch_size; itr++)
    {
       ptr_tmp_out = &outputData[outHeight*outWidth*input_channels*itr]; 
       ptr_tmp_in  = &inputData[height*width*input_channels*itr];

        err_f =  xa_nn_avgpool_asym8(ptr_tmp_out,
                ptr_tmp_in,
                height,
                width,
                input_channels,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_top,
                outHeight,
                outWidth,
                0,
                0,
                p_scratch);
    }
    
    int total_elements = outHeight * outWidth * input_channels * batch_size;
    err_f = xa_nn_vec_activation_min_max_asym8_asym8(outputData,
                                           (const uint8_t *)outputData,
                                           output_activation_min,
                                           output_activation_max,
                                           total_elements
                                          );
    (void)err_f; /* Unused return error */
#endif

    return true;
}

bool l2PoolFloat32(const float* inputData, const Shape& inputShape,
                   int32_t padding_left, int32_t padding_right,
                   int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   float* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_BUILD
    tflite::reference_ops::L2Pool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::L2Pool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#endif

    return true;
}

bool maxPoolFloat32(const float* inputData, const Shape& inputShape,
                    int32_t padding_left, int32_t padding_right,
                    int32_t padding_top, int32_t padding_bottom,
                    int32_t stride_width, int32_t stride_height,
                    int32_t filter_width, int32_t filter_height, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                    float* outputData, const Shape& outputShape) {
#else
                    float* outputData, const Shape& outputShape, void* p_scratch) {
#endif


    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    ANDROID_NN_POOLING_PARAMETERS
    tflite::reference_ops::MaxPool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#else
    ANDROID_NN_POOLING_PARAMETERS_HIFI_BUILD
    int input_channels = getSizeOfDimension(inputShape, 3);
    int err_f, itr;
    int batch_size = (int)getSizeOfDimension(outputShape, 0);
    const float *ptr_tmp_in;
    float *ptr_tmp_out;

    for(itr=0; itr<batch_size; itr++)
    {
       ptr_tmp_out = &outputData[outHeight*outWidth*input_channels*itr]; 
       ptr_tmp_in  = &inputData[height*width*input_channels*itr];

        err_f =  xa_nn_maxpool_f32(ptr_tmp_out,
                ptr_tmp_in,
                height,
                width,
                input_channels,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_top,
                outHeight,
                outWidth,
                0,
                0,
                p_scratch);
    }
    
    int total_elements = outHeight * outWidth * input_channels * batch_size;
    err_f = xa_nn_vec_activation_min_max_f32_f32(outputData,
                                         (const float *)outputData,
                                         output_activation_min,
                                         output_activation_max,
                                         total_elements
                                        );
    (void)err_f; /* Unused return error */
#endif

    return true;
}

bool maxPoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                   int32_t padding_left, int32_t padding_right,
                   int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
#ifndef HIFI_NNLIB_OPT
                   uint8_t* outputData, const Shape& outputShape) {
#else
                   uint8_t* outputData, const Shape& outputShape, void* p_scratch) {
#endif


    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_NNLIB_OPT
    ANDROID_NN_POOLING_PARAMETERS
    tflite::reference_ops::MaxPool(
            inputData, convertShapeToDims(inputShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            filter_width, filter_height,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));
#else
    ANDROID_NN_POOLING_PARAMETERS_HIFI_BUILD
    int input_channels = getSizeOfDimension(inputShape, 3);
    int err_f, itr;
    int batch_size = (int)getSizeOfDimension(outputShape, 0);
    uint8_t *ptr_tmp_out;
    const uint8_t *ptr_tmp_in;  

    for(itr=0; itr<batch_size; itr++)
    {
       ptr_tmp_out = &outputData[outHeight*outWidth*input_channels*itr]; 
       ptr_tmp_in  = &inputData[height*width*input_channels*itr];

        err_f =  xa_nn_maxpool_asym8(ptr_tmp_out,
                ptr_tmp_in,
                height,
                width,
                input_channels,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_top,
                outHeight,
                outWidth,
                0,
                0,
                p_scratch);
    }

    int total_elements = outHeight * outWidth * input_channels * batch_size;
    err_f = xa_nn_vec_activation_min_max_asym8_asym8(outputData,
                                           (const uint8_t *)outputData,
                                           output_activation_min,
                                           output_activation_max,
                                           total_elements
                                          );
    (void)err_f; /* Unused return error */
#endif

    return true;
}

#undef ANDROID_NN_POOLING_PARAMETERS
}  // namespace nn
}  // namespace android
