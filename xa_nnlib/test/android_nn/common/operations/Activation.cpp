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

bool reluFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
#if HIFI_VFPU && defined HIFI_NNLIB_OPT 
#if 0
    xa_nn_vec_relu_f32_f32(outputData, inputData, INFINITY, numElements);
#else
    xa_nn_vec_activation_min_max_f32_f32(
                                    outputData,
                                    inputData,
                                    0.f,
                                    INFINITY,
                                    numElements
                                );
#endif
#else
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::max(0.f, *inputData);
    }
#endif
    return true;
}

bool relu1Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
#if HIFI_VFPU && defined HIFI_NNLIB_OPT 
    xa_nn_vec_activation_min_max_f32_f32(
                                    outputData,
                                    inputData,
                                    -1.f,
                                    1.f,
                                    numElements
                                );
#else
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::min(std::max(-1.f, *inputData), 1.f);
    }
#endif
    return true;
}

bool relu6Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
#if HIFI_VFPU && defined HIFI_NNLIB_OPT 
#if 0
    xa_nn_vec_relu6_f32_f32(outputData, inputData, numElements);
#else
    xa_nn_vec_activation_min_max_f32_f32(
                                    outputData,
                                    inputData,
                                    0.f,
                                    6.f,
                                    numElements
                                );
#endif
#else
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::min(std::max(0.f, *inputData), 6.f);
    }
#endif
    return true;
}

bool tanhFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::tanh(*inputData);
    }
#else
    int err;
    err = xa_nn_vec_tanh_f32_f32(outputData, inputData, numElements); 
    (void)err; /* Unused return value */
#endif
    return true;
}

bool logisticFloat32(const float* inputData, const Shape& inputShape,
                     float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = 1.f / (1.f + std::exp(-*inputData));
    }
#else
    int err;
    err = xa_nn_vec_sigmoid_f32_f32(outputData, inputData, numElements);
    (void)err; /* Unused return value */
#endif
    return true;
}

bool softmaxFloat32(const float* inputData, const Shape& inputShape,
                    const float beta,
                    float* outputData, const Shape& outputShape) {
    tflite::Dims<4> dim;
    if (getNumberOfDimensions(inputShape) == 2) {
        uint32_t batch_size = getSizeOfDimension(inputShape, 0);
        uint32_t input_size = getNumberOfElements(inputShape) / batch_size;

        Shape shapeIn4D;
#ifndef HIFI_BUILD
        shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
#else
        shapeIn4D.dimensions.push_back(batch_size);
        shapeIn4D.dimensions.push_back(1);
        shapeIn4D.dimensions.push_back(1);
        shapeIn4D.dimensions.push_back(input_size);
#endif
        dim = convertShapeToDims(shapeIn4D);
    } else if (getNumberOfDimensions(inputShape) == 4) {
        dim = convertShapeToDims(inputShape);
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "only 2D and 4D tensors supported";
#endif //HIFI_BUILD
        return false;
    }

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    tflite::reference_ops::Softmax(inputData, dim, beta,
                                   outputData, dim);
#else
    {
        int err;
        float *p_in_tmp = (float *)inputData;
        const int batches = MatchingArraySize(dim, 3, dim, 3);
        const int height  = MatchingArraySize(dim, 2, dim, 2);
        const int width   = MatchingArraySize(dim, 1, dim, 1);
        const int depth   = MatchingArraySize(dim, 0, dim, 0);

        if(beta != 1)
        {
            for (int b = 0; b < batches*height*width*depth; ++b) 
            {
                p_in_tmp[b] = p_in_tmp[b]*beta; 
            }
        }

        for (int b = 0; b < batches; ++b) 
        {
            for (int x = 0; x < width; ++x) 
            {
                for (int y = 0; y < height; ++y) 
                {
                    int offset;

                    offset = Offset(dim, 0, x, y, b);                    

                    err = xa_nn_vec_softmax_f32_f32(&outputData[offset], 
                            &inputData[offset], 
                            depth);
                }
            }
        }
        (void)err; /* Unused return value */
    }
#endif
    return true;
}

#define ANDROID_NN_RELUX_QUANT8(activation)                             \
    int numElements = getNumberOfElements(inputShape);                  \
    int32_t output_activation_min = 0;                                  \
    int32_t output_activation_max = 0;                                  \
                                                                        \
    CalculateActivationRangeUint8(activation, inputShape,               \
                                  &output_activation_min,               \
                                  &output_activation_max);              \
                                                                        \
    for (int i=0; i<numElements; i++, inputData++, outputData++) {      \
        *outputData = std::min((uint8_t)output_activation_max,          \
                std::max((uint8_t)output_activation_min, *inputData));  \
    }


bool reluQuant8(const uint8_t* inputData, const Shape& inputShape,
                uint8_t* outputData, const Shape& outputShape) {
#ifdef HIFI_NNLIB_OPT
    int numElements = getNumberOfElements(inputShape);                  
    int32_t output_activation_min = 0;                                  
    int32_t output_activation_max = 0;                                  
    
    CalculateActivationRangeUint8(kActivationRelu, inputShape,               
                                  &output_activation_min,               
                                  &output_activation_max);              
                                                                        
    xa_nn_vec_activation_min_max_asym8_asym8(outputData, 
                                   inputData, 
                                   output_activation_min,
                                   output_activation_max,
                                   numElements);
#else
    ANDROID_NN_RELUX_QUANT8(kActivationRelu)
#endif
    return true;
}

bool relu1Quant8(const uint8_t* inputData, const Shape& inputShape,
                 uint8_t* outputData, const Shape& outputShape) {
#ifdef HIFI_NNLIB_OPT
    int numElements = getNumberOfElements(inputShape);                  
    int32_t output_activation_min = 0;                                  
    int32_t output_activation_max = 0;                                  
    
    CalculateActivationRangeUint8(kActivationRelu1, inputShape,               
                                  &output_activation_min,               
                                  &output_activation_max);              
                                                                        
    xa_nn_vec_activation_min_max_asym8_asym8(outputData, 
                                   inputData, 
                                   output_activation_min,
                                   output_activation_max,
                                   numElements);
#else
    ANDROID_NN_RELUX_QUANT8(kActivationRelu1)
#endif
    return true;
}

bool relu6Quant8(const uint8_t* inputData, const Shape& inputShape,
                 uint8_t* outputData, const Shape& outputShape) {
#ifdef HIFI_NNLIB_OPT
    int numElements = getNumberOfElements(inputShape);                  
    int32_t output_activation_min = 0;                                  
    int32_t output_activation_max = 0;                                  
    
    CalculateActivationRangeUint8(kActivationRelu6, inputShape,               
                                  &output_activation_min,               
                                  &output_activation_max);              
                                                                        
    xa_nn_vec_activation_min_max_asym8_asym8(outputData, 
                                   inputData, 
                                   output_activation_min,
                                   output_activation_max,
                                   numElements);
#else
    ANDROID_NN_RELUX_QUANT8(kActivationRelu6)
#endif
    return true;
}

#undef ANDROID_NN_RELUX_QUANT8

bool logisticQuant8(const uint8_t* inputData, const Shape& inputShape,
                    uint8_t* outputData, const Shape& outputShape) {
    if (outputShape.offset != 0 || outputShape.scale != 1.f / 256) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "incorrect scale / offset for output";
#endif //HIFI_BUILD
        return false;
    }

    int numElements = getNumberOfElements(inputShape);
    static constexpr int kInputIntegerBits = 4;

    const double input_real_multiplier =
            inputShape.scale *
            static_cast<double>(1 << (31 - kInputIntegerBits));

    int32_t input_multiplier = 0;
    int32_t input_left_shift = 0;
    if (!QuantizeMultiplierGreaterThanOne(input_real_multiplier,
                                          &input_multiplier,
                                          &input_left_shift)) {
        return false;
    }
    int32_t input_range_radius =
            CalculateInputRadius(kInputIntegerBits, input_left_shift);
#ifndef HIFI_NNLIB_OPT
    tflite::reference_ops::Logistic(
            inputData, convertShapeToDims(inputShape),
            inputShape.offset, input_range_radius,
            input_multiplier, input_left_shift,
            outputData, convertShapeToDims(outputShape));
#else
    int err;
    const int size = RequiredBufferSizeForDims(convertShapeToDims(inputShape));


    err = xa_nn_vec_sigmoid_asym8_asym8(outputData, 
                               inputData, 
                               inputShape.offset, 
                               input_range_radius, 
                               input_multiplier,
                               input_left_shift, 
                               size);
    (void)err; /* Unused return value */
#endif

#ifndef HIFI_WARNINGS
    (void)numElements;
#endif
    return true;
}

bool softmaxQuant8(const uint8_t* inputData, const Shape& inputShape,
                   const float beta,
#ifndef HIFI_NNLIB_OPT
                   uint8_t* outputData, const Shape& outputShape) {
#else                   
                   uint8_t* outputData, const Shape& outputShape, void *p_scratch) {
#endif
    tflite::Dims<4> dim;
    if (getNumberOfDimensions(inputShape) == 2) {
        uint32_t batch_size = getSizeOfDimension(inputShape, 0);
        uint32_t input_size = getNumberOfElements(inputShape) / batch_size;

        Shape shapeIn4D;
#ifndef HIFI_BUILD
        shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
#else
        shapeIn4D.dimensions.push_back(batch_size);
        shapeIn4D.dimensions.push_back(1);
        shapeIn4D.dimensions.push_back(1);
        shapeIn4D.dimensions.push_back(input_size);
#endif
        dim = convertShapeToDims(shapeIn4D);
    } else if (getNumberOfDimensions(inputShape) == 4) {
        dim = convertShapeToDims(inputShape);
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "only 2D and 4D tensors supported";
#endif //HIFI_BUILD
        return false;
    }

    if (outputShape.offset != 0 || outputShape.scale != 1.f / 256) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "incorrect scale / offset for output";
#endif //HIFI_BUILD
        return false;
    }

    static const int32_t kScaledDiffIntegerBits = 5;
    const double input_beta_real_multiplier = std::min(
            1.0 * beta * inputShape.scale * (1 << (31 - kScaledDiffIntegerBits)),
            (1ll << 31) - 1.0);

    int32_t input_multiplier = 0;
    int32_t input_left_shift = 0;
    if (!QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                          &input_multiplier,
                                          &input_left_shift)) {
        return false;
    }
    float diff_min = -1.0f * CalculateInputRadius(kScaledDiffIntegerBits,
                                                  input_left_shift);

#ifndef HIFI_NNLIB_OPT
    tflite::reference_ops::Softmax(inputData, dim, input_multiplier,
                                   input_left_shift, diff_min,
                                   outputData, dim);
#else
    {
        int err;
        const int batches = MatchingArraySize(dim, 3, dim, 3);
        const int height  = MatchingArraySize(dim, 2, dim, 2);
        const int width   = MatchingArraySize(dim, 1, dim, 1);
        const int depth   = MatchingArraySize(dim, 0, dim, 0);

        for (int b = 0; b < batches; ++b) 
        {
            for (int x = 0; x < width; ++x) 
            {
                for (int y = 0; y < height; ++y) 
                {
                    int offset;

                    offset = Offset(dim, 0, x, y, b);                    
                    err = xa_nn_vec_softmax_asym8_asym8(&outputData[offset], 
                            &inputData[offset], 
                            diff_min,
                            input_left_shift,
                            input_multiplier,                
                            depth,
                            p_scratch);
                }
            }
        }
        (void)err; /* Unused return value */
    }
#endif
    return true;
}


}  // namespace nn
}  // namespace android
