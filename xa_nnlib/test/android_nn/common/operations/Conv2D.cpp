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

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#endif

namespace android {
namespace nn {

// If possible we will use this static buffer for the tensor.
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
static constexpr size_t kStaticBufferSize = 1605632;
static char static_scratch_buffer[kStaticBufferSize];
#endif

#ifndef HIFI_BUILD
// executionMutex is used to protect concurrent access of the static_scratch_buffer
// and other non-threadsafe resources like gemmlowp::GemmContext.
// std::mutex is safe for pthreads on Android.
static std::mutex executionMutex;
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
#define ANDROID_NN_CONV_PARAMETERS(Type)                                        \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
    uint32_t inDepth      = getSizeOfDimension(inputShape, 3);                  \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;                             \
                                                                                \
    tflite::Dims<4> im2colDim;                                                  \
    im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);               \
    im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);               \
    im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);               \
    im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;             \
                                                                                \
    im2colDim.strides[0] = 1;                                                   \
    for (int i=1; i<4; i++) {                                                   \
        im2colDim.strides[i] = im2colDim.strides[i-1] * im2colDim.sizes[i-1];   \
    }                                                                           \
                                                                                \
    Type* im2colData = nullptr;                                                 \
    uint64_t im2colByteSize = sizeof(Type);                                     \
    std::unique_ptr<Type[]> im2colGuard;                                        \
    for (int i=0; i<4; i++) {                                                   \
        im2colByteSize *= im2colDim.sizes[i];                                   \
    }                                                                           \
    /* http://b/77982879, tflite::optimized_ops::Conv uses int for offsets */   \
    if (im2colByteSize >= 0x7fffffff)  {                                        \
        LOG(ERROR) << "Conv size is too large, not enough memory";              \
        return false;                                                           \
    }                                                                           \
    if (im2colByteSize <= kStaticBufferSize) {                                  \
        im2colData = reinterpret_cast<Type *>(static_scratch_buffer);           \
    } else {                                                                    \
        im2colData = new (std::nothrow) Type[im2colByteSize / sizeof(Type)];    \
        if (im2colData == nullptr) {                                            \
            LOG(ERROR) << "Conv size is too large, not enough memory";          \
            return false;                                                       \
        }                                                                       \
        im2colGuard.reset(im2colData);                                          \
    }
#else /* without LOG prints */
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
#define ANDROID_NN_CONV_PARAMETERS(Type)                                        \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
    uint32_t inDepth      = getSizeOfDimension(inputShape, 3);                  \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;                             \
                                                                                \
    tflite::Dims<4> im2colDim;                                                  \
    im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);               \
    im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);               \
    im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);               \
    im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;             \
                                                                                \
    im2colDim.strides[0] = 1;                                                   \
    for (int i=1; i<4; i++) {                                                   \
        im2colDim.strides[i] = im2colDim.strides[i-1] * im2colDim.sizes[i-1];   \
    }                                                                           \
                                                                                \
    Type* im2colData = nullptr;                                                 \
    uint64_t im2colByteSize = sizeof(Type);                                     \
    std::unique_ptr<Type[]> im2colGuard;                                        \
    for (int i=0; i<4; i++) {                                                   \
        im2colByteSize *= im2colDim.sizes[i];                                   \
    }                                                                           \
    /* http://b/77982879, tflite::optimized_ops::Conv uses int for offsets */   \
    if (im2colByteSize >= 0x7fffffff)  {                                        \
        return false;                                                           \
    }                                                                           \
    if (im2colByteSize <= kStaticBufferSize) {                                  \
        im2colData = reinterpret_cast<Type *>(static_scratch_buffer);           \
    } else {                                                                    \
        im2colData = (Type *)malloc(im2colByteSize);                            \
        if (im2colData == nullptr) {                                            \
            return false;                                                       \
        }                                                                       \
        im2colGuard.reset(im2colData);                                          \
    }
#else // HIFI_NNLIB_OPT
#define ANDROID_NN_CONV_PARAMETERS(Type)                                        \
    int32_t height       = (int32_t)getSizeOfDimension(inputShape, 1);          \
    int32_t width        = (int32_t)getSizeOfDimension(inputShape, 2);          \
    int32_t filterHeight = (int32_t)getSizeOfDimension(filterShape, 1);         \
    int32_t filterWidth  = (int32_t)getSizeOfDimension(filterShape, 2);         \
    int32_t outHeight    = (int32_t)getSizeOfDimension(outputShape, 1);         \
    int32_t outWidth     = (int32_t)getSizeOfDimension(outputShape, 2);         \
    int32_t inDepth      = (int32_t)getSizeOfDimension(inputShape, 3);          
#endif // HIFI_NNLIB_OPT

#ifndef HIFI_NNLIB_OPT
#define ANDROID_NN_CONV_PARAMETERS_QUANT8(Type)                                        \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
    uint32_t inDepth      = getSizeOfDimension(inputShape, 3);                  \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;                             \
                                                                                \
    tflite::Dims<4> im2colDim;                                                  \
    im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);               \
    im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);               \
    im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);               \
    im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;             \
                                                                                \
    im2colDim.strides[0] = 1;                                                   \
    for (int i=1; i<4; i++) {                                                   \
        im2colDim.strides[i] = im2colDim.strides[i-1] * im2colDim.sizes[i-1];   \
    }                                                                           \
                                                                                \
    Type* im2colData = nullptr;                                                 \
    uint64_t im2colByteSize = sizeof(Type);                                     \
    std::unique_ptr<Type[]> im2colGuard;                                        \
    for (int i=0; i<4; i++) {                                                   \
        im2colByteSize *= im2colDim.sizes[i];                                   \
    }                                                                           \
    /* http://b/77982879, tflite::optimized_ops::Conv uses int for offsets */   \
    if (im2colByteSize >= 0x7fffffff)  {                                        \
        return false;                                                           \
    }                                                                           \
    if (im2colByteSize <= kStaticBufferSize) {                                  \
        im2colData = reinterpret_cast<Type *>(static_scratch_buffer);           \
    } else {                                                                    \
        im2colData = (Type *)malloc(im2colByteSize);                            \
        if (im2colData == nullptr) {                                            \
            return false;                                                       \
        }                                                                       \
        im2colGuard.reset(im2colData);                                          \
    }
#else
#define ANDROID_NN_CONV_PARAMETERS_QUANT8(Type)                                 \
    int32_t height       = (int32_t)getSizeOfDimension(inputShape, 1);          \
    int32_t width        = (int32_t)getSizeOfDimension(inputShape, 2);          \
    int32_t filterHeight = (int32_t)getSizeOfDimension(filterShape, 1);         \
    int32_t filterWidth  = (int32_t)getSizeOfDimension(filterShape, 2);         \
    int32_t outHeight    = (int32_t)getSizeOfDimension(outputShape, 1);         \
    int32_t outWidth     = (int32_t)getSizeOfDimension(outputShape, 2);         \
    int32_t inDepth      = (int32_t)getSizeOfDimension(inputShape, 3);          
#endif
#endif //HIFI_BUILD

bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 int32_t activation,
#ifndef HIFI_NNLIB_OPT
                 float* outputData, const Shape& outputShape) {
#else
                 float* outputData, const Shape& outputShape, void *p_scratch) {
#endif

    ANDROID_NN_CONV_PARAMETERS(float)

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_BUILD
    // Prevent concurrent executions that may access the scratch buffer.
    std::unique_lock<std::mutex> lock(executionMutex);
#endif //HIFI_BUILD
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    (void) height;
    (void) width;
    (void) outWidth;
    (void) outHeight;
    tflite::reference_ops::Conv(
            inputData, convertShapeToDims(inputShape),
            filterData, convertShapeToDims(filterShape),
            biasData, convertShapeToDims(biasShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape),
            im2colData, im2colDim);
    return true;
#else
    int32_t ret, batches = (int)getSizeOfDimension(outputShape, 0), i;
    int32_t out_data_format = 0, outDepth = (int)getSizeOfDimension(outputShape, 3);
    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_conv2d_std_f32(&outputData[i*outHeight*outWidth*outDepth],
                                   &inputData[i*height*width*inDepth],
                                   filterData,
                                   biasData,
                                   height,
                                   width,
                                   inDepth,
                                   filterHeight,
                                   filterWidth,
                                   outDepth,
                                   stride_width,
                                   stride_height,
                                   padding_left,
                                   padding_top,
                                   outHeight,
                                   outWidth,
                                   out_data_format,
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

bool convQuant8(const uint8_t* inputData, const Shape& inputShape,
                const uint8_t* filterData, const Shape& filterShape,
                const int32_t* biasData, const Shape& biasShape,
                int32_t padding_left, int32_t padding_right,
                int32_t padding_top, int32_t padding_bottom,
                int32_t stride_width, int32_t stride_height,
                int32_t activation,
#ifndef HIFI_NNLIB_OPT
                uint8_t* outputData, const Shape& outputShape) {
#else
                uint8_t* outputData, const Shape& outputShape, void *p_scratch) {
#endif

    ANDROID_NN_CONV_PARAMETERS_QUANT8(uint8_t)

    int32_t inputOffset = -inputShape.offset;
    int32_t filterOffset = -filterShape.offset;
    int32_t outputOffset = outputShape.offset;

    float real_multiplier = 0.0;
    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    if (!GetQuantizedConvolutionMultipler(inputShape, filterShape, biasShape,
                                          outputShape, &real_multiplier) ||
            !QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier,
                                              &output_shift)){
        return false;
    }
    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_BUILD
    static gemmlowp::GemmContext gemm_context;

    // Prevent concurrent executions that may access the scratch buffer and
    // gemm_context.
    std::unique_lock<std::mutex> lock(executionMutex);
    // Alow gemmlowp automatically decide how many threads to use.
#endif //HIFI_BUILD
#ifndef HIFI_NNLIB_OPT
    static gemmlowp::GemmContext gemm_context;
    tflite::reference_ops::Conv(
            inputData, convertShapeToDims(inputShape), inputOffset,
            filterData, convertShapeToDims(filterShape), filterOffset,
            biasData, convertShapeToDims(biasShape),
            stride_width, stride_height, paddingWidth, paddingHeight,
            outputOffset, output_multiplier, output_shift,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape),
            im2colData, im2colDim, &gemm_context);
    return true;
#else
    int32_t ret, batches = (int)getSizeOfDimension(outputShape, 0), i;
    int32_t out_data_format = 0, outDepth = (int)getSizeOfDimension(outputShape, 3);
    /* output_shift is negated because it is always right shift in the
    tensorflow version used, this may need to be modified when switching
    to newer version of tensorflow/ANN */
    for(i = 0; i < batches; i++)
    {
        ret = xa_nn_conv2d_std_asym8xasym8(&outputData[i*outHeight*outWidth*outDepth],
                                           &inputData[i*height*width*inDepth],
                                           filterData,
                                           biasData,
                                           height,
                                           width,
                                           inDepth,
                                           filterHeight,
                                           filterWidth,
                                           outDepth,
                                           stride_width,
                                           stride_height,
                                           padding_left,
                                           padding_top,
                                           outHeight,
                                           outWidth,
                                           inputOffset,
                                           filterOffset,
                                           output_multiplier,
                                           -output_shift,
                                           outputOffset,
                                           out_data_format,
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

#undef ANDROID_NN_CONV_PARAMETERS
}  // namespace nn
}  // namespace android
