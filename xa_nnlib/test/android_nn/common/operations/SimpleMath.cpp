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

// Contains the implementation of the operations.

#define LOG_TAG "Operations"

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
#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            LOG(ERROR) << "Unsupported fused activation function type";     \
            return false;                                                   \
    }
#else
#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            return false;                                                   \
    }
#endif //HIFI_BUILD

bool addFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    if (needBroadcast) {
#ifndef HIFI_BUILD
        #define ANDROID_NN_BROADCAST_ADD(activation)                                              \
            tflite::optimized_ops::BroadcastAdd<tflite::FusedActivationFunctionType::activation>( \
                    in1, convertShapeToDims(shape1),                                              \
                    in2, convertShapeToDims(shape2),                                              \
                    out, convertShapeToDims(shapeOut))
#else
        #define ANDROID_NN_BROADCAST_ADD(activation)                                              \
            tflite::reference_ops::BroadcastAdd<tflite::FusedActivationFunctionType::activation>( \
                    in1, convertShapeToDims(shape1),                                              \
                    in2, convertShapeToDims(shape2),                                              \
                    out, convertShapeToDims(shapeOut))
#endif //HiFi_BUILD

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_ADD)
        #undef ANDROID_NN_BROADCAST_ADD
    } else {
        float output_activation_min, output_activation_max;
        CalculateActivationRangeFloat(activation, &output_activation_min,
                                      &output_activation_max);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
        tflite::reference_ops::Add(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        ret = xa_nn_elm_add_f32xf32_f32(out, in1, in2, num_elms);
        if(ret)
            return false;
        ret = xa_nn_vec_activation_min_max_f32_f32(out, out,
                output_activation_min, output_activation_max,
                num_elms);
#endif
    }

    return true;
}

bool addQuant8(const uint8_t* in1, const Shape& shape1,
               const uint8_t* in2, const Shape& shape2,
               int32_t activation,
               uint8_t* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    const int32_t input1_offset = -shape1.offset;
    const int32_t input2_offset = -shape2.offset;
    const int32_t output_offset = shapeOut.offset;
    const int left_shift = 20;
    const double twice_max_input_scale = 2 * std::max(shape1.scale, shape2.scale);
    const double real_input1_multiplier = shape1.scale / twice_max_input_scale;
    const double real_input2_multiplier = shape2.scale / twice_max_input_scale;
    const double real_output_multiplier =
            twice_max_input_scale /
            ((1 << left_shift) * shapeOut.scale);

    int32_t input1_multiplier;
    int32_t input1_shift;
    if (!QuantizeMultiplierSmallerThanOne(real_input1_multiplier,
                                          &input1_multiplier, &input1_shift)) {
        return false;
    }
    int32_t input2_multiplier;
    int32_t input2_shift;
    if (!QuantizeMultiplierSmallerThanOne(real_input2_multiplier,
                                          &input2_multiplier, &input2_shift)) {
        return false;
    }
    int32_t output_multiplier;
    int32_t output_shift;
    if (!QuantizeMultiplierSmallerThanOne(real_output_multiplier,
                                          &output_multiplier, &output_shift)) {
        return false;
    }
    int32_t output_activation_min;
    int32_t output_activation_max;
    CalculateActivationRangeUint8(activation, shapeOut,
                                  &output_activation_min,
                                  &output_activation_max);

    if (needBroadcast) {
#ifndef HIFI_BUILD
        tflite::reference_ops::BroadcastAdd(
                left_shift,
                in1, convertShapeToDims(shape1),
                input1_offset, input1_multiplier, input1_shift,
                in2, convertShapeToDims(shape2),
                input2_offset, input2_multiplier, input2_shift,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        tflite::reference_ops::BroadcastAdd(
                left_shift,
                in1, convertShapeToDims(shape1),
                input1_offset, input1_multiplier, input1_shift,
                in2, convertShapeToDims(shape2),
                input2_offset, input2_multiplier, input2_shift,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_NNLIB_OPT
        #define ANDROID_NN_NORMAL_ADD(activation)                                        \
            tflite::reference_ops::Add<tflite::FusedActivationFunctionType::activation>( \
                    left_shift,                                                          \
                    in1, convertShapeToDims(shape1),                                     \
                    input1_offset, input1_multiplier, input1_shift,                      \
                    in2, convertShapeToDims(shape2),                                     \
                    input2_offset, input2_multiplier, input2_shift,                      \
                    output_offset, output_multiplier, output_shift,                      \
                    output_activation_min, output_activation_max,                        \
                    out, convertShapeToDims(shapeOut))

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_NORMAL_ADD)
        #undef ANDROID_NN_NORMAL_ADD
#else
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        /* Shifts are negated because these are always right shifts in the
        tensorflow version used, this may need to be modified when switching
        to newer version of tensorflow/ANN */
        ret = xa_nn_elm_add_asym8xasym8_asym8(out, output_offset, -output_shift, output_multiplier,
                output_activation_min, output_activation_max,
                in1, input1_offset, -input1_shift, input1_multiplier,
                in2, input2_offset, -input2_shift, input2_multiplier,
                left_shift, num_elms);
        if(ret)
            return false;
#endif
    }

    return true;
}

bool mulFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    if (needBroadcast) {
#ifndef HIFI_BUILD
    #define ANDROID_NN_BROADCAST_MUL(activation)                                              \
        tflite::reference_ops::BroadcastMul<tflite::FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                                              \
                in2, convertShapeToDims(shape2),                                              \
                out, convertShapeToDims(shapeOut))
#else
    #define ANDROID_NN_BROADCAST_MUL(activation)                                              \
        tflite::reference_ops::BroadcastMul<tflite::FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                                              \
                in2, convertShapeToDims(shape2),                                              \
                out, convertShapeToDims(shapeOut))
#endif //HiFi_BUILD

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_MUL)
        #undef ANDROID_NN_BROADCAST_MUL
    } else {
        float output_activation_min, output_activation_max;
        CalculateActivationRangeFloat(activation, &output_activation_min,
                                      &output_activation_max);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
        tflite::reference_ops::Mul(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        ret = xa_nn_elm_mul_f32xf32_f32(out, in1, in2, num_elms);
        if(ret)
            return false;
        ret = xa_nn_vec_activation_min_max_f32_f32(out, out,
                output_activation_min, output_activation_max,
                num_elms);
#endif
    }

    return true;
}

bool mulQuant8(const uint8_t* in1, const Shape& shape1,
               const uint8_t* in2, const Shape& shape2,
               int32_t activation,
               uint8_t* out, const Shape& shapeOut) {
    const int32_t input1_offset = -shape1.offset;
    const int32_t input2_offset = -shape2.offset;
    const int32_t output_offset = shapeOut.offset;
    const double input_product_scale = shape1.scale * shape2.scale;
    const double real_multiplier = input_product_scale / shapeOut.scale;
    int32 output_multiplier;
    int output_shift;
    if (!QuantizeMultiplierSmallerThanOne(real_multiplier, &output_multiplier,
                                          &output_shift)) {
        return false;
    }
    int32_t output_activation_min;
    int32_t output_activation_max;
    CalculateActivationRangeUint8(activation, shapeOut,
                                  &output_activation_min,
                                  &output_activation_max);

#ifndef HIFI_NNLIB_OPT
    // Use BROADCAST version to handle the normal case.
    tflite::reference_ops::BroadcastMul(
                in1, convertShapeToDims(shape1), input1_offset,
                in2, convertShapeToDims(shape2), input2_offset,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
    bool needBroadcast = !SameShape(shape1, shape2);
    if (needBroadcast) {
        tflite::reference_ops::BroadcastMul(
                    in1, convertShapeToDims(shape1), input1_offset,
                    in2, convertShapeToDims(shape2), input2_offset,
                    output_offset, output_multiplier, output_shift,
                    output_activation_min, output_activation_max,
                    out, convertShapeToDims(shapeOut));
    }
    else {
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        /* output_shift is negated because it is always right shift in the
        tensorflow version used, this may need to be modified when switching
        to newer version of tensorflow/ANN */
        ret = xa_nn_elm_mul_asym8xasym8_asym8(out, output_offset, -output_shift, output_multiplier,
                output_activation_min, output_activation_max,
                in1, input1_offset, in2, input2_offset, num_elms);
        if(ret)
            return false;
    }
#endif

    return true;
}

bool floorFloat32(const float* inputData,
                  float* outputData,
                  const Shape& shape) {
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    tflite::Dims<4> dim = convertShapeToDims(shape);
    tflite::reference_ops::Floor(inputData, dim, outputData, dim);
#else
    int ret, num_elms;
    num_elms = getNumberOfElements(shape);
    ret = xa_nn_elm_floor_f32_f32(outputData, inputData, num_elms);
    if(ret)
        return false;
#endif
    return true;
}

bool dequantizeQuant8ToFloat32(const uint8_t* inputData,
                               float* outputData,
                               const Shape& shape) {
    tflite::Dims<4> dim = convertShapeToDims(shape);
#ifndef HIFI_BUILD
    tflite::reference_ops::Dequantize(inputData, dim,
                                      shape.offset, shape.scale,
                                      outputData, dim);
#else
    tflite::reference_ops::Dequantize(inputData, dim,
                                      shape.offset, shape.scale,
                                      outputData, dim);
#endif //HiFi_BUILD
    return true;
}

bool subFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    bool needBroadcast = !SameShape(shape1, shape2);
    if (needBroadcast) {
#ifndef HIFI_BUILD
        tflite::reference_ops::BroadcastSub(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        tflite::reference_ops::BroadcastSub(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#endif //HiFi_BUILD
    } else {
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
        tflite::reference_ops::Sub(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        ret = xa_nn_elm_sub_f32xf32_f32(out, in1, in2, num_elms);
        if(ret)
            return false;
        ret = xa_nn_vec_activation_min_max_f32_f32(out, out,
                output_activation_min, output_activation_max,
                num_elms);
#endif
    }
    return true;
}

bool divFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    bool needBroadcast = !SameShape(shape1, shape2);
    if (needBroadcast) {
#ifndef HIFI_BUILD
        tflite::reference_ops::BroadcastDiv(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        tflite::reference_ops::BroadcastDiv(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#endif //HiFi_BUILD
    } else {
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
        tflite::reference_ops::Div(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
#else
        int ret, num_elms;
        num_elms = getNumberOfElements(shape1);
        ret = xa_nn_elm_div_f32xf32_f32(out, in1, in2, num_elms);
        if(ret)
            return false;
        ret = xa_nn_vec_activation_min_max_f32_f32(out, out,
                output_activation_min, output_activation_max,
                num_elms);
#endif
    }
    return true;
}

bool meanGeneric(const uint8_t* inputData, const Shape& inputShape,
                 const int32_t* axis, const Shape& axisShape, bool keepDims,
                 uint8_t* outputData, const Shape& outputShape) {
    // Creates a temp index to iterate through input data.
    int32_t* scratchBuffer = new int32_t[getNumberOfDimensions(inputShape)];

    // Creates a temp tensor to store resolved axis given input data.
    int32_t axisSize = static_cast<int32_t>(getSizeOfDimension(axisShape, 0));
    int32_t* resolvedAxis = new int32_t[axisSize];

    bool result = true;
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::reference_ops::Mean<float>(
                const_cast<float*>(reinterpret_cast<const float*>(inputData)),
                reinterpret_cast<const int*>(inputShape.dimensions.data()),
                getNumberOfDimensions(inputShape),
                reinterpret_cast<float*>(outputData),
                reinterpret_cast<const int*>(outputShape.dimensions.data()),
                getNumberOfDimensions(outputShape),
                axis, axisSize, keepDims, scratchBuffer, resolvedAxis);
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::reference_ops::Mean<uint8_t>(
                const_cast<uint8_t*>(inputData),
                reinterpret_cast<const int*>(inputShape.dimensions.data()),
                getNumberOfDimensions(inputShape),
                outputData,
                reinterpret_cast<const int*>(outputShape.dimensions.data()),
                getNumberOfDimensions(outputShape),
                axis, axisSize, keepDims, scratchBuffer, resolvedAxis);
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        result = false;
    }
    delete[] scratchBuffer;
    delete[] resolvedAxis;
    return result;
}
} // namespace nn
} // namespace android
