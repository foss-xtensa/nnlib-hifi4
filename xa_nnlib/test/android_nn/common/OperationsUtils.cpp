/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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

#define LOG_TAG "OperationsUtils"

#ifndef HIFI_BUILD
#include "OperationsUtils.h"
#include "Operations.h"
#else
#include "xa_nnlib_ann_api.h"
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
#include "Utils.h"
#else
#include "Utils_lib.h"
#endif //HiFi_BUILD

#include <cmath>

namespace android {
namespace nn {

bool SameShape(const Shape& in1, const Shape& in2) {
    if (in1.type != in2.type || in1.dimensions.size() != in2.dimensions.size()) {
        return false;
    }
    for (size_t i = 0; i < in1.dimensions.size(); i++) {
        if (in1.dimensions[i] != in2.dimensions[i]) {
            return false;
        }
    }
    return true;
}

bool SetShape(const Shape& in, Shape* out) {
    if (in.type != out->type || in.dimensions.size() != out->dimensions.size()) {
        return false;
    }
    out->dimensions = in.dimensions;
    return true;
}

uint32_t getNumberOfElements(const Shape& shape) {
    uint32_t count = 1;
    for (size_t i = 0; i < shape.dimensions.size(); i++) {
        count *= shape.dimensions[i];
    }
    return count;
}

uint32_t getNumberOfDimensions(const Shape& shape) {
    return shape.dimensions.size();
}

uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx) {
    if (dimensionIdx >= shape.dimensions.size()) {
        // TODO, log the error
        return 0;
    }
    return shape.dimensions[dimensionIdx];
}

bool QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift) {
    NN_OPS_CHECK(double_multiplier >= 0.);
    NN_OPS_CHECK(double_multiplier < 1.);
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *right_shift = 0;
        return true;
    }
    NN_OPS_CHECK(double_multiplier > 0.);
    const double q = std::frexp(double_multiplier, right_shift);
    *right_shift *= -1;
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
    NN_OPS_CHECK(q_fixed <= (1ll << 31));
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        --*right_shift;
    }
    NN_OPS_CHECK(*right_shift >= 0);
    NN_OPS_CHECK(q_fixed <= std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
    return true;
}

bool QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
    NN_OPS_CHECK(double_multiplier > 1.);
    const double q = std::frexp(double_multiplier, left_shift);
    int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
    NN_OPS_CHECK(q_fixed <= (1ll << 31));
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*left_shift;
    }
    NN_OPS_CHECK(*left_shift >= 0);
    NN_OPS_CHECK(q_fixed <= std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
    return true;
}

bool GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier) {
    const float input_product_scale = inputShape.scale * filterShape.scale;
    const float bias_scale = biasShape.scale;
    const float output_scale = outputShape.scale;

    // The following conditions must be guaranteed by the training pipeline.
    NN_OPS_CHECK(std::abs(input_product_scale - bias_scale) <=
              1e-6 * std::min(input_product_scale, bias_scale));
    NN_OPS_CHECK(input_product_scale >= 0);
    NN_OPS_CHECK(input_product_scale < output_scale);
    *multiplier = input_product_scale / output_scale;
#ifndef HIFI_WARNINGS
    (void)bias_scale;
#endif
    return true;
}

void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max) {
    const int32_t qmin = std::numeric_limits<uint8_t>::min();
    const int32_t qmax = std::numeric_limits<uint8_t>::max();

    const auto scale = outputShape.scale;
    const auto zero_point = outputShape.offset;

    auto quantize = [scale, zero_point](float f) {
        return zero_point + static_cast<int32_t>(std::round(f / scale));
    };

    if (activation == kActivationRelu) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = qmax;
    } else if (activation == kActivationRelu6) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = std::min(qmax, quantize(6.0));
    } else if (activation == kActivationRelu1) {
        *act_min = std::max(qmin, quantize(-1.0));
        *act_max = std::min(qmax, quantize(1.0));
    } else if (activation == kActivationNone){
        *act_min = qmin;
        *act_max = qmax;
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported fused activation function.";
#endif //HIFI_BUILD
    }
}

void CalculateActivationRangeFloat(int32_t activation,
                                   float* activation_min,
                                   float* activation_max) {
    if (activation == kActivationRelu) {
        *activation_min = 0.f;
        *activation_max = std::numeric_limits<float>::max();
    } else if (activation == kActivationRelu6) {
        *activation_min = 0.f;
        *activation_max = 6.f;
    } else if (activation == kActivationRelu1) {
        *activation_min = -1.f;
        *activation_max = 1.f;
    } else if (activation == kActivationNone){
        *activation_min = std::numeric_limits<float>::lowest();
        *activation_max = std::numeric_limits<float>::max();
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported fused activation function.";
#endif //HIFI_BUILD
    }
}

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift) {
    const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                      (1ll << (31 - input_integer_bits)) /
                                      (1ll << input_left_shift);
    // Tighten bound using floor.  Suppose that we could use the exact value.
    // After scaling the difference, the result would be at the maximum.  Thus we
    // must ensure that our value has lower magnitude.
    return static_cast<int32_t>(std::floor(max_input_rescaled));
}

bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out) {
    NN_OPS_CHECK(getNumberOfDimensions(in1) <= 4 && getNumberOfDimensions(in2) <= 4);
    NN_OPS_CHECK(in1.type == in2.type);
    if (SameShape(in1, in2)) {
        return SetShape(in1, out);
    } else {
        // BroadcastAdd needed
        uint32_t numberOfDims1 = getNumberOfDimensions(in1);
        uint32_t numberOfDims2 = getNumberOfDimensions(in2);
        uint32_t maxDims = std::max(numberOfDims1, numberOfDims2);
        out->dimensions = std::vector<uint32_t>(maxDims);
        for (uint32_t i = 1; i <= maxDims; i++) {
            uint32_t dim1 = 1;
            if (i <= numberOfDims1) {
                dim1 = getSizeOfDimension(in1, numberOfDims1 - i);
            }
            uint32_t dim2 = 1;
            if (i <= numberOfDims2) {
                dim2 = getSizeOfDimension(in2, numberOfDims2 - i);
            }
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Dimensions mismatch for BroadcastAdd";
#endif //HIFI_BUILD
                return false;
            }
            out->dimensions[maxDims - i] = std::max(dim1, dim2);
        }
    }
    return true;
}

bool floorPrepare(const Shape& input, Shape* output) {
    return SetShape(input, output);
}

bool dequantizePrepare(const Shape& input, Shape* output) {
    if (input.type != OperandType::TENSOR_QUANT8_ASYMM ||
            output->type != OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "bad input / output operand type.";
#endif //HIFI_BUILD
        return false;
    }
    if (input.dimensions.size() != output->dimensions.size()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "input and output tensors don't have the same rank.";
#endif //HIFI_BUILD
        return false;
    }
    output->dimensions = input.dimensions;
    return true;
}

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
#ifndef HIFI_NNLIB_OPT
                 Shape* output) {
#else
                 Shape* output, int32_t& scratch_size) {
#endif
    NN_OPS_CHECK(input.type == filter.type);
    if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
        NN_OPS_CHECK(bias.type == OperandType::TENSOR_INT32);
    } else {
        NN_OPS_CHECK(input.type == bias.type);
    }
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    NN_OPS_CHECK(getNumberOfDimensions(filter) == 4);
    NN_OPS_CHECK(getNumberOfDimensions(bias) == 1);

    NN_OPS_CHECK(getSizeOfDimension(filter, 0) == getSizeOfDimension(bias, 0));
    NN_OPS_CHECK(getSizeOfDimension(filter, 3) == getSizeOfDimension(input, 3));

    uint32_t channels_out = getSizeOfDimension(filter, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    uint32_t outWidth = computeOutSize(width, filterWidth, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filterHeight, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches, outHeight, outWidth, channels_out};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back(outHeight);
    output->dimensions.push_back(outWidth);
    output->dimensions.push_back(channels_out);
#endif
#ifdef HIFI_NNLIB_OPT
    int32_t input_precision=0;
    if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
        input_precision = -3;
    } else if (input.type == OperandType::TENSOR_FLOAT32) {
        input_precision = -1;
    }
    scratch_size = xa_nn_conv2d_std_getsize(height, (int32_t)getSizeOfDimension(input, 3),
                                            filterHeight, filterWidth,
                                            stride_height,
                                            padding_top,
                                            outHeight,
                                            channels_out,
                                            input_precision);
#endif
    return true;
}

bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
#ifndef HIFI_NNLIB_OPT
                          Shape* output) {
#else                      
                          Shape* output, int32_t& scratch_size) {
#endif
    NN_OPS_CHECK(input.type == filter.type);
    if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
        NN_OPS_CHECK(bias.type == OperandType::TENSOR_INT32);
    } else {
        NN_OPS_CHECK(input.type == bias.type);
    }
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    NN_OPS_CHECK(getNumberOfDimensions(filter) == 4);
    NN_OPS_CHECK(getNumberOfDimensions(bias) == 1);

    NN_OPS_CHECK(getSizeOfDimension(filter, 3) == getSizeOfDimension(bias, 0));

    uint32_t channels_out = getSizeOfDimension(filter, 3);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t filterWidth  = getSizeOfDimension(filter, 2);
    uint32_t filterHeight = getSizeOfDimension(filter, 1);
    uint32_t batches      = getSizeOfDimension(input, 0);

    uint32_t outWidth = computeOutSize(width, filterWidth, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filterHeight, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches, outHeight, outWidth, channels_out};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back(outHeight);
    output->dimensions.push_back(outWidth);
    output->dimensions.push_back(channels_out);
#endif
#ifdef HIFI_NNLIB_OPT
    uint32_t channels_in   = getSizeOfDimension(input, 3);
    int32_t input_precision=0;
    if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
        input_precision = -3;
    } else if (input.type == OperandType::TENSOR_FLOAT32) {
        input_precision = -1;
    }
    scratch_size = xa_nn_conv2d_depthwise_getsize(height, width, channels_in,
                                                  filterHeight, filterWidth, channels_out/channels_in,
                                                  stride_width, stride_height,
                                                  padding_left, padding_top,
                                                  outHeight, outWidth,
                                                  input_precision, 0);
#endif
    return true;
}


bool genericPoolingPrepare(const Shape& input,
                           int32_t padding_left, int32_t padding_right,
                           int32_t padding_top, int32_t padding_bottom,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
#ifndef HIFI_NNLIB_OPT
                           Shape* output) {
#else
                           Shape* output, const Operation& operation, int32_t& scratch_size) {
#endif
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);

    uint32_t batches      = getSizeOfDimension(input, 0);
    uint32_t width        = getSizeOfDimension(input, 2);
    uint32_t height       = getSizeOfDimension(input, 1);
    uint32_t channels_out = getSizeOfDimension(input, 3);

    uint32_t outWidth = computeOutSize(width, filter_width, stride_width,
                                       padding_left, padding_right);
    uint32_t outHeight = computeOutSize(height, filter_height, stride_height,
                                        padding_top, padding_bottom);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches, outHeight, outWidth, channels_out};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back(outHeight);
    output->dimensions.push_back(outWidth);
    output->dimensions.push_back(channels_out);
#endif

#ifdef HIFI_NNLIB_OPT
    int32_t input_precision;

    if(input.type == OperandType::TENSOR_QUANT8_ASYMM)
    {
        input_precision = -3;
    }
    else if(input.type == OperandType::TENSOR_FLOAT32)
    {
        input_precision = -1;
    }

    if(operation.type == OperationType::AVERAGE_POOL_2D)
    {
        scratch_size = xa_nn_avgpool_getsize(
                channels_out,
                input_precision,
                input_precision,
                height,
                width,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_bottom,
                outHeight,
                outWidth,
                0,
                0);
    }
    else if(operation.type == OperationType::MAX_POOL_2D)
    {
        scratch_size = xa_nn_maxpool_getsize(
                channels_out,
                input_precision,
                input_precision,
                height,
                width,
                filter_height,
                filter_width,
                stride_width,
                stride_height,
                padding_left,
                padding_bottom,
                outHeight,
                outWidth,
                0,
                0);
    }
    else
    {
        scratch_size = 0;
    }
#endif
    return true;
}


#ifndef HIFI_NNLIB_OPT
bool genericActivationPrepare(const Shape& input,
                              Shape* output) {
    NN_OPS_CHECK(getNumberOfDimensions(input) <= 4);

    return SetShape(input, output);
}
#else
bool genericActivationPrepare(const Shape& input,
                              Shape* output, const Operation& operation, int32_t& scratch_size) {
    NN_OPS_CHECK(getNumberOfDimensions(input) <= 4);

    int32_t input_precision=0, io_length=0, batch_size=0;

    if(input.type == OperandType::TENSOR_QUANT8_ASYMM)
    {
        input_precision = -3;
    }
    else if(input.type == OperandType::TENSOR_FLOAT32)
    {
        input_precision = -1;
    }

    if(operation.type == OperationType::SOFTMAX)
    {
        if (getNumberOfDimensions(input) == 2) 
        {
            batch_size = getSizeOfDimension(input, 0);
            io_length = getNumberOfElements(input) / batch_size;
        }
        else if (getNumberOfDimensions(input) == 4)
        {
//            io_length = MatchingArraySize(input, 0, input, 0);
            io_length = getNumberOfElements(input);
        }
        scratch_size = get_softmax_scratch_size(input_precision, input_precision, io_length);
    }
    else
    {
        scratch_size = 0;
    }
    return SetShape(input, output);
}
#endif

bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output) {
    // Check all the parameters of tensor match within themselves and match the
    // input configuration.
    NN_OPS_CHECK(input.type == weights.type);
    if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
        NN_OPS_CHECK(bias.type == OperandType::TENSOR_INT32);
    } else {
        NN_OPS_CHECK(input.type == bias.type);
    }
    // The Tensorflow fully connected layer specification says that input should
    // be of at least rank 2, so we check. Tflite doesn't check.
    NN_OPS_CHECK(getNumberOfDimensions(input) >= 2);
    NN_OPS_CHECK(getNumberOfDimensions(weights) == 2);
    uint32_t input_n_elements = getNumberOfElements(input);
    uint32_t num_units  = getSizeOfDimension(weights, 0);
    uint32_t input_size = getSizeOfDimension(weights, 1);
    uint32_t batch_size = input_n_elements / input_size;

    NN_OPS_CHECK(getSizeOfDimension(bias, 0) == num_units);
    NN_OPS_CHECK(input_size * batch_size == input_n_elements);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batch_size, num_units};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batch_size);
    output->dimensions.push_back(num_units);
#endif

    return true;
}

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output) {

    int num_inputs = inputShapes.size();
    OperandType input_type = inputShapes[0].type;
    uint32_t num_dimensions = getNumberOfDimensions(inputShapes[0]);

    NN_OPS_CHECK(axis >= 0);
    NN_OPS_CHECK(axis < (int32_t)num_dimensions);

    int sumAxis = getSizeOfDimension(inputShapes[0], axis);
    for (int i = 1; i < num_inputs; ++i) {
        NN_OPS_CHECK(getNumberOfDimensions(inputShapes[i]) == num_dimensions);
        NN_OPS_CHECK(inputShapes[i].type == inputShapes[0].type);
        if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
            NN_OPS_CHECK(inputShapes[0].offset == inputShapes[i].offset);
            NN_OPS_CHECK(inputShapes[0].scale == inputShapes[i].scale);
        }
        for (int d = 0; d < (int32_t)num_dimensions; ++d) {
            if (d == axis) {
                sumAxis += getSizeOfDimension(inputShapes[i], axis);
            } else {
                NN_OPS_CHECK(getSizeOfDimension(inputShapes[0], d) ==
                           getSizeOfDimension(inputShapes[i], d));
            }
        }
    }

    output->type = input_type;
    output->dimensions = inputShapes[0].dimensions;
    output->dimensions[axis] = sumAxis;

    if (input_type == OperandType::TENSOR_QUANT8_ASYMM) {
        NN_OPS_CHECK(inputShapes[0].offset == output->offset);
        NN_OPS_CHECK(inputShapes[0].scale == output->scale);
    }

    return true;
}


bool genericNormalizationPrepare(const Shape& input, Shape* output) {
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    return SetShape(input, output);
}

bool reshapePrepare(const Shape& input,
                    const int32_t* targetDims,
                    const int32_t targetDimsSize,
                    Shape* output) {
    // Reshape allows one of the targetDims components to have the
    // special -1 value, meaning it will be calculated automatically based on the
    // input. Here we calculate what that dimension should be so that the number
    // of output elements in the same as the number of input elements.
    int32_t numInputElements = (int32_t) getNumberOfElements(input);

    std::vector<uint32_t> outDims(targetDimsSize);
    int32_t numOutputElements = 1;
    int32_t strechDim = -1;
    for (int32_t i = 0; i < targetDimsSize; ++i) {
        int32_t value = targetDims[i];
        if (value == -1) {
            NN_OPS_CHECK(strechDim == -1);
            strechDim = i;
        } else {
            numOutputElements *= value;
            outDims[i] = (uint32_t)value;
        }
    }
    if (strechDim != -1) {
        int32_t strechValue = numInputElements / numOutputElements;
        outDims[strechDim] = (uint32_t) strechValue;
        numOutputElements *= strechValue;
    }

    NN_OPS_CHECK(numInputElements == numOutputElements);

    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool resizeBilinearPrepare(const Shape& input,
                           int32_t width,
                           int32_t height,
                           Shape* output) {
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t channels = getSizeOfDimension(input, 3);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches, (uint32_t)height, (uint32_t)width, channels};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back((uint32_t)height);
    output->dimensions.push_back((uint32_t)width);
    output->dimensions.push_back(channels);
#endif

    return true;
}

bool depthToSpacePrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output) {
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    NN_OPS_CHECK(blockSize > 0);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    NN_OPS_CHECK(channels % (blockSize * blockSize) == 0);
    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches,
                          height * blockSize,
                          width * blockSize,
                          channels / (blockSize * blockSize)};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back(height * blockSize);
    output->dimensions.push_back(width * blockSize);
    output->dimensions.push_back(channels / (blockSize * blockSize));
#endif
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool spaceToDepthPrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output) {
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);
    NN_OPS_CHECK(blockSize > 0);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    NN_OPS_CHECK(height % blockSize == 0);
    NN_OPS_CHECK(width % blockSize == 0);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches,
                          height / blockSize,
                          width / blockSize,
                          channels * (blockSize * blockSize)};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches);
    output->dimensions.push_back(height / blockSize);
    output->dimensions.push_back(width / blockSize);
    output->dimensions.push_back(channels * (blockSize * blockSize));
#endif
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool embeddingLookupPrepare(const Shape &valueShape,
                            const Shape &lookupShape,
                            Shape *outputShape) {
    NN_OPS_CHECK(getNumberOfDimensions(valueShape) >= 2);
    NN_OPS_CHECK(getNumberOfDimensions(lookupShape) == 1);

    const uint32_t rows     = getSizeOfDimension(valueShape, 0);
    const uint32_t columns  = getSizeOfDimension(valueShape, 1);

    const uint32_t lookups  = getSizeOfDimension(lookupShape, 0);

    outputShape->type = valueShape.type;
#ifndef HIFI_BUILD
    outputShape->dimensions = { lookups, columns };
#else
    outputShape->dimensions.clear();
    outputShape->dimensions.push_back(lookups);
    outputShape->dimensions.push_back(columns);
#endif
    for (uint32_t i = 2; i < getNumberOfDimensions(valueShape); i++) {
        outputShape->dimensions.push_back(getSizeOfDimension(valueShape, i));
    }
    outputShape->offset = valueShape.offset;
    outputShape->scale = valueShape.scale;
#ifndef HIFI_WARNINGS 
    (void)rows;
#endif
    return true;
}

bool hashtableLookupPrepare(const Shape &lookupShape,
                            const Shape &keyShape,
                            const Shape &valueShape,
                            Shape *outputShape,
                            Shape *hitShape) {
    NN_OPS_CHECK(getNumberOfDimensions(lookupShape) == 1);
    NN_OPS_CHECK(getNumberOfDimensions(keyShape) == 1);
    NN_OPS_CHECK(getNumberOfDimensions(valueShape) >= 1);

    const uint32_t lookups  = getSizeOfDimension(lookupShape, 0);
    const uint32_t keys     = getSizeOfDimension(keyShape, 0);
    const uint32_t rows     = getSizeOfDimension(valueShape, 0);
    outputShape->type = valueShape.type;
#ifndef HIFI_BUILD
    outputShape->dimensions = { lookups };
#else
    outputShape->dimensions.clear();
    outputShape->dimensions.push_back(lookups);
#endif
    for (uint32_t i = 1; i < getNumberOfDimensions(valueShape); i++) {
        outputShape->dimensions.push_back(getSizeOfDimension(valueShape, i));
    }
    outputShape->offset = valueShape.offset;
    outputShape->scale = valueShape.scale;

    hitShape->type = OperandType::TENSOR_QUANT8_ASYMM;
#ifndef HIFI_BUILD
    hitShape->dimensions = { lookups };
#else
    hitShape->dimensions.clear();
    hitShape->dimensions.push_back(lookups);
#endif
    hitShape->offset = 0;
    hitShape->scale = 1.f;
#ifndef HIFI_WARNINGS 
    (void)rows;
    (void)keys;
#endif
    return true;
}

bool padPrepare(const Shape& input,
                const int32_t* paddingsData,
                const Shape& paddingsShape,
                Shape* output) {
    // Currently only 4D tensors are supported.
    uint32_t numInputDims = getNumberOfDimensions(input);
    NN_OPS_CHECK(numInputDims == 4);

    // paddings need to be provided as a 2-D int32 tensor.
    NN_OPS_CHECK(paddingsShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(paddingsShape) == 2);
    NN_OPS_CHECK(getSizeOfDimension(paddingsShape, 0) == numInputDims);
    NN_OPS_CHECK(getSizeOfDimension(paddingsShape, 1) == 2);

    std::vector<uint32_t> outDims(numInputDims);
    for (uint32_t i = 0; i < numInputDims; ++i) {
        int32_t beforePadding = *paddingsData++;
        int32_t afterPadding = *paddingsData++;
        // Pad value has to be greater than equal to 0.
        NN_OPS_CHECK(beforePadding >= 0 && afterPadding >= 0);
        outDims[i] = beforePadding + getSizeOfDimension(input, i) + afterPadding;
    }
    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool batchToSpacePrepare(const Shape& input,
                         const int32_t* blockSizeData,
                         const Shape& blockSizeShape,
                         Shape* output) {
    // Only 4D NHWC tensors are supported.
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);

    // blockSize need to be provided as a 1-D int32 tensor.
    NN_OPS_CHECK(blockSizeShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(blockSizeShape) == 1);
    // Only applies to spatial dimensions.
    NN_OPS_CHECK(getSizeOfDimension(blockSizeShape, 0) == 2);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    NN_OPS_CHECK(batches % (blockSizeData[0] * blockSizeData[1]) == 0);
    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches / (blockSizeData[0] * blockSizeData[1]),
                          height * blockSizeData[0],
                          width * blockSizeData[1],
                          channels};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches / (blockSizeData[0] * blockSizeData[1]));
    output->dimensions.push_back(height * blockSizeData[0]);
    output->dimensions.push_back(width * blockSizeData[1]);
    output->dimensions.push_back(channels);
#endif
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool spaceToBatchPrepare(const Shape& input,
                         const int32_t* blockSizeData,
                         const Shape& blockSizeShape,
                         const int32_t* paddingsData,
                         const Shape& paddingsShape,
                         Shape* output) {
    // Only 4D NHWC tensors are supported.
    NN_OPS_CHECK(getNumberOfDimensions(input) == 4);

    // blockSize need to be provided as a 1-D int32 tensor.
    NN_OPS_CHECK(blockSizeShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(blockSizeShape) == 1);
    // Only applies to spatial dimensions.
    NN_OPS_CHECK(getSizeOfDimension(blockSizeShape, 0) == 2);

    // paddings need to be provided as a 2-D int32 tensor.
    NN_OPS_CHECK(paddingsShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(paddingsShape) == 2);
    NN_OPS_CHECK(getSizeOfDimension(paddingsShape, 0) == 2);
    NN_OPS_CHECK(getSizeOfDimension(paddingsShape, 1) == 2);

    uint32_t batches  = getSizeOfDimension(input, 0);
    uint32_t height   = getSizeOfDimension(input, 1);
    uint32_t width    = getSizeOfDimension(input, 2);
    uint32_t channels = getSizeOfDimension(input, 3);

    uint32_t paddedHeight = paddingsData[0] + height + paddingsData[1];
    uint32_t paddedWidth = paddingsData[2] + width + paddingsData[3];

    NN_OPS_CHECK(paddedHeight % blockSizeData[0] == 0);
    NN_OPS_CHECK(paddedWidth % blockSizeData[1] == 0);

    output->type = input.type;
#ifndef HIFI_BUILD
    output->dimensions = {batches * (blockSizeData[0] * blockSizeData[1]),
                          paddedHeight / blockSizeData[0],
                          paddedWidth / blockSizeData[1],
                          channels};
#else
    output->dimensions.clear();
    output->dimensions.push_back(batches * (blockSizeData[0] * blockSizeData[1]));
    output->dimensions.push_back(paddedHeight / blockSizeData[0]);
    output->dimensions.push_back(paddedWidth / blockSizeData[1]);
    output->dimensions.push_back(channels);
#endif
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool squeezePrepare(const Shape& input,
                    const int32_t* squeezeDims,
                    const Shape& squeezeDimsShape,
                    Shape* output) {
    int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(input));

    // squeezeDims need to be provided as a 1-D int32 tensor.
    NN_OPS_CHECK(squeezeDimsShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(squeezeDimsShape) == 1);

    int32_t squeezeDimsSize = static_cast<int32_t>(getSizeOfDimension(squeezeDimsShape, 0));
    std::vector<bool> shouldSqueeze(numInputDims, false);
    int32_t numDimsSqueezed = 0;

    if (squeezeDimsSize == 0) {
        // If squeezeDimsSize is 0, all dims with value 1 will be squeezed.
        for (int32_t idx = 0; idx < numInputDims; ++idx) {
            if (getSizeOfDimension(input, idx) == 1) {
                shouldSqueeze[idx] = true;
                ++numDimsSqueezed;
            }
        }
    } else {
        for (int32_t idx = 0; idx < squeezeDimsSize; ++idx) {
            int32_t current = squeezeDims[idx] < 0 ? squeezeDims[idx] + numInputDims
                                               : squeezeDims[idx];
            NN_OPS_CHECK(current >= 0 && current < numInputDims &&
                         getSizeOfDimension(input, current) == 1);
            if (!shouldSqueeze[current]) ++numDimsSqueezed;
            shouldSqueeze[current] = true;
      }
    }

    // Sets output dimensions.
    std::vector<uint32_t> outDims(numInputDims - numDimsSqueezed);
    for (int32_t inIdx = 0, outIdx = 0; inIdx < numInputDims; ++inIdx) {
        if (!shouldSqueeze[inIdx]) {
            outDims[outIdx++] = getSizeOfDimension(input, inIdx);
        }
    }

    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool transposePrepare(const Shape& input,
                      const int32_t* permData,
                      const Shape& permShape,
                      Shape* output) {
    uint32_t numInputDims = getNumberOfDimensions(input);
    // Transpose op only supports 1D-4D input arrays.
    NN_OPS_CHECK(numInputDims <= 4);

    // perm need to be provided as a 1-D int32 tensor.
    NN_OPS_CHECK(permShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(permShape) == 1);
    NN_OPS_CHECK(numInputDims == getSizeOfDimension(permShape, 0));

    std::vector<uint32_t> outDims(numInputDims);
    for (int32_t idx = 0; idx < static_cast<int32_t>(numInputDims); ++idx) {
        NN_OPS_CHECK(permData[idx] >= 0 && permData[idx] < static_cast<int32_t>(numInputDims));
        outDims[idx] = getSizeOfDimension(input, permData[idx]);
    }

    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool meanPrepare(const Shape& input,
                 const int32_t* axisData,
                 const Shape& axisShape,
                 bool keepDims,
                 Shape* output) {

    // perm need to be provided as a 1-D int32 tensor.
    NN_OPS_CHECK(axisShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(getNumberOfDimensions(axisShape) == 1);

    int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(input));
    int32_t axisSize = static_cast<int32_t>(getSizeOfDimension(axisShape, 0));

    // Determines size of output tensor.
    if (keepDims) {
        std::vector<uint32_t> outDims(numInputDims);
        for (int32_t idx = 0; idx < numInputDims; ++idx) {
            bool isAxis = false;
            for (int32_t axisIdx = 0; axisIdx < axisSize; ++axisIdx) {
                if (axisData[axisIdx] == idx || axisData[axisIdx] + numInputDims == idx) {
                    isAxis = true;
                    break;
                }
            }
            if (isAxis) {
                outDims[idx] = 1;
            } else {
                outDims[idx] = getSizeOfDimension(input, idx);
            }
        }
        output->dimensions = outDims;
    } else {
        // Calculates size of reducing axis.
        int32_t numReduceAxis = axisSize;
        for (int32_t i = 0; i < axisSize; ++i) {
            int32_t current = axisData[i];
            if (current < 0) {
                current += numInputDims;
            }
            NN_OPS_CHECK(current >= 0 && current < numInputDims);
            for (int32_t j = 0; j < i; ++j) {
                int32_t previous = axisData[j];
                if (previous < 0) {
                    previous += numInputDims;
                }
                if (current == previous) {
                    --numReduceAxis;
                    break;
                }
            }
        }
        // Determines output dimensions.
        std::vector<uint32_t> outDims(numInputDims - numReduceAxis);
        int32_t numSkipAxis = 0;
        for (int32_t idx = 0; idx < numInputDims; ++idx) {
            bool isAxis = false;
            for (int32_t axisIdx = 0; axisIdx < axisSize; ++axisIdx) {
                if (axisData[axisIdx] == idx || axisData[axisIdx] + numInputDims == idx) {
                    ++numSkipAxis;
                    isAxis = true;
                    break;
                }
            }
            if (!isAxis) {
                outDims[idx - numSkipAxis] = getSizeOfDimension(input, idx);
            }
        }
        output->dimensions = outDims;
    }

    output->type = input.type;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}

bool stridedSlicePrepare(const Shape& input,
                         const int32_t* beginData, const Shape& beginShape,
                         const int32_t* endData, const Shape& endShape,
                         const int32_t* stridesData, const Shape& stridesShape,
                         int32_t beginMask, int32_t endMask, int32_t shrinkAxisMask,
                         Shape* output) {
    uint32_t numInputDims = getNumberOfDimensions(input);
    // StridedSlice op only supports 1D-4D input arrays.
    NN_OPS_CHECK(numInputDims <= 4);

    NN_OPS_CHECK(getNumberOfDimensions(beginShape) == 1);
    NN_OPS_CHECK(getNumberOfDimensions(endShape) == 1);
    NN_OPS_CHECK(getNumberOfDimensions(stridesShape) == 1);

    NN_OPS_CHECK(getSizeOfDimension(beginShape, 0) == numInputDims);
    NN_OPS_CHECK(getSizeOfDimension(endShape, 0) == numInputDims);
    NN_OPS_CHECK(getSizeOfDimension(stridesShape, 0) == numInputDims);

    NN_OPS_CHECK(beginShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(endShape.type == OperandType::TENSOR_INT32);
    NN_OPS_CHECK(stridesShape.type == OperandType::TENSOR_INT32);

    // Determine size of output tensor and map indices
    std::vector<uint32_t> outDims;
    for (int32_t idx = 0; idx < static_cast<int32_t>(numInputDims); idx++) {
      int32_t dim = static_cast<int32_t>(getSizeOfDimension(input, idx));
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

      // This is valid for both positive and negative strides
      int32_t outDim = ceil((end - begin) / static_cast<float>(stride));
      outDim = outDim < 0 ? 0 : static_cast<uint32_t>(outDim);
      if (!(shrinkAxisMask & (1 << idx))) {
          outDims.push_back(outDim);
      } else {
          if (outDim != 1) {
#ifndef HIFI_BUILD
              LOG(ERROR) << "Outdim " << idx << " is " << outDim << ", expected 1";
#endif //HIFI_BUILD
              NN_OPS_CHECK(outDim == 1);
          }
      }
    }

    output->type = input.type;
    output->dimensions = outDims;
    output->offset = input.offset;
    output->scale = input.scale;

    return true;
}
} // namespace nn
} // namespace android
