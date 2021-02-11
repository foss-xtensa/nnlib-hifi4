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

#ifndef ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
#define ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H

#ifndef HIFI_BUILD
#include "Utils.h"
#include "HalInterfaces.h"
#else
#define nnAssert(v)
#endif

#include <cstdint>
#include <vector>
#ifndef HIFI_BUILD
#else
#include <algorithm>
#endif //HiFi_BUILD

#ifndef HIFI_BUILD
// Macro to check if the input parameters for operation are valid or not.
#define NN_CHECK(v)                                                     \
  do {                                                                  \
    if (!(v)) {                                                         \
      LOG(ERROR) << "NN_CHECK failed: "  << #v << "'\n";                \
      return false;                                                     \
    }                                                                   \
  } while(0);

#define NN_CHECK_EQ(actual, expected)           \
  NN_CHECK((actual) == (expected))

#define NN_OPS_CHECK NN_CHECK

#else
//TODO: implementation pending
#define NN_CHECK(x)
#define NN_CHECK_EQ(actual, expected)
#define NN_OPS_CHECK NN_CHECK
#define __wur
#endif //HIFI_BUILD

namespace android {
namespace nn {

enum PaddingScheme {
    kPaddingUnknown = 0,
    kPaddingSame = 1,
    kPaddingValid = 2,
};

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

#ifndef HIFI_BUILD
#else
// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // TODO Storing the type here is redundant, as it won't change during execution.
    OperandType type;
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    std::vector<uint32_t> dimensions;

    float scale;
    int32_t zeroPoint;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    uint8_t* buffer;
    // The length of the buffer.
    uint32_t length;
    // Whether this is a temporary variable, a model input, a constant, etc.
    OperandLifeTime lifetime;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;

    Shape shape() const {
        return Shape{.type = type, .dimensions = dimensions, .scale = scale, .offset = zeroPoint};
    }
};
#endif //HiFi_BUILD

// Verifies that the two shapes are the same.
bool SameShape(const Shape& in1, const Shape& in2);

// Sets out to the same shape as in.
bool SetShape(const Shape& in, Shape* out);

// Return the total number of elements, i.e. all the dimensions multiplied
// together. For a scalar, returns one.
uint32_t getNumberOfElements(const Shape& shape);

uint32_t getNumberOfDimensions(const Shape& shape);

uint32_t getSizeOfDimension(const Shape& shape, uint32_t dimensionIdx);

inline uint32_t computeOutSize(uint32_t imageSize, uint32_t filterSize, uint32_t stride,
                               uint32_t paddingHead, uint32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

__wur
bool QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int32_t* right_shift);

__wur
bool QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

__wur
bool GetQuantizedConvolutionMultipler(const Shape& inputShape,
                                      const Shape& filterShape,
                                      const Shape& biasShape,
                                      const Shape& outputShape,
                                      float* multiplier);

void CalculateActivationRangeUint8(int32_t activation,
                                   const Shape& outputShape,
                                   int32_t* act_min,
                                   int32_t* act_max);

void CalculateActivationRangeFloat(int32_t activation,
                                   float* activation_min,
                                   float* activation_max);

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);

inline void calculateExplicitPadding(int32_t in_size, int32_t stride,
                                     int32_t filter_size, int32_t padding_implicit,
                                     int32_t* padding_head, int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

#ifndef HIFI_BUILD
inline PaddingScheme getPaddingScheme(int32_t inWidth, int32_t inHeight,
                                      int32_t strideWidth, int32_t strideHeight,
                                      int32_t filterWidth, int32_t filterHeight,
                                      int32_t paddingLeft, int32_t paddingRight,
                                      int32_t paddingTop, int32_t paddingBottom) {
    if (paddingLeft == 0 && paddingRight == 0 && paddingTop == 0 && paddingBottom == 0) {
        return kPaddingValid;
    }

    int32_t expectedPaddingLeft, expectedPaddingRight;
    int32_t expectedPaddingTop, expectedPaddingBottom;

    calculateExplicitPadding(inWidth, strideWidth, filterWidth, kPaddingSame,
                             &expectedPaddingLeft, &expectedPaddingRight);
    calculateExplicitPadding(inHeight, strideHeight, filterHeight, kPaddingSame,
                             &expectedPaddingTop, &expectedPaddingBottom);
    if (expectedPaddingLeft == paddingLeft && expectedPaddingRight == paddingRight &&
        expectedPaddingTop == paddingTop && expectedPaddingBottom == paddingBottom) {
        return kPaddingSame;
    } else {
        return kPaddingUnknown;
    }
}
#endif //HiFi_BUILD

// TODO: add more documentation from upstream.
// Reverse order of bits in the mask to match the expected order in kernel
inline int ReverseMaskBits(int mask, int num_dimensions) {
  int out = 0;
  for (int dim = 0; dim < num_dimensions; dim++) {
    out <<= 1;
    out += (mask & 1);
    mask >>= 1;
  }
  return out;
}

// TODO: add more documentation from upstream.
inline int32_t PositiveRemainder(int32_t dividend, int32_t divisor) {
  return (divisor + (dividend % divisor)) % divisor;
}

// TODO: add more documentation from upstream.
inline int32_t ClampedIndex(int32_t index, int dim, bool pos_stride) {
  return pos_stride
             ? (index >= dim ? dim
                             : PositiveRemainder(
                                   std::min(std::max(index, -dim), dim), dim))
             : (index < -dim
                    ? -1
                    : PositiveRemainder(
                          std::min(std::max(index, -dim), dim - 1), dim));
}

// Preparation functions for the corresponding ops
bool addMulPrepare(const Shape& in1, const Shape& in2, Shape* out1);

bool floorPrepare(const Shape& input, Shape* output);

bool dequantizePrepare(const Shape& input, Shape* output);

bool depthwiseConvPrepare(const Shape& input,
                          const Shape& filter,
                          const Shape& bias,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
#ifndef HIFI_NNLIB_OPT
                          Shape* output);
#else
                          Shape* output, int32_t& scratch_size);
#endif

bool convPrepare(const Shape& input,
                 const Shape& filter,
                 const Shape& bias,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
#ifndef HIFI_NNLIB_OPT
                 Shape* output);
#else
                 Shape* output, int32_t& scratch_size);
#endif

bool genericPoolingPrepare(const Shape& input,
                           int32_t padding_left, int32_t padding_right,
                           int32_t padding_top, int32_t padding_bottom,
                           int32_t stride_width, int32_t stride_height,
                           int32_t filter_width, int32_t filter_height,
#ifndef HIFI_NNLIB_OPT
                           Shape* output);
#else
                           Shape* output, const Operation& operation, int32_t& scratch_size);
#endif

#ifndef HIFI_NNLIB_OPT
bool genericActivationPrepare(const Shape& input, Shape* output);
#else
bool genericActivationPrepare(const Shape& input, Shape* output, const Operation& operation, int32_t& scratch_size);
#endif

bool fullyConnectedPrepare(const Shape& input,
                           const Shape& weights,
                           const Shape& bias,
                           Shape* output);

bool concatenationPrepare(const std::vector<Shape>& inputShapes,
                          int32_t axis,
                          Shape* output);

bool genericNormalizationPrepare(const Shape& input, Shape* output);

bool reshapePrepare(const Shape& input,
                    const int32_t* targetDims,
                    const int32_t targetDimsSize,
                    Shape* output);

bool resizeBilinearPrepare(const Shape& input,
                           int32_t height,
                           int32_t width,
                           Shape* output);

bool depthToSpacePrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

bool spaceToDepthPrepare(const Shape& input,
                         int32_t blockSize,
                         Shape* output);

bool embeddingLookupPrepare(const Shape &valueShape,
                            const Shape &lookupShape,
                            Shape *outputShape);

bool hashtableLookupPrepare(const Shape &lookupShape,
                            const Shape &keyShape,
                            const Shape &valueShape,
                            Shape *outputShape,
                            Shape *hitShape);

bool padPrepare(const Shape& input,
                const int32_t* paddingsData,
                const Shape& paddingsShape,
                Shape* output);

bool batchToSpacePrepare(const Shape& input,
                         const int32_t* blockSizeData,
                         const Shape& blockSizeShape,
                         Shape* output);

bool spaceToBatchPrepare(const Shape& input,
                         const int32_t* blockSizeData,
                         const Shape& blockSizeShape,
                         const int32_t* paddingsData,
                         const Shape& paddingsShape,
                         Shape* output);

bool squeezePrepare(const Shape& input,
                    const int32_t* squeezeDims,
                    const Shape& squeezeDimsShape,
                    Shape* output);

bool transposePrepare(const Shape& input,
                      const int32_t* permData,
                      const Shape& permShape,
                      Shape* output);

bool meanPrepare(const Shape& input,
                 const int32_t* axisData,
                 const Shape& axisShape,
                 bool keepDims,
                 Shape* output);

bool stridedSlicePrepare(const Shape& input,
                         const int32_t* beginData, const Shape& beginShape,
                         const int32_t* endData, const Shape& endShape,
                         const int32_t* stridesData, const Shape& stridesShape,
                         int32_t beginMask, int32_t endMask, int32_t shrinkAxisMask,
                         Shape* output);
} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_OPERATIONS_UTILS_H
