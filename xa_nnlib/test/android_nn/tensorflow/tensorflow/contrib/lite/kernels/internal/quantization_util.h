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
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef PHOTOS_VISION_LEARNING_TENSORFLOW_MINI_QUANTIZATION_UTIL_H_
#define PHOTOS_VISION_LEARNING_TENSORFLOW_MINI_QUANTIZATION_UTIL_H_

#include <cstdint>

namespace tflite {

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* right_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);

// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference
// in Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift);

}  // namespace tflite

#endif  // PHOTOS_VISION_LEARNING_TENSORFLOW_MINI_QUANTIZATION_UTIL_H_
