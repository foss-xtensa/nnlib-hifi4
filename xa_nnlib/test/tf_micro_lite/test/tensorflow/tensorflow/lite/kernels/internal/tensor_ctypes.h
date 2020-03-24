/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

template <typename T>
inline T* GetTensorData(TfLiteTensor* tensor);

template <>
inline float* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.f : nullptr;
}

template <>
inline uint8_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.uint8 : nullptr;
}

template <>
inline int16_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i16 : nullptr;
}

template <>
inline int32_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i32 : nullptr;
}

template <>
inline int64_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i64 : nullptr;
}

template <>
inline bool* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.b : nullptr;
}

template <>
inline int8_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.int8 : nullptr;
}

template <typename T>
inline const T* GetTensorData(const TfLiteTensor* tensor);

template <>
inline const float* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.f : nullptr;
}

template <>
inline const uint8_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.uint8 : nullptr;
}

template <>
inline const int8_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.int8 : nullptr;
}

template <>
inline const int16_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i16 : nullptr;
}

template <>
inline const int32_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i32 : nullptr;
}

template <>
inline const int64_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.i64 : nullptr;
}

template <>
inline const bool* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr ? tensor->data.b : nullptr;
}

inline RuntimeShape GetTensorShape(const TfLiteTensor* tensor) {
  if (tensor == nullptr) {
    return RuntimeShape();
  }

  TfLiteIntArray* dims = tensor->dims;
  const int dims_size = dims->size;
  const int32_t* dims_data = dims->data;
  return RuntimeShape(dims_size, dims_data);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_CTYPES_H_
