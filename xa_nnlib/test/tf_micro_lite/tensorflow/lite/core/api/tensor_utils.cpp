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
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/api/tensor_utils.h"

#include <string.h>

namespace tflite {

TfLiteStatus ResetVariableTensor(TfLiteTensor* tensor) {
  if (!tensor->is_variable) {
    return kTfLiteOk;
  }
  // TODO(b/115961645): Implement - If a variable tensor has a buffer, reset it
  // to the value of the buffer.
  int value = 0;
  if (tensor->type == kTfLiteInt8) {
    value = tensor->params.zero_point;
  }
  // TODO(b/139446230): Provide a platform header to better handle these
  // specific scenarios.
#if __ANDROID__ || defined(__x86_64__) || defined(__i386__) || \
    defined(__i386) || defined(__x86__) || defined(__X86__) || \
    defined(_X86_) || defined(_M_IX86) || defined(_M_X64)
  memset(tensor->data.raw, value, tensor->bytes);
#else
  char* raw_ptr = tensor->data.raw;
  for (size_t i = 0; i < tensor->bytes; ++i) {
    *raw_ptr = value;
    raw_ptr++;
  }
#endif
  return kTfLiteOk;
}

}  // namespace tflite
