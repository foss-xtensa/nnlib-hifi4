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
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_OP_RESOLVER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// This is an interface for the OpResolver for TFLiteMicro. The differences from
// the TFLite OpResolver base class are to:
//  * explicitly remove support for Op versions
//  * allow for finer grained registration of the Builtin Ops to reduce code
//    size for TFLiteMicro.
//
// We need an interface class instead of directly using MicroMutableOpResolver
// because MicroMutableOpResolver is a class template with the number of
// registered Ops as the template parameter.
class MicroOpResolver : public OpResolver {
 public:
  typedef TfLiteStatus (*BuiltinParseFunction)(const Operator* op,
                                               ErrorReporter* error_reporter,
                                               BuiltinDataAllocator* allocator,
                                               void** builtin_data);

  // Returns the Op registration struct corresponding to the enum code from the
  // flatbuffer schema. Returns nullptr if the op is not found or if op ==
  // BuiltinOperator_CUSTOM.
  virtual const TfLiteRegistration* FindOp(BuiltinOperator op) const = 0;

  // Returns the Op registration struct corresponding to the custom operator by
  // name.
  virtual const TfLiteRegistration* FindOp(const char* op) const = 0;

  // This implementation exists for compatibility with the OpResolver base class
  // and disregards the version parameter.
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const final {
    return FindOp(op);
  }

  // This implementation exists for compatibility with the OpResolver base class
  // and disregards the version parameter.
  const TfLiteRegistration* FindOp(const char* op, int version) const final {
    return FindOp(op);
  }

  // Returns the operator specific parsing function for the OpData for a
  // BuiltinOperator (if registered), else nullptr.
  virtual BuiltinParseFunction GetOpDataParser(BuiltinOperator op) const = 0;

  ~MicroOpResolver() override {}
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_OP_RESOLVER_H_
