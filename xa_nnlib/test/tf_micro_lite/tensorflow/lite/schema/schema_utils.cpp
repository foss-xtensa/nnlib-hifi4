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
#include "tensorflow/lite/schema/schema_utils.h"

#include <algorithm>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

// The following GetBuiltinCode methods are the utility methods for reading
// builtin operatore code, ensuring compatibility issues between v3 and v3a
// schema. Always the maximum value of the two fields always will be the correct
// value as follows:
//
// - Supporting schema version v3 models
//
// The `builtin_code` field is not available in the v3 models. Flatbuffer
// library will feed zero value, which is the default value in the v3a schema.
// The actual builtin operatore code value will exist in the
// `deprecated_builtin_code` field. At the same time, it implies that
// `deprecated_builtin_code` >= `builtin_code` and the maximum value of the two
// fields will be same with `deprecated_builtin_code'.
//
// - Supporting builtin operator codes beyonds 127
//
// New builtin operators, whose operator code is larger than 127, can not be
// assigned to the `deprecated_builtin_code` field. In such cases, the
// value of the `builtin_code` field should be used for the builtin operator
// code. In the case, the maximum value of the two fields will be the value of
// the `builtin_code` as the right value.

BuiltinOperator GetBuiltinCode(const OperatorCode* op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return std::max(
      op_code->builtin_code(),
      static_cast<BuiltinOperator>(op_code->deprecated_builtin_code()));
}

BuiltinOperator GetBuiltinCode(const OperatorCodeT* op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return std::max(op_code->builtin_code, static_cast<BuiltinOperator>(
                                             op_code->deprecated_builtin_code));
}

}  // namespace tflite
