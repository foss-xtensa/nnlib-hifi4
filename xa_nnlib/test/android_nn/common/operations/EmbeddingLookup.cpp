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
#else
#include "Utils_lib.h"
#endif //HiFi_BUILD

#include "EmbeddingLookup.h"

#include "CpuExecutor.h"
#ifndef HIFI_BUILD
#include "HalInterfaces.h"
#include "Operations.h"
#else
#include "xa_nnlib_ann_api.h"
#endif //HiFi_BUILD

namespace android {
namespace nn {

EmbeddingLookup::EmbeddingLookup(const Operation& operation,
                                 std::vector<RunTimeOperandInfo>& operands) {
  value_ = GetInput(operation, operands, kValueTensor);
  lookup_ = GetInput(operation, operands, kLookupTensor);

  output_ = GetOutput(operation, operands, kOutputTensor);
}

bool EmbeddingLookup::Eval() {
  const int row_size = value_->shape().dimensions[0];
  const int total_bytes = sizeOfData(value_->type, value_->dimensions);
  const int row_bytes = total_bytes/row_size;

  for (uint32_t i = 0; i < lookup_->shape().dimensions[0]; i++) {
    int idx = (reinterpret_cast<int*>(lookup_->buffer))[i];
    if (idx >= row_size || idx < 0) {
#ifndef HIFI_BUILD
      LOG(ERROR) << "Embedding Lookup: index out of bounds.";
#endif //HIFI_BUILD
      return false;
    } else {
      memcpy(output_->buffer + i * row_bytes, value_->buffer + idx * row_bytes,
             row_bytes);
    }
  }

  return true;
}

}  // namespace nn
}  // namespace android
