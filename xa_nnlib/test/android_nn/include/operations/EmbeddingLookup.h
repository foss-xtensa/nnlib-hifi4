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

#ifndef FRAMEWORKS_ML_NN_EMBEDDING_LOOKUP_H
#define FRAMEWORKS_ML_NN_EMBEDDING_LOOKUP_H

#include <vector>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_1 {
struct Operation;
}
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

namespace android {
namespace nn {

struct RunTimeOperandInfo;

class EmbeddingLookup {
 public:
  EmbeddingLookup(
      const android::hardware::neuralnetworks::V1_1::Operation &operation,
      std::vector<RunTimeOperandInfo> &operands);

  bool Eval();

  static constexpr int kLookupTensor = 0;
  static constexpr int kValueTensor = 1;

  static constexpr int kOutputTensor = 0;

 private:
  const RunTimeOperandInfo *value_;
  const RunTimeOperandInfo *lookup_;

  RunTimeOperandInfo *output_;
};

}  // namespace nn
}  // namespace android

#endif  // FRAMEWORKS_ML_NN_EMBEDDING_LOOKUP_H
