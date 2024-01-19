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

#ifndef FRAMEWORKS_ML_NN_RNN_H
#define FRAMEWORKS_ML_NN_RNN_H

#ifndef HIFI_BUILD
#else
#include <vector>
#endif //HIFI_BUILD

#include "ActivationFunctor.h"

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
struct Shape;

class RNN {
 public:
  RNN(const android::hardware::neuralnetworks::V1_1::Operation &operation,
      std::vector<RunTimeOperandInfo> &operands);

  static bool Prepare(const android::hardware::neuralnetworks::V1_1::Operation &operation,
                      std::vector<RunTimeOperandInfo> &operands,
                      Shape *hiddenStateShape,
                      Shape *outputShape);
  bool Eval();

  static constexpr int kInputTensor = 0;
  static constexpr int kWeightsTensor = 1;  // Optional
  static constexpr int kRecurrentWeightsTensor = 2;
  static constexpr int kBiasTensor = 3;
  static constexpr int kHiddenStateInTensor = 4;
  static constexpr int kActivationParam = 5;

  static constexpr int kHiddenStateOutTensor = 0;
  static constexpr int kOutputTensor = 1;

 private:
  ActivationFn activation_;

  const RunTimeOperandInfo *input_;
  const RunTimeOperandInfo *weights_;
  const RunTimeOperandInfo *recurrent_weights_;
  const RunTimeOperandInfo *bias_;
  const RunTimeOperandInfo *hidden_state_in_;

  RunTimeOperandInfo *hidden_state_out_;
  RunTimeOperandInfo *output_;
};

}  // namespace nn
}  // namespace android

#endif  // FRAMEWORKS_ML_NN_RNN_H
