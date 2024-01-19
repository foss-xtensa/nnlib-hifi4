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

#ifndef FRAMEWORKS_ML_NN_SVDF_H
#define FRAMEWORKS_ML_NN_SVDF_H

#ifndef HIFI_BUILD
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#else
// Possible fused activation functions.
// TODO(aselle): rename to TfLiteActivation
typedef enum {
  svdf_kTfLiteActNone = 0,
  svdf_kTfLiteActRelu,
  svdf_kTfLiteActRelu1,
  svdf_kTfLiteActRelu6,
  svdf_kTfLiteActTanh,
  svdf_kTfLiteActSignBit,
  svdf_kTfLiteActSigmoid,
} svdf_TfLiteFusedActivation;
#endif //HiFi_BUILD

#include <algorithm>
#include <cmath>
#ifndef HIFI_BUILD
#else
#include <vector>
#endif //HIFI_BUILD

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

struct SVDFParams {
    int rank_;
#ifndef HIFI_BUILD
    TfLiteFusedActivation activation_;
#else
    svdf_TfLiteFusedActivation activation_;
#endif //HiFi_BUILD
};

struct RunTimeOperandInfo;
struct Shape;

class SVDF {
public:
    SVDF(const android::hardware::neuralnetworks::V1_1::Operation &operation,
         std::vector<RunTimeOperandInfo>& operands);

    static bool Prepare(
        const hardware::neuralnetworks::V1_1::Operation &operation,
        std::vector<RunTimeOperandInfo> &operands, Shape *stateShape,
        Shape *outputShape);
    bool Eval();

    static constexpr int kInputTensor = 0;
    static constexpr int kWeightsFeatureTensor = 1;
    static constexpr int kWeightsTimeTensor = 2;
    static constexpr int kBiasTensor = 3;  // Optional
    static constexpr int kStateInTensor = 4;
    static constexpr int kRankParam = 5;
    static constexpr int kActivationParam = 6;

    static constexpr int kStateOutTensor = 0;
    static constexpr int kOutputTensor = 1;

private:
    SVDFParams params_;

    const RunTimeOperandInfo *input_;
    const RunTimeOperandInfo *weights_feature_;
    const RunTimeOperandInfo *weights_time_;
    const RunTimeOperandInfo *bias_;
    const RunTimeOperandInfo *state_in_;

    RunTimeOperandInfo *state_out_;
    RunTimeOperandInfo *output_;
};

}  // namespace nn
}  // namespace android

#endif
