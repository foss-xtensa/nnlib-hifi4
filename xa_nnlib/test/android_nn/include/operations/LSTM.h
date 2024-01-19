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

#ifndef FRAMEWORKS_ML_NN_LSTMCELL_H
#define FRAMEWORKS_ML_NN_LSTMCELL_H

#ifndef HIFI_BUILD
#include "ActivationFunctor.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#else
// Possible fused activation functions.
// TODO(aselle): rename to TfLiteActivation
typedef enum {
  lstm_kTfLiteActNone = 0,
  lstm_kTfLiteActRelu,
  lstm_kTfLiteActRelu1,
  lstm_kTfLiteActRelu6,
  lstm_kTfLiteActTanh,
  lstm_kTfLiteActSignBit,
  lstm_kTfLiteActSigmoid,
} lstm_TfLiteFusedActivation;
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

struct LSTMParams {
#ifndef HIFI_BUILD
  TfLiteFusedActivation activation_;
#else
  lstm_TfLiteFusedActivation activation_;
#endif //HiFi_BUILD
  float cell_clip_;
  float proj_clip_;
};

struct RunTimeOperandInfo;
struct Shape;

class LSTMCell {
 public:
  LSTMCell(const android::hardware::neuralnetworks::V1_1::Operation &operation,
           std::vector<RunTimeOperandInfo> &operands);

  static bool Prepare(const android::hardware::neuralnetworks::V1_1::Operation &operation,
                      std::vector<RunTimeOperandInfo> &operands,
                      Shape *scratchShape,
                      Shape *outputStateShape,
                      Shape *cellStateShape,
                      Shape *outputShape);
  bool Eval();

  // Input Tensors of size {n_batch, n_input}
  static constexpr int kInputTensor = 0;

  // Input weight tensors of size: {n_cell, n_input}
  static constexpr int kInputToInputWeightsTensor = 1;  // Optional
  static constexpr int kInputToForgetWeightsTensor = 2;
  static constexpr int kInputToCellWeightsTensor = 3;
  static constexpr int kInputToOutputWeightsTensor = 4;

  // Recurrent weight tensors of size {n_cell, n_output}
  static constexpr int kRecurrentToInputWeightsTensor = 5;  // Optional
  static constexpr int kRecurrentToForgetWeightsTensor = 6;
  static constexpr int kRecurrentToCellWeightsTensor = 7;
  static constexpr int kRecurrentToOutputWeightsTensor = 8;

  // Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
  static constexpr int kCellToInputWeightsTensor = 9;    // Optional
  static constexpr int kCellToForgetWeightsTensor = 10;  // Optional
  static constexpr int kCellToOutputWeightsTensor = 11;  // Optional

  // Gates bias tensors of size {n_cell}
  static constexpr int kInputGateBiasTensor = 12;  // Optional
  static constexpr int kForgetGateBiasTensor = 13;
  static constexpr int kCellGateBiasTensor = 14;
  static constexpr int kOutputGateBiasTensor = 15;

  // Projection weight tensor of size {n_output, n_cell}
  static constexpr int kProjectionWeightsTensor = 16;  // Optional
  // Projection bias tensor of size {n_output}
  static constexpr int kProjectionBiasTensor = 17;  // Optional

  static constexpr int kOutputStateInTensor = 18;
  static constexpr int kCellStateInTensor = 19;

  static constexpr int kActivationParam = 20;
  static constexpr int kCellClipParam = 21;
  static constexpr int kProjClipParam = 22;

  // Output tensors.
  static constexpr int kScratchBufferTensor = 0;
  static constexpr int kOutputStateOutTensor = 1;
  static constexpr int kCellStateOutTensor = 2;
  static constexpr int kOutputTensor = 3;

 private:
  static bool CheckInputTensorDimensions(
      const android::hardware::neuralnetworks::V1_1::Operation &operation,
      std::vector<RunTimeOperandInfo> &operands, uint32_t n_input,
      uint32_t n_output, uint32_t n_cell);
  LSTMParams params_;

  const RunTimeOperandInfo *input_;

  const RunTimeOperandInfo *input_to_input_weights_;
  const RunTimeOperandInfo *input_to_forget_weights_;
  const RunTimeOperandInfo *input_to_cell_weights_;
  const RunTimeOperandInfo *input_to_output_weights_;

  const RunTimeOperandInfo *recurrent_to_input_weights_;
  const RunTimeOperandInfo *recurrent_to_forget_weights_;
  const RunTimeOperandInfo *recurrent_to_cell_weights_;
  const RunTimeOperandInfo *recurrent_to_output_weights_;

  const RunTimeOperandInfo *cell_to_input_weights_;
  const RunTimeOperandInfo *cell_to_forget_weights_;
  const RunTimeOperandInfo *cell_to_output_weights_;

  const RunTimeOperandInfo *input_gate_bias_;
  const RunTimeOperandInfo *forget_gate_bias_;
  const RunTimeOperandInfo *cell_bias_;
  const RunTimeOperandInfo *output_gate_bias_;

  const RunTimeOperandInfo *projection_weights_;
  const RunTimeOperandInfo *projection_bias_;

  const RunTimeOperandInfo *output_state_in_;
  const RunTimeOperandInfo *cell_state_in_;

  RunTimeOperandInfo *output_state_out_;
  RunTimeOperandInfo *cell_state_out_;
  RunTimeOperandInfo *output_;

  RunTimeOperandInfo *scratch_buffer_;
};

}  // namespace nn
}  // namespace android

#endif  // FRAMEWORKS_ML_NN_LSTMCELL_H
