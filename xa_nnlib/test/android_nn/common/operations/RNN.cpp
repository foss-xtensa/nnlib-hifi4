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

#include "RNN.h"

#ifndef HIFI_BUILD
#else
#include "xa_nnlib_ann_api.h"
#endif //HiFi_NNLIB_OPT

#include "CpuExecutor.h"

#ifndef HIFI_BUILD
#include "HalInterfaces.h"
#endif //HiFi_BUILD


namespace android {
namespace nn {

RNN::RNN(const Operation& operation,
         std::vector<RunTimeOperandInfo>& operands) {
  input_ = GetInput(operation, operands, kInputTensor);
  weights_ = GetInput(operation, operands, kWeightsTensor);
  recurrent_weights_ = GetInput(operation, operands, kRecurrentWeightsTensor);
  hidden_state_in_ = GetInput(operation, operands, kHiddenStateInTensor);
  bias_ = GetInput(operation, operands, kBiasTensor);

  activation_ = static_cast<ActivationFn>(
      getScalarData<int32_t>(operands[operation.inputs[kActivationParam]]));

  hidden_state_out_ = GetOutput(operation, operands, kHiddenStateOutTensor);
  output_ = GetOutput(operation, operands, kOutputTensor);
}

bool RNN::Prepare(const Operation &operation,
                  std::vector<RunTimeOperandInfo> &operands,
                  Shape *hiddenStateShape,
                  Shape *outputShape) {
  // Check we have all the inputs and outputs we need.
  const int num_inputs = NumInputsWithValues(operation, operands);
  NN_CHECK(num_inputs == 5 || num_inputs == 6);
  NN_CHECK_EQ(NumOutputs(operation), 2);

  const RunTimeOperandInfo *input =
      GetInput(operation, operands, kInputTensor);
  const RunTimeOperandInfo *input_weights =
      GetInput(operation, operands, kWeightsTensor);
  const RunTimeOperandInfo *recurrent_weights =
      GetInput(operation, operands, kRecurrentWeightsTensor);
  const RunTimeOperandInfo *bias =
      GetInput(operation, operands, kBiasTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const uint32_t batch_size = SizeOfDimension(input, 0);
  const uint32_t num_units = SizeOfDimension(input_weights, 0);
  NN_CHECK_EQ(SizeOfDimension(input, 1), SizeOfDimension(input_weights, 1));
  NN_CHECK_EQ(SizeOfDimension(input_weights, 0), SizeOfDimension(bias, 0));
  NN_CHECK_EQ(SizeOfDimension(recurrent_weights, 0), SizeOfDimension(bias, 0));
  NN_CHECK_EQ(SizeOfDimension(recurrent_weights, 1), SizeOfDimension(bias, 0));

  const Shape &inputShape = input->shape();

  // Resize state.
  hiddenStateShape->type = inputShape.type;
#ifndef HIFI_BUILD
  hiddenStateShape->dimensions = { batch_size, num_units };
#else
  hiddenStateShape->dimensions.clear();
  hiddenStateShape->dimensions.push_back(batch_size);
  hiddenStateShape->dimensions.push_back(num_units);
#endif

  // Resize output.
  outputShape->type = inputShape.type;
#ifndef HIFI_BUILD
  outputShape->dimensions = { batch_size, num_units };
#else
  outputShape->dimensions.clear();
  outputShape->dimensions.push_back(batch_size);
  outputShape->dimensions.push_back(num_units);
#endif
#ifndef HIFI_WARNINGS
  (void)recurrent_weights;
  (void)bias;
  (void)num_inputs;
#endif
  return true;
}

bool RNN::Eval() {
  const float* bias_ptr = reinterpret_cast<float*>(bias_->buffer);

  const uint32_t batch_size = input_->shape().dimensions[0];
  const uint32_t num_units = weights_->shape().dimensions[0];
  const uint32_t input_size = input_->shape().dimensions[1];
#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
  const uint32_t input_weights_stride = weights_->shape().dimensions[1];
  const uint32_t recurrent_weights_stride =
      recurrent_weights_->shape().dimensions[1];
#endif

  // For each batch
  for (uint32_t b = 0; b < batch_size; b++) {
    // Initialize the pointer to input, output and bias.
    const float* input_ptr_batch =
        reinterpret_cast<float*>(input_->buffer) + b * input_size;
    const float* hidden_state_in_ptr_batch =
        reinterpret_cast<float*>(hidden_state_in_->buffer) + b * num_units;
    float* output_ptr_batch =
        reinterpret_cast<float*>(output_->buffer) + b * num_units;
    float* hidden_state_out_ptr_batch =
        reinterpret_cast<float*>(hidden_state_out_->buffer) + b * num_units;

    // Initialize input_weights and recurrent_weights.
    const float* input_weights_ptr = reinterpret_cast<float*>(weights_->buffer);
    const float* recurrent_weights_ptr =
        reinterpret_cast<float*>(recurrent_weights_->buffer);

#if !HIFI_VFPU || !defined HIFI_NNLIB_OPT 
    // Output = bias
    for (uint32_t o = 0; o < num_units; o++) {
      output_ptr_batch[o] = bias_ptr[o];
    }

    // Output += input * input_weights
    for (uint32_t o = 0; o < num_units; o++) {
      for (uint32_t i = 0; i < input_size; i++) {
        output_ptr_batch[o] += input_ptr_batch[i] * input_weights_ptr[i];
      }
      input_weights_ptr += input_weights_stride;
    }

    // Output += recurrent_weights * hidden_state
    for (uint32_t o = 0; o < num_units; o++) {
      for (uint32_t h = 0; h < num_units; h++) {
        output_ptr_batch[o] +=
            hidden_state_in_ptr_batch[h] * recurrent_weights_ptr[h];
      }
      recurrent_weights_ptr += recurrent_weights_stride;
    }

    // Output = activation(Output) and update hidden_state
    for (uint32_t o = 0; o < num_units; o++) {
      output_ptr_batch[o] =
          (ActivationFunctor(activation_))(output_ptr_batch[o]);
      hidden_state_out_ptr_batch[o] = output_ptr_batch[o];
    }
#else
    int32_t ret;
    ret = xa_nn_matXvec_f32xf32_f32(output_ptr_batch,
        input_weights_ptr, recurrent_weights_ptr, input_ptr_batch, hidden_state_in_ptr_batch,
        bias_ptr, num_units, input_size, num_units, input_size, num_units);
    
    switch(activation_) {
      case kActivationNone:
        ret = 0;
        break;
      case kActivationRelu:
        ret = xa_nn_vec_activation_min_max_f32_f32(output_ptr_batch, output_ptr_batch,
            0.0f, std::numeric_limits<float>::max(), num_units);
        break;
      case kActivationRelu6:
        ret = xa_nn_vec_activation_min_max_f32_f32(output_ptr_batch, output_ptr_batch,
            0.0f, 6.0f, num_units);
        break;
      case kActivationTanh:
        ret = xa_nn_vec_tanh_f32_f32(output_ptr_batch, output_ptr_batch,
            num_units);
        break;
      case kActivationSigmoid:
        ret = xa_nn_vec_sigmoid_f32_f32(output_ptr_batch, output_ptr_batch,
            num_units);
        break;
      default:
        ret = -1;
    }
    if(ret != 0)
      return false;
    memcpy(hidden_state_out_ptr_batch, output_ptr_batch, num_units*sizeof(float));
#endif
  }

  return true;
}

}  // namespace nn
}  // namespace android
