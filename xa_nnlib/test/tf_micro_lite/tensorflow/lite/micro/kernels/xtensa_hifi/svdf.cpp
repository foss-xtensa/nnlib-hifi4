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
/*
* Copyright (c) 2019-2020 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
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
*/
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

#include <math.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

struct OpData {
  int32_t effective_scale_1_a;
  int32_t effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
  int scratch_tensor_index;
  int scratch_output_tensor_index;

  // Cached tensor zero point values for quantized operations.
  int input_zero_point;
  int output_zero_point;
};

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

/**
 * This version of SVDF is specific to TFLite Micro. It contains the following
 * differences between the TFLite version:
 *
 * 1.) Scratch tensor allocation - scratch tensors must be known ahead of time
 * for the Micro interpreter.
 * 2.) Output dimensions - the TFLite version determines output size and runtime
 * and resizes the output tensor. Micro runtime does not support tensor
 * resizing.
 */

static inline TfLiteStatus ApplyTimeWeightsBiasAndActivation(
    TfLiteContext* context, int batch_size, int memory_size, int num_filters,
    int num_units, int rank, const float* const __restrict__ weights_time_ptr,
    const float* const __restrict__ bias_ptr, TfLiteFusedActivation activation,
    float* const __restrict__ state_ptr, float* const __restrict__ scratch_ptr,
    float* const __restrict__ output_ptr) {
  // Compute matmul(activation_state, weights_time).
  for (int b = 0; b < batch_size; ++b) {
    // Perform batched vector dot product:
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;
    const float* vector1_ptr = weights_time_ptr;
    const float* vector2_ptr = state_ptr + b * memory_size * num_filters;
    for (int i = 0; i < num_filters; ++i) {
      *scratch_ptr_batch = 0.f;
      for (int j = 0; j < memory_size; ++j) {
        *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
      }
      scratch_ptr_batch++;
    }
  }

  // Initialize output with bias if provided.
  if (bias_ptr) {
    // VectorBatchVectorAssign
    for (int i = 0; i < batch_size; ++i) {
      float* output_data = output_ptr + i * num_units;
      const float* bias_data = bias_ptr;
      for (int j = 0; j < num_units; ++j) {
        *output_data++ = *bias_data++;
      }
    }
  } else {
    float* output_data = output_ptr;
    for (int i = 0; i < batch_size * num_units; ++i) {
      *output_data++ = 0.0f;
    }
  }

  // Reduction sum.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output_ptr + b * num_units;
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;

    // Reduction sum vector
    for (int i = 0; i < num_units; ++i) {
      for (int j = 0; j < rank; j++) {
        output_ptr_batch[i] += *scratch_ptr_batch++;
      }
    }
  }

  // Apply activation.
  for (int b = 0; b < batch_size; ++b) {
    float* output_ptr_batch = output_ptr + b * num_units;
    for (int i = 0; i < num_units; ++i) {
      *output_ptr_batch =
          tflite::ops::micro::ActivationValFloat(activation, *output_ptr_batch);
      ++output_ptr_batch;
    }
  }
  return kTfLiteOk;
}

inline TfLiteStatus EvalFloatSVDF(
    TfLiteContext* context, TfLiteNode* node, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* weights_feature,
    const TfLiteEvalTensor* weights_time, const TfLiteEvalTensor* bias,
    const TfLiteSVDFParams* params, int scratch_tensor_index,
    TfLiteEvalTensor* activation_state, TfLiteEvalTensor* output) {
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  const float* weights_feature_ptr =
      tflite::micro::GetTensorData<float>(weights_feature);
  const float* weights_time_ptr =
      tflite::micro::GetTensorData<float>(weights_time);
  const float* bias_ptr = tflite::micro::GetTensorData<float>(bias);
  const float* input_ptr = tflite::micro::GetTensorData<float>(input);

  float* state_ptr = tflite::micro::GetTensorData<float>(activation_state);

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  float* scratch_ptr = static_cast<float*>(
      context->GetScratchBuffer(context, scratch_tensor_index));

  float* output_ptr = tflite::micro::GetTensorData<float>(output);

  // Left shift the activation_state.
  {
    float* new_state_start = state_ptr;
    const float* old_state_start = state_ptr + 1;
    for (int i = 0; i < batch_size * num_filters * memory_size - 1; i++) {
      new_state_start[i] = old_state_start[i];
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Compute conv1d(inputs, weights_feature).
  // The activation_state's rightmost column is used to save current cycle
  // activation. This is achieved by starting at state_ptr[memory_size - 1] and
  // having the stride equal to memory_size.

  // Perform batched matrix vector multiply operation:
  {
    const float* matrix = weights_feature_ptr;
    const float* vector = input_ptr;
    float* result = &state_ptr[memory_size - 1];
    float* result_in_batch = result;

#if HIFI_VFPU && !defined NNLIB_HIFI5  
    float* out_scratch = scratch_ptr;
    float* bias_scratch = output_ptr;
    for (int i = 0; i < num_units; i++) bias_scratch[i] = 0.0f;

    int err = 0;
    for (int i = 0; i < batch_size; i++) {
      /* We are using output buffer for bias (it is needed by NNLib kernel,
      so only num_units size is guaranteed, so introduced rank loop and
      calling matXvec for num_units rows */
      for (int j = 0; j < rank; j++) {
        err = xa_nn_matXvec_f32xf32_f32(
            &out_scratch[j * num_units], &matrix[j * input_size * num_units],
            NULL, &vector[i * input_size], NULL, bias_scratch, num_units,
                                        input_size, 0, input_size, 0);
        CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_matXvec_f32xf32_f32 failed");
      }
      for (int j = 0; j < num_filters; ++j) {
        *result_in_batch = out_scratch[j];
        result_in_batch += memory_size;
      }
    }
#else
    for (int i = 0; i < batch_size; ++i) {
      const float* matrix_ptr = matrix;
      for (int j = 0; j < num_filters; ++j) {
        float dot_prod = 0.0f;
        const float* vector_in_batch = vector + i * input_size;
        for (int k = 0; k < input_size; ++k) {
          dot_prod += *matrix_ptr++ * *vector_in_batch++;
        }
        *result_in_batch = dot_prod;
        result_in_batch += memory_size;
      }
    }
#endif /* HIFI_VFPU */
  }

  return ApplyTimeWeightsBiasAndActivation(
      context, batch_size, memory_size, num_filters, num_units, rank,
      weights_time_ptr, bias_ptr, params->activation, state_ptr, scratch_ptr,
      output_ptr);
}

TfLiteStatus EvalIntegerSVDF(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteEvalTensor* input_tensor,
                             const TfLiteEvalTensor* weights_feature_tensor,
                             const TfLiteEvalTensor* weights_time_tensor,
                             const TfLiteEvalTensor* bias_tensor,
                             const TfLiteSVDFParams* params,
                             TfLiteEvalTensor* activation_state_tensor,
                             TfLiteEvalTensor* output_tensor,
                             const OpData& data) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

#ifdef NNLIB_HIFI5
  // Shift states.
  int16_t* const state_ptr =
      tflite::micro::GetTensorData<int16_t>(activation_state_tensor);

  // Left shift the activation_state.
  {
    ae_int16x8* pDst = reinterpret_cast<ae_int16x8*>(state_ptr);
    ae_int16x8* pSrc = reinterpret_cast<ae_int16x8*>(state_ptr + 1);
    ae_int16x4 d, d1;
    ae_valignx2 valign1 = AE_LA128_PP(pSrc);
    ae_valignx2 valign2 = AE_ZALIGN128();

    int loopcnt = (n_batch * n_filter * n_memory - 1);

    for (int cnt = 0; cnt < (loopcnt >> 3); cnt++) {
      AE_LA16X4X2_IP(d, d1, valign1, pSrc);
      AE_SA16X4X2_IP(d, d1, valign2, pDst);
    }
    AE_SA128POS_FP(valign2, pDst);

    for (int cnt1 = 0; cnt1 < (loopcnt & 0x7); cnt1++) {
      AE_L16_IP(d, (ae_int16*)pSrc, 2);
      AE_S16_0_IP(d, (ae_int16*)pDst, 2);
    }
  }
#else
  int32_t* scratch_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));
  int32_t* scratch_output_tensor = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_output_tensor_index));

  // Shift states.
  int16_t* const state_ptr =
      tflite::micro::GetTensorData<int16_t>(activation_state_tensor);

  // Left shift the activation_state.
  {
    int16_t* new_state_start = state_ptr;
    const int16_t* old_state_start = state_ptr + 1;
    const int16_t* old_state_end = state_ptr + n_batch * n_filter * n_memory;
    while (old_state_start != old_state_end) {
      *new_state_start++ = *old_state_start++;
    }
  }
#endif

// Note: no need to clear the latest activation, matmul is not accumulative.

// Feature matmul.
#ifndef NNLIB_HIFI5
  {
    int16_t* state =
        tflite::micro::GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = tflite::micro::GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        tflite::micro::GetTensorData<int8_t>(weights_feature_tensor);
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_ptr = weight_feature;
      for (int r = 0; r < n_filter; r++) {
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod +=
              *matrix_ptr++ * (*vector_in_batch++ - data.input_zero_point);
        }
        dot_prod = MultiplyByQuantizedMultiplier(
            dot_prod, data.effective_scale_1_a, data.effective_scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod_56.
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }
#else
  {
    int16_t* state =
        tflite::micro::GetTensorData<int16_t>(activation_state_tensor);
    const int8_t* input = tflite::micro::GetTensorData<int8_t>(input_tensor);
    const int8_t* weight_feature =
        tflite::micro::GetTensorData<int8_t>(weights_feature_tensor);
    int16_t* result_in_batch = state + (n_memory - 1);
    int err = 0;

    for (int b = 0; b < n_batch; b++) {
      err = xa_nn_matXvec_out_stride_sym8sxasym8s_16(
          &result_in_batch[b * n_filter * n_memory], weight_feature,
          &input[b * n_input], NULL, n_filter, n_input, n_input, n_memory,
          -data.input_zero_point, (data.effective_scale_1_a),
          data.effective_scale_1_b);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_vec_matXvec_sym8sxasym8s_16 failed");
    }
  }
#endif

#ifndef NNLIB_HIFI5
  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Perform batched vector dot product:
      const int16_t* vector1_ptr =
          tflite::micro::GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr =
          tflite::micro::GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;

      for (int i = 0; i < n_filter; i++) {
        *scratch_ptr_batch = 0;
        for (int j = 0; j < n_memory; j++) {
          *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
        }
        scratch_ptr_batch++;
      }
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Add bias.
    if (bias_tensor) {
      // Vector batch assign:
      const int32_t* bias_data =
          tflite::micro::GetTensorData<int32_t>(bias_tensor);
      for (int i = 0; i < n_batch; ++i) {
        int32_t* output_ptr = scratch_output_tensor + i * n_unit;
        const int32_t* bias_ptr = bias_data;
        for (int j = 0; j < n_unit; ++j) {
          *output_ptr++ = *bias_ptr++;
        }
      }
    } else {
      int32_t* output_ptr = scratch_output_tensor;
      for (int i = 0; i < n_batch * n_unit; ++i) {
        *output_ptr++ = 0;
      }
    }

    // Reduce.
    for (int b = 0; b < n_batch; ++b) {
      int32_t* output_temp_ptr = scratch_output_tensor + b * n_unit;
      int32_t* scratch_ptr_batch = scratch_tensor + b * n_filter;

      // Reduction sum vector
      for (int i = 0; i < n_unit; ++i) {
        for (int j = 0; j < n_rank; ++j) {
          output_temp_ptr[i] += *scratch_ptr_batch++;
        }
      }
    }

    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = scratch_output_tensor[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, data.effective_scale_2_a,
                                                 data.effective_scale_2_b);
      int32_t x3 = x2 + data.output_zero_point;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      tflite::micro::GetTensorData<int8_t>(output_tensor)[i] =
          static_cast<int8_t>(x4);
    }
  }
#else
  {
    for (int b = 0; b < n_batch; ++b) {
      int8_t* output_ptr =
          tflite::micro::GetTensorData<int8_t>(output_tensor) + b * n_unit;

      const int16_t* vector1_ptr =
          tflite::micro::GetTensorData<int16_t>(weights_time_tensor);
      const int16_t* vector2_ptr =
          tflite::micro::GetTensorData<int16_t>(activation_state_tensor) +
          b * n_memory * n_filter;
      int err = 0;
      const int32_t* bias_ptr =
          tflite::micro::GetTensorData<int32_t>(bias_tensor);
      err = xa_nn_dot_prod_16x16_asym8s(
          output_ptr, vector1_ptr, vector2_ptr, bias_ptr, n_memory * n_rank,
          (data.effective_scale_2_a), data.effective_scale_2_b,
          data.output_zero_point, n_unit);
      CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_dot_prod_16x16_asym8s failed");
    }
  }
#endif
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto* params = static_cast<const TfLiteSVDFParams*>(node->builtin_data);

  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  TF_LITE_ENSURE(context, weights_feature != nullptr);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  TF_LITE_ENSURE(context, weights_time != nullptr);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  const TfLiteTensor* activation_state =
      GetInput(context, node, kInputActivationStateTensor);
  TF_LITE_ENSURE(context, activation_state != nullptr);

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Validate Input Tensor:
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  // Validate Tensor Output:
  // [0] = float/int8_t, {2, batch_size, num_units}
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 2);
  TF_LITE_ENSURE_EQ(context, output->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, output->dims->data[1], num_units);

  // Validate Weights Feature Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_feature), 2);
  TF_LITE_ENSURE_EQ(context, weights_feature->dims->data[1], input_size);

  // Validate Weights Time Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_time), 2);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[1], memory_size);

  // Validate Optional Bias Input Tensor:
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);
  // Since is_variable is not part of TFLiteEvalTensor, check is_variable here.
  TF_LITE_ENSURE_EQ(context, activation_state->is_variable, true);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteInt16);
    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteInt16);
    if (bias != nullptr) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
    }

    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

    const double effective_scale_1 = static_cast<double>(
        input->params.scale * weights_feature->params.scale /
        activation_state->params.scale);
    const double effective_scale_2 =
        static_cast<double>(activation_state->params.scale *
                            weights_time->params.scale / output->params.scale);

    // TODO(b/162018098): Use TF_LITE_ENSURE_NEAR when it is ready.
    TF_LITE_ENSURE(
        context,
        std::abs(static_cast<double>(bias->params.scale) -
                 static_cast<double>(activation_state->params.scale *
                                     weights_time->params.scale)) < 1e-5);

    QuantizeMultiplier(effective_scale_1, &(data->effective_scale_1_a),
                       &(data->effective_scale_1_b));
    QuantizeMultiplier(effective_scale_2, &(data->effective_scale_2_a),
                       &(data->effective_scale_2_b));

    data->input_zero_point = input->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, batch_size * num_filters * sizeof(int32_t),
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);

    const TfLiteStatus scratch_output_status =
        context->RequestScratchBufferInArena(
            context, batch_size * num_units * sizeof(int32_t),
            &(data->scratch_output_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_output_status);
  } else {
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteFloat32);
    if (bias != nullptr) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteFloat32);
    }
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);

    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, batch_size * num_filters * sizeof(float),
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      return EvalFloatSVDF(context, node, input, weights_feature, weights_time,
                           bias, params, data.scratch_tensor_index,
                           activation_state, output);
      break;
    }

    case kTfLiteInt8: {
      EvalIntegerSVDF(context, node, input, weights_feature, weights_time, bias,
                      params, activation_state, output, data);
      return kTfLiteOk;
      break;
    }

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_SVDF() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
