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
/*******************************************************************************
* Copyright (c) 2021 Cadence Design Systems, Inc.
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

#include "tensorflow/lite/kernels/internal/reference/sub.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace sub {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32_t input1_multiplier;
  int32_t input2_multiplier;
  int32_t output_multiplier;
  int output_shift;
  int left_shift;
  int32_t input1_offset;
  int32_t input2_offset;
  int32_t output_offset;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteSubParams* params,
                             const TfLiteTensor* input1,
                             const TfLiteTensor* input2, TfLiteTensor* output,
                             OpData* data) {
  data->requires_broadcast = !HaveSameShapes(input1, input2);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    // 8bit -> 8bit general quantized path, with general rescalings
    data->input1_offset = -input1->params.zero_point;
    data->input2_offset = -input2->params.zero_point;
    data->output_offset = output->params.zero_point;
    data->left_shift = 20;
    const float twice_max_input_scale =
        2 * std::max(input1->params.scale, input2->params.scale);
    const double real_input1_multiplier =
        static_cast<double>(input1->params.scale / twice_max_input_scale);
    const double real_input2_multiplier =
        static_cast<double>(input2->params.scale / twice_max_input_scale);
    const double real_output_multiplier =
        static_cast<double>(twice_max_input_scale /
                            ((1 << data->left_shift) * output->params.scale));

    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->input1_multiplier, &data->input1_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->input2_multiplier, &data->input2_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

  const TfLiteTensor* input1 = GetInput(context, node, kInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const TfLiteTensor* input2 = GetInput(context, node, kInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_STATUS(
      CalculateOpData(context, params, input1, input2, output, data));
  return kTfLiteOk;
}

void EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpData* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  if (data->requires_broadcast) {
    tflite::reference_ops::BroadcastSubSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  } else {
    tflite::reference_ops::SubWithActivation(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  }
}

TfLiteStatus EvalSubQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpData* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    tflite::ArithmeticParams op_params;
    op_params.left_shift = data->left_shift;
    op_params.input1_offset = data->input1_offset;
    op_params.input1_multiplier = data->input1_multiplier;
    op_params.input1_shift = data->input1_shift;
    op_params.input2_offset = data->input2_offset;
    op_params.input2_multiplier = data->input2_multiplier;
    op_params.input2_shift = data->input2_shift;
    op_params.output_offset = data->output_offset;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    bool need_broadcast = reference_ops::ProcessBroadcastShapes(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorShape(input2), &op_params);

    if (output->type == kTfLiteInt8) {
      if (need_broadcast) {
        tflite::reference_ops::BroadcastSubSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
#ifdef NNLIB_HIFI5
        int err;
        const RuntimeShape& input1_shape = tflite::micro::GetTensorShape(input1);
        const RuntimeShape& input2_shape = tflite::micro::GetTensorShape(input2);
        const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
        const int flat_size = MatchingElementsSize(input1_shape, input2_shape, output_shape);

        err = xa_nn_elm_sub_asym8sxasym8s_asym8s(tflite::micro::GetTensorData<int8_t>(output),
                                              op_params.output_offset,
                                              op_params.output_shift,
                                              op_params.output_multiplier,
                                              op_params.quantized_activation_min,
                                              op_params.quantized_activation_max,
                                              tflite::micro::GetTensorData<int8_t>(input1) ,
                                              op_params.input1_offset,
                                              op_params.input1_shift,
                                              op_params.input1_multiplier,
                                              tflite::micro::GetTensorData<int8_t>(input2),
                                              op_params.input2_offset,
                                              op_params.input2_shift,
                                              op_params.input2_multiplier,
                                              op_params.left_shift,
                                              flat_size);

        CHECK_ERR_HIFI_NNLIB_KER(err, "xa_nn_sub_asym8sxasym8s_asym8s failed");
#else
        tflite::reference_ops::Sub(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
#endif /* NNLIB_HIFI5 */
      }
    } else {
      if (need_broadcast) {
        tflite::reference_ops::BroadcastSubSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<uint8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<uint8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<uint8_t>(output));
      } else {
        tflite::reference_ops::Sub(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<uint8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<uint8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<uint8_t>(output));
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  if (output->type == kTfLiteFloat32) {
    EvalSub(context, node, params, &data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, EvalSubQuantized(context, node, params, &data,
                                                input1, input2, output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(output->type), output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace sub

TfLiteRegistration Register_SUB() {
  return {/*init=*/sub::Init,
          /*free=*/nullptr,
          /*prepare=*/sub::Prepare,
          /*invoke=*/sub::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
