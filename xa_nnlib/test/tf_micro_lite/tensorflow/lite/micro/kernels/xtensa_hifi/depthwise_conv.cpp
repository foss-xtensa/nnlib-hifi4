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
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct OpData {
  TfLitePaddingValues padding;

  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  int scratch_tensor_index;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteDepthwiseConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    TF_LITE_ENSURE(context, filter != nullptr);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];

    return tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift), num_channels);
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

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8_t inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynimically allocate per-channel quantization parameters.
  data->per_channel_output_multiplier =
      reinterpret_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      reinterpret_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  // Calculate scratch memory requirements and request scratch buffer
  if ((input->type == kTfLiteInt8) ||
      (input->type == kTfLiteUInt8) ||
      (input->type == kTfLiteFloat32)) {
    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& output_shape = GetTensorShape(output);

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const int depth_multiplier = params->depth_multiplier;
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;

    int input_precision;

    if (input->type == kTfLiteInt8) {
      input_precision = -4; //PREC_ASYM8S;
    } else if (input->type == kTfLiteUInt8) {
      input_precision = PREC_ASYM8;
    } else {
      input_precision = PREC_F32;
    }

    int required_scratch = xa_nn_conv2d_depthwise_getsize(
        input_height, input_width, input_depth, filter_height, filter_width,
        depth_multiplier, stride_width, stride_height, pad_width, pad_height,
        output_height, output_width, input_precision, 0 /* NHWC */);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(
          context,
          "DepthwiseConv: xa_nn_conv2d_depthwise_getsize failed");
      return kTfLiteError;
    }

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, required_scratch,
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteDepthwiseConvParams* params, const OpData& data,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

#if HIFI_VFPU
  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const float *input_data, *filter_data, *bias_data;
    float* output_data;
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);

    input_data = tflite::micro::GetTensorData<float>(input);
    filter_data = tflite::micro::GetTensorData<float>(filter);
    bias_data = tflite::micro::GetTensorData<float>(bias);
    output_data = tflite::micro::GetTensorData<float>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int pad_width = data.padding.width;
    const int pad_height = data.padding.height;
    const int depth_multiplier = params->depth_multiplier;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    int32_t err, input_data_format = 0, output_data_format = 0;
    uint8_t* p_scratch;
    float* p_filter;

    p_scratch = static_cast<uint8_t*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    p_filter = const_cast<float*>(filter_data);

    for (int i = 0; i < batches; i++) {
      err = xa_nn_conv2d_depthwise_f32(
          &output_data[i * output_height * output_width * output_depth],
          p_filter, &input_data[i * input_height * input_width * input_depth],
          bias_data, input_height, input_width, input_depth, filter_height,
          filter_width, depth_multiplier, stride_width, stride_height,
          pad_width, pad_height, output_height, output_width, input_data_format,
          output_data_format, static_cast<void*>(p_scratch));

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "DepthwiseConvFloat: xa_nn_conv2d_depthwise_f32 failed");
    }

    int out_length = batches * output_height * output_width * output_depth;
    err = xa_nn_vec_activation_min_max_f32_f32(
        output_data, output_data, output_activation_min, output_activation_max,
        out_length);

    CHECK_ERR_HIFI_NNLIB_KER(
        err, "DepthwiseConvFloat: xa_nn_vec_activation_min_max_f32_f32 failed");

    return kTfLiteOk;
  }
#endif /* HIFI_VFPU */

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  tflite::reference_ops::DepthwiseConv(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteDepthwiseConvParams* params,
                                     const OpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output) {
  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data.padding.width;
  op_params.padding_values.height = data.padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.depth_multiplier = params->depth_multiplier;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = data.output_zero_point;
  // TODO(b/130439627): Use calculated value for clamping.
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  // If dilation is not required use the optimized NN Library kernel.
  // Otherwise call the reference implementation.
  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const int8_t *input_data, *filter_data;
    const int32_t* bias_data;
    int8_t* output_data;
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);

    input_data = tflite::micro::GetTensorData<int8_t>(input);
    filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    output_data = tflite::micro::GetTensorData<int8_t>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int pad_width = data.padding.width;
    const int pad_height = data.padding.height;
    const int depth_multiplier = params->depth_multiplier;
    const int32_t output_activation_min = data.output_activation_min;
    const int32_t output_activation_max = data.output_activation_max;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    int32_t err, i, input_data_format = 0, output_data_format = 0;
    uint8_t* p_scratch;

    p_scratch = static_cast<uint8_t*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    for (i = 0; i < batches; i++) {
      err = xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
          &output_data[i * output_height * output_width * output_depth],
          filter_data,
          &input_data[i * input_height * input_width * input_depth], bias_data,
          input_height, input_width, input_depth, filter_height, filter_width,
          depth_multiplier, stride_width, stride_height, pad_width, pad_height,
          output_height, output_width, op_params.input_offset,
          data.per_channel_output_multiplier, data.per_channel_output_shift,
          op_params.output_offset, input_data_format, output_data_format,
          p_scratch);

      CHECK_ERR_HIFI_NNLIB_KER(err,
                               "DepthwiseConvSym8PerChannel: "
                               "xa_nn_conv2d_depthwise_asym8xasym8 failed");
    }

    int out_length = batches * output_height * output_width * output_depth;
    err = xa_nn_vec_activation_min_max_8_8(output_data, output_data,
                                           output_activation_min,
                                           output_activation_max, out_length);

    CHECK_ERR_HIFI_NNLIB_KER(
        err,
        "DepthwiseConvSym8PerChannel: xa_nn_vec_activation_min_max_8_8 "
        "failed");
    return kTfLiteOk;
  }

  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, data.per_channel_output_multiplier,
      data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDepthwiseConvParams* params,
                           const OpData& data, const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output) {
  const int32_t input_offset = -data.input_zero_point;
  const int32_t filter_offset = -data.filter_zero_point;
  const int32_t output_offset = data.output_zero_point;

  if ((params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    const uint8_t *input_data, *filter_data;
    const int32_t* bias_data;
    uint8_t* output_data;
    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const RuntimeShape& bias_shape = tflite::micro::GetTensorShape(bias);

    input_data = tflite::micro::GetTensorData<uint8_t>(input);
    filter_data = tflite::micro::GetTensorData<uint8_t>(filter);
    bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    output_data = tflite::micro::GetTensorData<uint8_t>(output);

    const int stride_width = params->stride_width;
    const int stride_height = params->stride_height;
    const int pad_width = data.padding.width;
    const int pad_height = data.padding.height;
    const int depth_multiplier = params->depth_multiplier;
    const int32_t output_activation_min = data.output_activation_min;
    const int32_t output_activation_max = data.output_activation_max;
    const int32_t output_multiplier = data.output_multiplier;
    // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
    const int output_shift = -data.output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    int32_t err, i, input_data_format = 0, output_data_format = 0;

    uint8_t* p_scratch = static_cast<uint8_t*>(
        context->GetScratchBuffer(context, data.scratch_tensor_index));

    uint8_t* p_filter = const_cast<uint8_t*>(filter_data);

    for (i = 0; i < batches; i++) {
      err = xa_nn_conv2d_depthwise_asym8xasym8(
          &output_data[i * output_height * output_width * output_depth],
          p_filter,  // filter_data,
          &input_data[i * input_height * input_width * input_depth], bias_data,
          input_height, input_width, input_depth, filter_height, filter_width,
          depth_multiplier, stride_width, stride_height, pad_width, pad_height,
          output_height, output_width, input_offset, filter_offset,
          output_multiplier, output_shift, output_offset, input_data_format,
          output_data_format, static_cast<void*>(p_scratch));

      CHECK_ERR_HIFI_NNLIB_KER(
          err, "DepthwiseConvAsym8: xa_nn_conv2d_depthwise_asym8xasym8 failed");
    }

    int out_length = batches * output_height * output_width * output_depth;
    err = xa_nn_vec_activation_min_max_asym8_asym8(
        output_data, output_data, output_activation_min, output_activation_max,
        out_length);

    CHECK_ERR_HIFI_NNLIB_KER(
        err,
        "DepthwiseConvAsym8: xa_nn_vec_activation_min_max_asym8_asym8 "
        "failed");

  } else {
    tflite::DepthwiseParams op_params;
    // Padding type is ignored, but still set.
    op_params.padding_type = PaddingType::kSame;
    op_params.padding_values.width = data.padding.width;
    op_params.padding_values.height = data.padding.height;
    op_params.stride_width = params->stride_width;
    op_params.stride_height = params->stride_height;
    op_params.dilation_width_factor = params->dilation_width_factor;
    op_params.dilation_height_factor = params->dilation_height_factor;
    op_params.depth_multiplier = params->depth_multiplier;
    op_params.quantized_activation_min = data.output_activation_min;
    op_params.quantized_activation_max = data.output_activation_max;
    op_params.input_offset = input_offset;
    op_params.weights_offset = filter_offset;
    op_params.output_offset = output_offset;
    op_params.output_multiplier = data.output_multiplier;
    // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
    op_params.output_shift = -data.output_shift;

    tflite::reference_ops::DepthwiseConv(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<uint8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<uint8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<uint8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
          : nullptr;

  // TODO(aselle): Consider whether float conv and quantized conv should be
  // separate ops to avoid dispatch overhead here.
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(context, node, params, data, input, filter, bias, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel(context, node, params, data, input, filter, bias,
                              output);
      break;
    case kTfLiteUInt8:
      EvalQuantized(context, node, params, data, input, filter, bias, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
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
