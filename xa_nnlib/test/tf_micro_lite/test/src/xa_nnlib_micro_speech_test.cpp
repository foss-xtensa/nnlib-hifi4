/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/preprocessor.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/low_latency_conv_10kws_micro_features_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

#include <stdio.h>
#include <stdlib.h>

// Basic utilities
#define PRINT_VAR(var)  printf( "%d: %s = %d\n", __LINE__, #var, var);fflush(stdout); fflush(stderr);
#define PRINT_STR(var)  printf( "%s\n",var);fflush(stdout); fflush(stderr);

namespace micro_test {
  int tests_passed;
  int tests_failed;
  bool is_test_complete;
  bool did_test_fail;
  tflite::ErrorReporter* reporter;
}

#define WINDOW_SIZE 320
#define SAMPLE_RATE 16000
#define SIGNAL_CHANNEL_IMAGE_WIDTH 40
#define SIGNAL_CHANNEL_IMAGE_HEIGHT 49
#define AUDIO_SAMPLE_DATA SIGNAL_CHANNEL_IMAGE_WIDTH*SIGNAL_CHANNEL_IMAGE_HEIGHT
const int g_sample_data_size = 480;


extern "C" {
int frontendprocess_inference(void *in, void *out);
}

extern unsigned int yes_wav_len;
extern unsigned char yes_wav[];

int frontendprocess_inference(void *in, void *out) {

/****File Read upto 1Second audio data***/
  uint8_t audio_spectrogram[AUDIO_SAMPLE_DATA];

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
/***********Frontendprocess*************/

  FrontendConfig config;
  FrontendState g_micro_features_state;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    error_reporter->Report("FrontendPopulateState() failed");
    return kTfLiteError;
  }
  size_t num_samples_read;
  for(int j=0;j<SIGNAL_CHANNEL_IMAGE_HEIGHT;j++)
  {
      FrontendOutput frontend_output = FrontendProcessSamples(
              &g_micro_features_state, (const int16_t*)in+j*WINDOW_SIZE, g_sample_data_size, &num_samples_read);
      for (int i = 0; i < frontend_output.size; ++i) {
          // These scaling values are derived from those used in input_data.py in the
          // training pipeline.
          constexpr int32_t value_scale = (10 * 255);
          constexpr int32_t value_div = (256 * 26);
          int32_t value =
              ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
              value_div;
          if (value < 0) {
              value = 0;
          }
          if (value > 255) {
              value = 255;
          }
          audio_spectrogram[SIGNAL_CHANNEL_IMAGE_WIDTH*j+i] = value;
      }
  }

/***********inference*******************/
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(tf_low_latency_conv_10kws_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  tflite::MicroOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_MAX_POOL_2D,
      tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_CONV_2D,
      tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Create an area of memory to use for input, output, and intermediate arrays.
  const int tensor_arena_size = 10 * 1024 *20;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(49, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(40, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // Copy a spectrogram created from a .wav audio file of someone saying "Yes",
  // into the memory area used for the input.
  const uint8_t* yes_features_data = audio_spectrogram;
  //const uint8_t* yes_features_data = g_yes_micro_f2e59fea_nohash_1_data ;
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = yes_features_data[i];
  }

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(12, output->dims->data[1]); // 10 keywords + silence + unknown
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // There are four possible classes in the output, each with a score.
  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;

  const int kYesIndex = 2;
  const int kNoIndex = 3;
  const int kUpIndex = 4;
  const int kDownIndex = 5;
  const int kLeftIndex = 6;
  const int kRightIndex = 7;
  const int kOnIndex = 8;
  const int kOffIndex = 9;
  const int kStopIndex = 10;
  const int kGoIndex = 11;

  // Make sure that the expected "Yes" score is higher than the other classes.
  uint8_t silence_score = output->data.uint8[kSilenceIndex];
  uint8_t unknown_score = output->data.uint8[kUnknownIndex];
  uint8_t yes_score = output->data.uint8[kYesIndex];
  uint8_t no_score = output->data.uint8[kNoIndex];
  uint8_t up_score = output->data.uint8[kUpIndex];
  uint8_t down_score = output->data.uint8[kDownIndex];
  uint8_t left_score = output->data.uint8[kLeftIndex];
  uint8_t right_score = output->data.uint8[kRightIndex];
  uint8_t on_score = output->data.uint8[kOnIndex];
  uint8_t off_score = output->data.uint8[kOffIndex];
  uint8_t stop_score = output->data.uint8[kStopIndex];
  uint8_t go_score = output->data.uint8[kGoIndex];

  PRINT_VAR(silence_score)
  PRINT_VAR(unknown_score)
  PRINT_VAR(yes_score)
  PRINT_VAR(no_score)
  PRINT_VAR(up_score)
  PRINT_VAR(down_score)
  PRINT_VAR(left_score)
  PRINT_VAR(right_score)
  PRINT_VAR(on_score)
  PRINT_VAR(off_score)
  PRINT_VAR(stop_score)
  PRINT_VAR(go_score)


  //Debug logs are not implemented,so using out_ptr returning success value here
  // out_ptr[2] = 0xA;
  short *out_ptr = (short*)out;
  int i = 0;
  out_ptr[i++] = silence_score;
  out_ptr[i++] = unknown_score;
  out_ptr[i++] = yes_score;
  out_ptr[i++] = no_score;
  out_ptr[i++] = up_score;
  out_ptr[i++] = down_score;
  out_ptr[i++] = left_score;
  out_ptr[i++] = right_score;
  out_ptr[i++] = on_score;
  out_ptr[i++] = off_score;
  out_ptr[i++] = stop_score;
  out_ptr[i++] = go_score;
  error_reporter->Report("Ran successfully\n");

#ifndef HIFI_WARNINGS
  return 0;
#endif
}


