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

#include "tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

extern "C" {
int preprocessor_test(void *in, void *out);
}

int preprocessor_test(void *in, void *out) {

  short *out_ptr = (short*)out;

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  uint8_t yes_calculated_data[g_yes_power_spectrum_data_size];

  TfLiteStatus yes_status = Preprocess(
      error_reporter, g_yes_30ms_sample_data, g_yes_30ms_sample_data_size,
      g_yes_power_spectrum_data_size, yes_calculated_data);


  for (int i = 0; i < g_yes_power_spectrum_data_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(g_yes_power_spectrum_data[i],
                            yes_calculated_data[i]);
    if (g_yes_power_spectrum_data[i] != yes_calculated_data[i]) {
      error_reporter->Report("Expected value %d but found %d",
                             g_yes_power_spectrum_data[i],
                             yes_calculated_data[i]);
    }
  }

  uint8_t no_calculated_data[g_yes_power_spectrum_data_size];
  TfLiteStatus no_status = Preprocess(
      error_reporter, g_no_30ms_sample_data, g_no_30ms_sample_data_size,
      g_no_power_spectrum_data_size, no_calculated_data);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, no_status);

  for (int i = 0; i < g_no_power_spectrum_data_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(g_no_power_spectrum_data[i], no_calculated_data[i]);
    if (g_no_power_spectrum_data[i] != no_calculated_data[i]) {
      error_reporter->Report("Expected value %d but found %d",
                             g_no_power_spectrum_data[i],
                             no_calculated_data[i]);
    }
  }

  //Debug logs are not implemented,so using out_ptr returning success value here 
  out_ptr[0] = 0xA;
#ifndef HIFI_WARNINGS
  (void)yes_status;
  return 0;
#endif
}


