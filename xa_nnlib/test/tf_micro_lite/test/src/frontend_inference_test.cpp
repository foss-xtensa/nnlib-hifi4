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

#include "tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/no_30ms_sample_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/no_power_spectrum_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/yes_30ms_sample_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/yes_power_spectrum_data.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#define WINDOW_SIZE 320
#define SAMPLE_RATE 16000
#define SIGNAL_CHANNEL_IMAGE_WIDTH 43
#define SIGNAL_CHANNEL_IMAGE_HEIGHT 49
#define AUDIO_SAMPLE_DATA 43*49


extern "C" {
int frontendprocess_inference(void *in, void *out);
}

extern unsigned int yes_wav_len;
extern unsigned char yes_wav[];

int frontendprocess_inference(void *in, void *out) {

/****File Read upto 1Second audio data***/   
  short *audiobuffer = (short*)in;
  uint8_t audio_spectrogram[AUDIO_SAMPLE_DATA];
  short *out_ptr = (short*)out;
  //uint8_t *out_ptr = (uint8_t*)out;
  int i;

/***********Frontendprocess*************/
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  uint8_t yes_calculated_data[g_yes_power_spectrum_data_size];
  for(int j=0;j<SIGNAL_CHANNEL_IMAGE_HEIGHT;j++){
    TfLiteStatus yes_status = Preprocess(
    error_reporter, (short*)(audiobuffer+j*WINDOW_SIZE), g_yes_30ms_sample_data_size,
    g_yes_power_spectrum_data_size, yes_calculated_data);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, yes_status);
    for (i = 0; i < SIGNAL_CHANNEL_IMAGE_WIDTH; ++i) {
      audio_spectrogram[SIGNAL_CHANNEL_IMAGE_WIDTH*j+i] = yes_calculated_data[i];
    }
  }
 

/***********inference*******************/
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_tiny_conv_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  const int tensor_arena_size = 10 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(49, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(43, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // Copy a spectrogram created from a .wav audio file of someone saying "Yes",
  // into the memory area used for the input.
  const uint8_t* yes_features_data = audio_spectrogram;
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
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // There are four possible classes in the output, each with a score.
  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;
  const int kYesIndex = 2;
  const int kNoIndex = 3;

  // Make sure that the expected "Yes" score is higher than the other classes.
  uint8_t silence_score = output->data.uint8[kSilenceIndex];
  uint8_t unknown_score = output->data.uint8[kUnknownIndex];
  uint8_t yes_score = output->data.uint8[kYesIndex];
  uint8_t no_score = output->data.uint8[kNoIndex];

  //Debug logs are not implemented,so using out_ptr returning success value here   
  out_ptr[2] = 0xA;
  i=3;
  out_ptr[i++] = silence_score;
  out_ptr[i++] = unknown_score;
  out_ptr[i++] = yes_score;
  out_ptr[i++] = no_score;
  error_reporter->Report("Ran successfully\n");

#ifndef HIFI_WARNINGS
  return 0;
#endif
}


