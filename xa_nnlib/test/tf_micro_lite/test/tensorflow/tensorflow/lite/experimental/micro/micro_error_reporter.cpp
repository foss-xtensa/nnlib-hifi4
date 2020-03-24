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

#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

namespace tflite {
namespace {
void DebugLogPrintf(const char* format, va_list args) {
  const int output_cache_size = 64;
  char output_cache[output_cache_size + 1];
  int output_cache_index = 0;
  const char* current = format;
  while (*current != 0) {
    if (*current == '%') {
      const char next = *(current + 1);
      if ((next == 'd') || (next == 's')) {
        current += 1;
        if (output_cache_index > 0) {
          output_cache[output_cache_index] = 0;
          DebugLog(output_cache);
          output_cache_index = 0;
        }
        if (next == 'd') {
          DebugLogInt32(va_arg(args, int));
        } else if (next == 's') {
          DebugLog(va_arg(args, char*));
        }
      }
    } else {
      output_cache[output_cache_index] = *current;
      output_cache_index += 1;
    }
    if (output_cache_index >= output_cache_size) {
      output_cache[output_cache_index] = 0;
      DebugLog(output_cache);
      output_cache_index = 0;
    }
    current += 1;
  }
  if (output_cache_index > 0) {
    output_cache[output_cache_index] = 0;
    DebugLog(output_cache);
    output_cache_index = 0;
  }
  DebugLog("\n");
}
}  // namespace

int MicroErrorReporter::Report(const char* format, va_list args) {
  DebugLogPrintf(format, args);
  return 0;
}

}  // namespace tflite
