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

#ifndef TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_LINEAR_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_LINEAR_MEMORY_PLANNER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/memory_planner/memory_planner.h"

namespace tflite {

// The simplest possible memory planner that just lays out all buffers at
// increasing offsets without trying to reuse memory.
class LinearMemoryPlanner : public MemoryPlanner {
 public:
  LinearMemoryPlanner();
  ~LinearMemoryPlanner() override;

  TfLiteStatus AddBuffer(tflite::ErrorReporter* error_reporter, int size,
                         int first_time_used, int last_time_used) override;

  size_t GetMaximumMemorySize() override;
  int GetBufferCount() override;
  TfLiteStatus GetOffsetForBuffer(tflite::ErrorReporter* error_reporter,
                                  int buffer_index, int* offset) override;

 private:
  static constexpr int kMaxBufferCount = 1024;
  size_t buffer_offsets_[kMaxBufferCount];
  int current_buffer_count_;
  size_t next_free_offset_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MEMORY_PLANNER_LINEAR_MEMORY_PLANNER_H_
