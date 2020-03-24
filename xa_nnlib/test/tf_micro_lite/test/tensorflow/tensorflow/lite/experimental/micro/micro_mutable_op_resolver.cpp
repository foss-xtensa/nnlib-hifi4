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

#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"

namespace tflite {

const TfLiteRegistration* MicroMutableOpResolver::FindOp(
    tflite::BuiltinOperator op, int version) const {
  for (int i = 0; i < registrations_len_; ++i) {
    const TfLiteRegistration& registration = registrations_[i];
    if ((registration.builtin_code == op) &&
        (registration.version == version)) {
      return &registration;
    }
  }
  return nullptr;
}

const TfLiteRegistration* MicroMutableOpResolver::FindOp(const char* op,
                                                         int version) const {
  for (int i = 0; i < registrations_len_; ++i) {
    const TfLiteRegistration& registration = registrations_[i];
    if ((registration.builtin_code == -1) &&
        (strcmp(registration.custom_name, op) == 0) &&
        (registration.version == version)) {
      return &registration;
    }
  }
  return nullptr;
}

void MicroMutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                        TfLiteRegistration* registration,
                                        int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    if (registrations_len_ >= TFLITE_REGISTRATIONS_MAX) {
      // TODO(petewarden) - Add error reporting hooks so we can report this!
      return;
    }
    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = op;
    new_registration->version = version;
  }
}

void MicroMutableOpResolver::AddCustom(const char* name,
                                       TfLiteRegistration* registration,
                                       int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    if (registrations_len_ >= TFLITE_REGISTRATIONS_MAX) {
      // TODO(petewarden) - Add error reporting hooks so we can report this!
      return;
    }
    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = -1;
    new_registration->custom_name = name;
    new_registration->version = version;
  }
}

}  // namespace tflite
