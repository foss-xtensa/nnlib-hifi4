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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_

#ifndef HIFI_BUILD
#include <cstdio>
#endif //HiFi_BUILD

#ifndef HIFI_BUILD
#define TF_LITE_FATAL(msg)          \
  do {                              \
    fprintf(stderr, "%s\n", (msg)); \
    exit(1);                        \
  } while (0)
#else
//TODO: handle error
#define TF_LITE_FATAL(msg)          \
  do {                              \
  } while (0)
#endif
#define TF_LITE_ASSERT(x)        \
  do {                           \
    if (!(x)) TF_LITE_FATAL(#x); \
  } while (0)
#define TF_LITE_ASSERT_EQ(x, y)                            \
  do {                                                     \
    if ((x) != (y)) TF_LITE_FATAL(#x " didn't equal " #y); \
  } while (0)

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_OP_UTIL_H_
