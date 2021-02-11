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
/// \file
/// Memory management for TF Lite.
#ifndef TENSORFLOW_LITE_ALLOCATION_H_
#define TENSORFLOW_LITE_ALLOCATION_H_

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {

// A memory allocation handle. This could be a mmap or shared memory.
class Allocation {
 public:
  virtual ~Allocation() {}

  enum class Type {
    kMMap,
    kFileCopy,
    kMemory,
  };

  // Base pointer of this allocation
  virtual const void* base() const = 0;
  // Size in bytes of the allocation
  virtual size_t bytes() const = 0;
  // Whether the allocation is valid
  virtual bool valid() const = 0;
  // Return the type of the Allocation.
  Type type() const { return type_; }

 protected:
  Allocation(ErrorReporter* error_reporter, Type type)
      : error_reporter_(error_reporter), type_(type) {}
  ErrorReporter* error_reporter_;

 private:
  const Type type_;
};

class MMAPAllocation : public Allocation {
 public:
  MMAPAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~MMAPAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

  int fd() const { return mmap_fd_; }

  static bool IsSupported();

 protected:
  // Data required for mmap.
  int mmap_fd_ = -1;  // mmap file descriptor
  const void* mmapped_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class FileCopyAllocation : public Allocation {
 public:
  FileCopyAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~FileCopyAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  // Data required for mmap.
  std::unique_ptr<const char[]> copied_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class MemoryAllocation : public Allocation {
 public:
  // Allocates memory with the pointer and the number of bytes of the memory.
  // The pointer has to remain alive and unchanged until the destructor is
  // called.
  MemoryAllocation(const void* ptr, size_t num_bytes,
                   ErrorReporter* error_reporter);
  virtual ~MemoryAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  const void* buffer_;
  size_t buffer_size_bytes_ = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ALLOCATION_H_
