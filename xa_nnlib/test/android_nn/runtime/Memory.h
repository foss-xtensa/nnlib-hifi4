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
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_RUNTIME_MEMORY_H
#define ANDROID_ML_NN_RUNTIME_MEMORY_H

#include "NeuralNetworks.h"
#include "Utils.h"

#include <cutils/native_handle.h>
#ifndef HIFI_BUILD
#include <sys/mman.h>
#endif //HIFI_BUILD
#include <unordered_map>

namespace android {
namespace nn {

class ModelBuilder;

// Represents a memory region.
class Memory {
public:
    Memory() {}
    virtual ~Memory() {}

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    // Creates a shared memory object of the size specified in bytes.
    int create(uint32_t size);

    hardware::hidl_memory getHidlMemory() const { return mHidlMemory; }

    // Returns a pointer to the underlying memory of this memory object.
    virtual int getPointer(uint8_t** buffer) const {
#ifndef HIFI_BUILD
        *buffer = static_cast<uint8_t*>(static_cast<void*>(mMemory->getPointer()));
#else
        *buffer = mMemory;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_NO_ERROR;
    }

    virtual bool validateSize(uint32_t offset, uint32_t length) const;
protected:
    // The hidl_memory handle for this shared memory.  We will pass this value when
    // communicating with the drivers.
    hardware::hidl_memory mHidlMemory;
#ifndef HIFI_BUILD
    sp<IMemory> mMemory;
#else
    uint8_t * mMemory;
#endif //HIFI_BUILD
};

class MemoryFd : public Memory {
public:
    MemoryFd() {}
    ~MemoryFd();

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    MemoryFd(const MemoryFd&) = delete;
    MemoryFd& operator=(const MemoryFd&) = delete;

    // Create the native_handle based on input size, prot, and fd.
    // Existing native_handle will be deleted, and mHidlMemory will wrap
    // the newly created native_handle.
    int set(size_t size, int prot, int fd, size_t offset);

    int getPointer(uint8_t** buffer) const override;

private:
    native_handle_t* mHandle = nullptr;
    mutable uint8_t* mMapping = nullptr;
};

// A utility class to accumulate mulitple Memory objects and assign each
// a distinct index number, starting with 0.
//
// The user of this class is responsible for avoiding concurrent calls
// to this class from multiple threads.
class MemoryTracker {
private:
    // The vector of Memory pointers we are building.
    std::vector<const Memory*> mMemories;
    // A faster way to see if we already have a memory than doing find().
    std::unordered_map<const Memory*, uint32_t> mKnown;

public:
    // Adds the memory, if it does not already exists.  Returns its index.
    // The memories should survive the tracker.
    uint32_t add(const Memory* memory);
    // Returns the number of memories contained.
    uint32_t size() const { return static_cast<uint32_t>(mKnown.size()); }
    // Returns the ith memory.
    const Memory* operator[](size_t i) const { return mMemories[i]; }
    // Iteration
    decltype(mMemories.begin()) begin() { return mMemories.begin(); }
    decltype(mMemories.end())   end()   { return mMemories.end(); }
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_MEMORY_H
