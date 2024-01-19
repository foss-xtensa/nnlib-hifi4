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

#ifndef ANDROID_ML_NN_COMMON_HAL_INTERFACES_H
#define ANDROID_ML_NN_COMMON_HAL_INTERFACES_H

#ifndef HIFI_BUILD
#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#endif //HIFI_BUILD
#include <android/hardware/neuralnetworks/1.0/types.h>
#ifndef HIFI_BUILD
#include <android/hardware/neuralnetworks/1.1/IDevice.h>
#endif //HIFI_BUILD
#include <android/hardware/neuralnetworks/1.1/types.h>
#ifndef HIFI_BUILD
#include <android/hidl/allocator/1.0/IAllocator.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#endif //HIFI_BUILD

using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
#ifndef HIFI_BUILD
using ::android::hardware::hidl_vec;
#else
#define hidl_vec std::vector
#endif //HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_0::DataLocation;
#ifndef HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_0::DeviceStatus;
#endif //HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_0::ErrorStatus;
using ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc;
#ifndef HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_0::IExecutionCallback;
using ::android::hardware::neuralnetworks::V1_0::IPreparedModel;
using ::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback;
#endif //HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_0::Operand;
using ::android::hardware::neuralnetworks::V1_0::OperandLifeTime;
using ::android::hardware::neuralnetworks::V1_0::OperandType;
using ::android::hardware::neuralnetworks::V1_0::PerformanceInfo;
using ::android::hardware::neuralnetworks::V1_0::Request;
using ::android::hardware::neuralnetworks::V1_0::RequestArgument;
using ::android::hardware::neuralnetworks::V1_1::Capabilities;
using ::android::hardware::neuralnetworks::V1_1::ExecutionPreference;
#ifndef HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_1::IDevice;
#endif //HIFI_BUILD
using ::android::hardware::neuralnetworks::V1_1::Model;
using ::android::hardware::neuralnetworks::V1_1::Operation;
using ::android::hardware::neuralnetworks::V1_1::OperationType;
#ifndef HIFI_BUILD
using ::android::hidl::allocator::V1_0::IAllocator;
#endif //HIFI_BUILD
using ::android::hidl::memory::V1_0::IMemory;

namespace V1_0 = ::android::hardware::neuralnetworks::V1_0;
namespace V1_1 = ::android::hardware::neuralnetworks::V1_1;

namespace android {
namespace nn {

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_HAL_INTERFACES_H
