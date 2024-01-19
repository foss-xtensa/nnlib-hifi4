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

#ifndef ANDROID_ML_NN_COMMON_UTILS_H
#define ANDROID_ML_NN_COMMON_UTILS_H

#ifndef HIFI_BUILD
#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "ValidateHal.h"
#else
#include "xa_nnlib_ann_types.h"
#endif //HIFI_BUILD

#include <vector>

#ifndef HIFI_BUILD
#include <android-base/logging.h>
#endif //HIFI_BUILD

namespace android {
namespace nn {

// The number of data types (OperandCode) defined in NeuralNetworks.h.
const int kNumberOfDataTypes = 6;

// The number of operation types (OperationCode) defined in NeuralNetworks.h.
const int kNumberOfOperationTypes = 38;

// The number of execution preferences defined in NeuralNetworks.h.
const int kNumberOfPreferences = 3;

// The number of data types (OperandCode) defined in NeuralNetworksOEM.h.
const int kNumberOfDataTypesOEM = 2;

// The number of operation types (OperationCode) defined in NeuralNetworksOEM.h.
const int kNumberOfOperationTypesOEM = 1;

// The lowest number assigned to any OEM Code in NeuralNetworksOEM.h.
const int kOEMCodeBase = 10000;

/* IMPORTANT: if you change the following list, don't
 * forget to update the corresponding 'tags' table in
 * the initVlogMask() function implemented in Utils.cpp.
 */
enum VLogFlags {
    MODEL = 0,
    COMPILATION,
    EXECUTION,
    CPUEXE,
    MANAGER,
    DRIVER
};

#ifndef HIFI_BUILD //TODO: ppn:
#define VLOG_IS_ON(TAG) \
    ((vLogMask & (1 << (TAG))) != 0)

#define VLOG(TAG)         \
    if (LIKELY(!VLOG_IS_ON(TAG))) \
        ;                 \
    else                  \
        LOG(INFO)
#else

#define VLOG_IS_ON(TAG)  0
#define VLOG(TAG)         \
    if (1)  \
        ;                 \
    else                  \
        LOG(INFO)
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
extern int vLogMask;
void initVLogMask();
#endif //HIFI_BUILD

#ifdef NN_DEBUGGABLE
#define SHOW_IF_DEBUG(msg) msg
#else
#define SHOW_IF_DEBUG(msg) ""
#endif

#ifndef HIFI_BUILD //TODO: ppn
// Assert macro, as Android does not generally support assert.
#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)
#else

#define nnAssert(v)

#define LOG(x) std::cout

#endif //HIFI_BUILD

// Returns the amount of space needed to store a value of the specified
// dimensions and type.
uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions);

// Returns the amount of space needed to store a value of the dimensions and
// type of this operand.
inline uint32_t sizeOfData(const Operand& operand) {
    return sizeOfData(operand.type, operand.dimensions);
}

#ifndef HIFI_BUILD
// Returns the name of the operation type in ASCII.
const char* getOperationName(OperationType opCode);

// Returns the name of the operand type in ASCII.
const char* getOperandTypeName(OperandType type);

// Memory is unmapped.
// Memory is reference counted by hidl_memory instances, and is deallocated
// once there are no more references.
hidl_memory allocateSharedMemory(int64_t size);

// Returns the number of padding bytes needed to align data of the
// specified length.  It aligns object of length:
// 2, 3 on a 2 byte boundary,
// 4+ on a 4 byte boundary.
// We may want to have different alignments for tensors.
// TODO: This is arbitrary, more a proof of concept.  We need
// to determine what this should be.
uint32_t alignBytesNeeded(uint32_t index, size_t length);

// Does a detailed LOG(INFO) of the model
void logModelToInfo(const V1_0::Model& model);
void logModelToInfo(const V1_1::Model& model);

inline std::string toString(uint32_t obj) {
    return std::to_string(obj);
}

template <typename Type>
std::string toString(const std::vector<Type>& range) {
    std::string os = "[";
    for (size_t i = 0; i < range.size(); ++i) {
        os += (i == 0 ? "" : ", ") + toString(range[i]);
    }
    return os += "]";
}

inline bool validCode(uint32_t codeCount, uint32_t codeCountOEM, uint32_t code) {
    return (code < codeCount) || (code >= kOEMCodeBase && (code - kOEMCodeBase) < codeCountOEM);
}
#endif //HiFi_BUILD

#ifndef HIFI_BUILD /* not used in library */
int validateOperandType(const ANeuralNetworksOperandType& type, const char* tag, bool allowPartial);
int validateOperandList(uint32_t count, const uint32_t* list, uint32_t operandCount,
                        const char* tag);
int validateOperation(ANeuralNetworksOperationType opType,
                      uint32_t inputCount, const uint32_t* inputIndexes,
                      uint32_t outputCount, const uint32_t* outputIndexes,
                      const std::vector<Operand>& operands);
#endif

inline size_t getSizeFromInts(int lower, int higher) {
    return (uint32_t)(lower) + ((uint64_t)(uint32_t)(higher) << 32);
}

#ifndef HIFI_BUILD
// Convert ANEURALNETWORKS_* result code to ErrorStatus.
// Not guaranteed to be a 1-to-1 mapping.
ErrorStatus convertResultCodeToErrorStatus(int resultCode);

// Convert ErrorStatus to ANEURALNETWORKS_* result code.
// Not guaranteed to be a 1-to-1 mapping.
int convertErrorStatusToResultCode(ErrorStatus status);

// Versioning

bool compliantWithV1_0(V1_0::OperationType type);
bool compliantWithV1_0(V1_1::OperationType type);
bool compliantWithV1_1(V1_0::OperationType type);
bool compliantWithV1_1(V1_1::OperationType type);

bool compliantWithV1_0(const V1_0::Capabilities& capabilities);
bool compliantWithV1_0(const V1_1::Capabilities& capabilities);
bool compliantWithV1_1(const V1_0::Capabilities& capabilities);
bool compliantWithV1_1(const V1_1::Capabilities& capabilities);

bool compliantWithV1_0(const V1_0::Operation& operation);
bool compliantWithV1_0(const V1_1::Operation& operation);
bool compliantWithV1_1(const V1_0::Operation& operation);
bool compliantWithV1_1(const V1_1::Operation& operation);

bool compliantWithV1_0(const V1_0::Model& model);
bool compliantWithV1_0(const V1_1::Model& model);
bool compliantWithV1_1(const V1_0::Model& model);
bool compliantWithV1_1(const V1_1::Model& model);

V1_0::OperationType convertToV1_0(V1_0::OperationType type);
V1_0::OperationType convertToV1_0(V1_1::OperationType type);
V1_1::OperationType convertToV1_1(V1_0::OperationType type);
V1_1::OperationType convertToV1_1(V1_1::OperationType type);

V1_0::Capabilities convertToV1_0(const V1_0::Capabilities& capabilities);
V1_0::Capabilities convertToV1_0(const V1_1::Capabilities& capabilities);
V1_1::Capabilities convertToV1_1(const V1_0::Capabilities& capabilities);
V1_1::Capabilities convertToV1_1(const V1_1::Capabilities& capabilities);

V1_0::Operation convertToV1_0(const V1_0::Operation& operation);
V1_0::Operation convertToV1_0(const V1_1::Operation& operation);
V1_1::Operation convertToV1_1(const V1_0::Operation& operation);
V1_1::Operation convertToV1_1(const V1_1::Operation& operation);

V1_0::Model convertToV1_0(const V1_0::Model& model);
V1_0::Model convertToV1_0(const V1_1::Model& model);
V1_1::Model convertToV1_1(const V1_0::Model& model);
V1_1::Model convertToV1_1(const V1_1::Model& model);
#endif //HiFi_BUILD


#ifdef NN_DEBUGGABLE
uint32_t getProp(const char* str, uint32_t defaultValue = 0);
#endif  // NN_DEBUGGABLE

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_UTILS_H
