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

#define LOG_TAG "Utils"

#ifndef HIFI_BUILD
#include "Utils.h"
#else
#include "Utils_lib.h"
#endif //HiFi_BUILD

#ifndef HIFI_BUILD
#include "NeuralNetworks.h"
#include "NeuralNetworksOEM.h"
#include <android-base/properties.h>
#include <android-base/strings.h>
#include <android-base/logging.h>
#include <sys/system_properties.h>
#endif //HIFI_BUILD
#include <unordered_map>

#ifndef HIFI_BUILD
using ::android::hidl::allocator::V1_0::IAllocator;
#endif //HIFI_BUILD

namespace android {
namespace nn {

#ifndef HIFI_BUILD
int vLogMask = ~0;

const char kVLogPropKey[] = "debug.nn.vlog";
// Split the space separated list of tags from verbose log setting and build the
// logging mask from it. note that '1' and 'all' are special cases to enable all
// verbose logging.
//
// NN API verbose logging setting comes from system property debug.nn.vlog.
// Example:
// setprop debug.nn.vlog 1 : enable all logging tags.
// setprop debug.nn.vlog "model compilation" : only enable logging for MODEL and
//                                             COMPILATION tags.
void initVLogMask() {
    vLogMask = 0;
    const std::string vLogSetting = android::base::GetProperty(kVLogPropKey, "");
    if (vLogSetting.empty()) {
        return;
    }

    std::unordered_map<std::string, int> vLogFlags = {
        {"1", -1},
        {"all", -1},
        {"model", MODEL},
        {"compilation", COMPILATION},
        {"execution", EXECUTION},
        {"cpuexe", CPUEXE},
        {"manager", MANAGER},
        {"driver", DRIVER}};

    std::vector<std::string> elements = android::base::Split(vLogSetting, " ,:");
    for (const auto& elem : elements) {
        const auto& flag = vLogFlags.find(elem);
        if (flag == vLogFlags.end()) {
            LOG(ERROR) << "Unknown trace flag: " << elem;
            continue;
        }

        if (flag->second == -1) {
            // -1 is used for the special values "1" and "all" that enable all
            // tracing.
            vLogMask = ~0;
            return;
        } else {
            vLogMask |= 1 << flag->second;
        }
    }
}
#endif //HIFI_BUILD

namespace {

template <typename EntryType, uint32_t entryCount, uint32_t entryCountOEM>
EntryType tableLookup(const EntryType (&table)[entryCount],
                      const EntryType (&tableOEM)[entryCountOEM],
                      uint32_t code) {
    if (code < entryCount) {
        return table[code];
    } else if (code >= kOEMCodeBase && (code - kOEMCodeBase) < entryCountOEM) {
        return tableOEM[code - kOEMCodeBase];
    } else {
        nnAssert(!"tableLookup: bad code");
        return EntryType();
    }
}

};  // anonymous namespace

#define COUNT(X) (sizeof(X) / sizeof(X[0]))

#ifndef HIFI_BUILD
const char* kTypeNames[kNumberOfDataTypes] = {
#else
const char* const kTypeNames[kNumberOfDataTypes] = {
#endif
        "FLOAT32",        "INT32",        "UINT32",
        "TENSOR_FLOAT32", "TENSOR_INT32", "TENSOR_QUANT8_ASYMM",
};

static_assert(COUNT(kTypeNames) == kNumberOfDataTypes, "kTypeNames is incorrect");

#ifndef HIFI_BUILD
const char* kTypeNamesOEM[kNumberOfDataTypesOEM] = {
#else
const char* const kTypeNamesOEM[kNumberOfDataTypesOEM] = {
#endif
        "OEM",            "TENSOR_OEM_BYTE",
};

static_assert(COUNT(kTypeNamesOEM) == kNumberOfDataTypesOEM, "kTypeNamesOEM is incorrect");

const char* getOperandTypeName(OperandType type) {
    uint32_t n = static_cast<uint32_t>(type);
    return tableLookup(kTypeNames, kTypeNamesOEM, n);
}

// TODO Check if this useful
#ifndef HIFI_BUILD
const char* kErrorNames[] = {
        "NO_ERROR", "OUT_OF_MEMORY", "INCOMPLETE", "NULL", "BAD_DATA",
};
#endif

#ifndef HIFI_BUILD
const char* kOperationNames[kNumberOfOperationTypes] = {
#else
const char* const kOperationNames[kNumberOfOperationTypes] = {
#endif
        "ADD",
        "AVERAGE_POOL",
        "CONCATENATION",
        "CONV",
        "DEPTHWISE_CONV",
        "DEPTH_TO_SPACE",
        "DEQUANTIZE",
        "EMBEDDING_LOOKUP",
        "FLOOR",
        "FULLY_CONNECTED",
        "HASHTABLE_LOOKUP",
        "L2_NORMALIZATION",
        "L2_POOL",
        "LOCAL_RESPONSE_NORMALIZATION",
        "LOGISTIC",
        "LSH_PROJECTION",
        "LSTM",
        "MAX_POOL",
        "MUL",
        "RELU",
        "RELU1",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RNN",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SVDF",
        "TANH",
        "BATCH_TO_SPACE_ND",
        "DIV",
        "MEAN",
        "PAD",
        "SPACE_TO_BATCH_ND",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "TRANSPOSE",
};

static_assert(COUNT(kOperationNames) == kNumberOfOperationTypes, "kOperationNames is incorrect");

#ifndef HIFI_BUILD
const char* kOperationNamesOEM[kNumberOfOperationTypesOEM] = {
#else
const char* const kOperationNamesOEM[kNumberOfOperationTypesOEM] = {
#endif
        "OEM_OPERATION",
};

static_assert(COUNT(kOperationNamesOEM) == kNumberOfOperationTypesOEM,
              "kOperationNamesOEM is incorrect");

const char* getOperationName(OperationType type) {
    uint32_t n = static_cast<uint32_t>(type);
    return tableLookup(kOperationNames, kOperationNamesOEM, n);
}

const uint32_t kSizeOfDataType[]{
        4, // ANEURALNETWORKS_FLOAT32
        4, // ANEURALNETWORKS_INT32
        4, // ANEURALNETWORKS_UINT32
        4, // ANEURALNETWORKS_TENSOR_FLOAT32
        4, // ANEURALNETWORKS_TENSOR_INT32
        1  // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
};

static_assert(COUNT(kSizeOfDataType) == kNumberOfDataTypes, "kSizeOfDataType is incorrect");

const bool kScalarDataType[]{
        true,  // ANEURALNETWORKS_FLOAT32
        true,  // ANEURALNETWORKS_INT32
        true,  // ANEURALNETWORKS_UINT32
        false, // ANEURALNETWORKS_TENSOR_FLOAT32
        false, // ANEURALNETWORKS_TENSOR_INT32
        false, // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
};

static_assert(COUNT(kScalarDataType) == kNumberOfDataTypes, "kScalarDataType is incorrect");

const uint32_t kSizeOfDataTypeOEM[]{
        0, // ANEURALNETWORKS_OEM
        1, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kSizeOfDataTypeOEM) == kNumberOfDataTypesOEM,
              "kSizeOfDataTypeOEM is incorrect");

const bool kScalarDataTypeOEM[]{
        true,  // ANEURALNETWORKS_OEM
        false, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kScalarDataTypeOEM) == kNumberOfDataTypesOEM,
              "kScalarDataTypeOEM is incorrect");

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions) {
    int n = static_cast<int>(type);

    uint32_t size = tableLookup(kSizeOfDataType, kSizeOfDataTypeOEM, n);

    if (tableLookup(kScalarDataType, kScalarDataTypeOEM, n) == true) {
        return size;
    }

    for (auto d : dimensions) {
        size *= d;
    }
    return size;
}

#ifndef HIFI_BUILD
hidl_memory allocateSharedMemory(int64_t size) {
    static const std::string type = "ashmem";
    static sp<IAllocator> allocator = IAllocator::getService(type);

    hidl_memory memory;

    // TODO: should we align memory size to nearest page? doesn't seem necessary...
    allocator->allocate(size, [&](bool success, const hidl_memory& mem) {
        if (!success) {
            LOG(ERROR) << "unable to allocate " << size << " bytes of " << type;
        } else {
            memory = mem;
        }
    });

    return memory;
}
#else

#ifndef HIFI_BUILD
#define PATH_MAX 4096
static int ashmem_create_region(const char* /*ignored*/, size_t size) {
    char pattern[PATH_MAX];
    snprintf(pattern, sizeof(pattern), "/tmp/android-ashmem-%d-XXXXXXXXX", getpid());
    int fd = mkstemp(pattern);
    if (fd == -1) return -1;

    unlink(pattern);

    if (TEMP_FAILURE_RETRY(ftruncate(fd, size)) == -1) {
        close(fd);
        return -1;
    }

    return fd;
}

static hidl_memory allocateOne(uint64_t size) {
    int fd = ashmem_create_region("AshmemAllocator_hidl", size);
    if (fd < 0) {
        LOG(WARNING) << "ashmem_create_region(" << size << ") fails with " << fd;
        return hidl_memory();
    }

    native_handle_t* handle = native_handle_create(1, 0);
    handle->data[0] = fd;
    LOG(VERBOSE) << "ashmem_create_region(" << size << ") returning hidl_memory(" << handle << ", "
        << size << ")";
    return hidl_memory("ashmem", handle, size);
}


hidl_memory allocateSharedMemory(int64_t size) {
    static const std::string type = "ashmem";

    hidl_memory memory;

    // TODO: should we align memory size to nearest page? doesn't seem necessary...
    {
        hidl_memory memory_temp = allocateOne(size);
        if (!(memory_temp.handle() != nullptr /* success */)) {
            LOG(ERROR) << "unable to allocate " << size << " bytes of " << type;
        } else {
            memory = memory_temp;
        }
        //cleanup(std::move(memory_temp)); //ppn TODO?
    }

    return memory;
}
#endif //HIFI_BUILD
#endif //HIFI_BUILD

uint32_t alignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0; // No alignment necessary
    } else if (length < 4) {
        pattern = 1; // Align on 2-byte boundary
    } else {
        pattern = 3; // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

#ifndef HIFI_BUILD
void logModelToInfo(const V1_0::Model& model) {
    LOG(INFO) << "V1_0::Model start";
    LOG(INFO) << "operands" << toString(model.operands);
    LOG(INFO) << "operations" << toString(model.operations);
    LOG(INFO) << "inputIndexes" << toString(model.inputIndexes);
    LOG(INFO) << "outputIndexes" << toString(model.outputIndexes);
    LOG(INFO) << "operandValues size" << model.operandValues.size();
    LOG(INFO) << "pools" << SHOW_IF_DEBUG(toString(model.pools));
}

void logModelToInfo(const V1_1::Model& model) {
    LOG(INFO) << "V1_1::Model start";
    LOG(INFO) << "operands" << toString(model.operands);
    LOG(INFO) << "operations" << toString(model.operations);
    LOG(INFO) << "inputIndexes" << toString(model.inputIndexes);
    LOG(INFO) << "outputIndexes" << toString(model.outputIndexes);
    LOG(INFO) << "operandValues size" << model.operandValues.size();
    LOG(INFO) << "pools" << SHOW_IF_DEBUG(toString(model.pools));
}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD /* not used in library */
// Validates the type. The used dimensions can be underspecified.
int validateOperandType(const ANeuralNetworksOperandType& type, const char* tag,
                        bool allowPartial) {
    if (!allowPartial) {
        for (uint32_t i = 0; i < type.dimensionCount; i++) {
            if (type.dimensions[i] == 0) {
#ifndef HIFI_BUILD
                LOG(ERROR) << tag << " OperandType invalid dimensions[" << i
                           << "] = " << type.dimensions[i];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
        }
    }
    if (!validCode(kNumberOfDataTypes, kNumberOfDataTypesOEM, type.type)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << tag << " OperandType invalid type " << type.type;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type.type == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM) {
        if (type.zeroPoint < 0 || type.zeroPoint > 255) {
#ifndef HIFI_BUILD
            LOG(ERROR) << tag << " OperandType invalid zeroPoint " << type.zeroPoint;
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
        if (type.scale <= 0.f) {
#ifndef HIFI_BUILD
            LOG(ERROR) << tag << " OperandType invalid scale " << type.scale;
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    if (type.type == ANEURALNETWORKS_FLOAT32 ||
        type.type == ANEURALNETWORKS_INT32 ||
        type.type == ANEURALNETWORKS_UINT32 ||
        type.type == ANEURALNETWORKS_OEM_SCALAR) {
        if (type.dimensionCount != 0 || type.dimensions != nullptr) {
#ifndef HIFI_BUILD
            LOG(ERROR) << tag << " Invalid dimensions for scalar type";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    return ANEURALNETWORKS_NO_ERROR;
}

#ifndef HIFI_BUILD
int validateOperandList(uint32_t count, const uint32_t* list, uint32_t operandCount,
                        const char* tag) {
    for (uint32_t i = 0; i < count; i++) {
        if (list[i] >= operandCount) {
#ifndef HIFI_BUILD
            LOG(ERROR) << tag << " invalid operand index at " << i << " = " << list[i]
                       << ", operandCount " << operandCount;
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int validateOperationOperandTypes(const std::vector<Operand>& operands,
                                  uint32_t inOperandCount, const uint32_t* inOperandIndexes,
                                  const std::vector<OperandType>& inExpectedTypes,
                                  uint32_t outOperandCount, const uint32_t* outOperandIndexes,
                                  const std::vector<OperandType>& outExpectedInTypes) {
    if (inOperandCount > static_cast<uint32_t>(inExpectedTypes.size()) ||
        outOperandCount > static_cast<uint32_t>(outExpectedInTypes.size())) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    for (uint32_t i = 0; i < inOperandCount; i++) {
        if (operands[inOperandIndexes[i]].type != inExpectedTypes[i]) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid input tensor type "
                       << toString(operands[inOperandIndexes[i]].type)
                       << " for input " << i << ", expected " << toString(inExpectedTypes[i]);
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    for (uint32_t i = 0; i < outOperandCount; i++) {
        if (operands[outOperandIndexes[i]].type != outExpectedInTypes[i]) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid output tensor type "
                       << toString(operands[outOperandIndexes[i]].type)
                       << " for input " << i << ", expected " << toString(outExpectedInTypes[i]);
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int validateOperation(ANeuralNetworksOperationType opType,
                      uint32_t inputCount, const uint32_t* inputIndexes,
                      uint32_t outputCount, const uint32_t* outputIndexes,
                      const std::vector<Operand>& operands) {
    int n = validateOperandList(inputCount, inputIndexes, static_cast<uint32_t>(operands.size()),
                                "ANeuralNetworksModel_addOperation inputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = validateOperandList(outputCount, outputIndexes, static_cast<uint32_t>(operands.size()),
                            "ANeuralNetworksModel_addOperation outputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    auto logInvalidInOutNumber = [opType, inputCount, outputCount](int expIn, int expOut) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Invalid number of input operands ("
                   << inputCount << ", expected " << expIn << ") or output operands ("
                   << outputCount << ", expected " << expOut << ") for operation "
                   << kOperationNames[opType];
#endif //HIFI_BUILD
    };

    switch (opType) {
        case ANEURALNETWORKS_OEM_OPERATION: {
            return ANEURALNETWORKS_NO_ERROR;
        }
        case ANEURALNETWORKS_ADD: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_MUL: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_FLOOR: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_DEQUANTIZE: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_DEPTHWISE_CONV_2D: {
            if ((inputCount != 11 && inputCount != 8) || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected 11 or 8) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            if (inputCount == 11) {
                std::vector<OperandType> explicitScalarTypes(3, OperandType::INT32);
                inExpectedTypes.insert(inExpectedTypes.end(),
                                       explicitScalarTypes.begin(),
                                       explicitScalarTypes.end());
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_CONV_2D: {
            if ((inputCount != 10 && inputCount != 7) || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected 10 or 7) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            if (inputCount == 10) {
                std::vector<OperandType> explicitScalarTypes(3, OperandType::INT32);
                inExpectedTypes.insert(inExpectedTypes.end(),
                                       explicitScalarTypes.begin(),
                                       explicitScalarTypes.end());
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_AVERAGE_POOL_2D: {
            if ((inputCount != 10 && inputCount != 7) || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected 10 or 7) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            if (inputCount == 10) {
                std::vector<OperandType> explicitScalarTypes(3, OperandType::INT32);
                inExpectedTypes.insert(inExpectedTypes.end(),
                                       explicitScalarTypes.begin(),
                                       explicitScalarTypes.end());
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_L2_POOL_2D: {
            if ((inputCount != 10 && inputCount != 7) || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected 10 or 7) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            if (inputCount == 10) {
                std::vector<OperandType> explicitScalarTypes(3, OperandType::INT32);
                inExpectedTypes.insert(inExpectedTypes.end(),
                                       explicitScalarTypes.begin(),
                                       explicitScalarTypes.end());
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_MAX_POOL_2D: {
            if ((inputCount != 10 && inputCount != 7) || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected 10 or 7) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            if (inputCount == 10) {
                std::vector<OperandType> explicitScalarTypes(3, OperandType::INT32);
                inExpectedTypes.insert(inExpectedTypes.end(),
                                       explicitScalarTypes.begin(),
                                       explicitScalarTypes.end());
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RELU: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RELU1: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RELU6: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_TANH: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_LOGISTIC: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SOFTMAX: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_FULLY_CONNECTED: {
            if (inputCount != 4 || outputCount != 1) {
                logInvalidInOutNumber(4, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_CONCATENATION: {
            if (inputCount < 2 || outputCount != 1) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Invalid number of input operands ("
                           << inputCount << ", expected at least 2) or output operands ("
                           << outputCount << ", expected 1) for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes(inputCount, inputType);
            std::vector<OperandType> outExpectedTypes = {inputType};
            // The last one is the activation function.
            inExpectedTypes.back() = OperandType::INT32;
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_L2_NORMALIZATION: {
            if (inputCount != 1 || outputCount != 1) {
                logInvalidInOutNumber(1, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION: {
            if (inputCount != 5 || outputCount != 1) {
                logInvalidInOutNumber(5, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::FLOAT32,
                                   OperandType::FLOAT32,
                                   OperandType::FLOAT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RESHAPE: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RESIZE_BILINEAR: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_DEPTH_TO_SPACE: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SPACE_TO_DEPTH: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_EMBEDDING_LOOKUP: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[1]].type;
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_INT32,
                                                        inputType};
            std::vector<OperandType> outExpectedTypes = {inputType};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_HASHTABLE_LOOKUP: {
            if (inputCount != 3 || outputCount != 2) {
                logInvalidInOutNumber(3, 2);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[2]].type;
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_INT32,
                                                        OperandType::TENSOR_INT32,
                                                        inputType};
            std::vector<OperandType> outExpectedTypes = {inputType,
                                                         OperandType::TENSOR_QUANT8_ASYMM};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_LSH_PROJECTION: {
            if (inputCount != 4 || outputCount != 1) {
                logInvalidInOutNumber(4, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[1]].type;
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                        inputType,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::INT32};
            std::vector<OperandType> outExpectedTypes = {OperandType::TENSOR_INT32};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_LSTM: {
            if (inputCount != 23 || outputCount != 4) {
                logInvalidInOutNumber(23, 4);
                return ANEURALNETWORKS_BAD_DATA;
            }
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::INT32,
                                                        OperandType::FLOAT32,
                                                        OperandType::FLOAT32};
            std::vector<OperandType> outExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                         OperandType::TENSOR_FLOAT32,
                                                         OperandType::TENSOR_FLOAT32,
                                                         OperandType::TENSOR_FLOAT32};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_RNN: {
            if (inputCount != 6 || outputCount != 2) {
                logInvalidInOutNumber(6, 2);
                return ANEURALNETWORKS_BAD_DATA;
            }
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::INT32};
            std::vector<OperandType> outExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                         OperandType::TENSOR_FLOAT32};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SVDF: {
            if (inputCount != 7 || outputCount != 2) {
                logInvalidInOutNumber(7, 2);
                return ANEURALNETWORKS_BAD_DATA;
            }
            std::vector<OperandType> inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::TENSOR_FLOAT32,
                                                        OperandType::INT32,
                                                        OperandType::INT32};
            std::vector<OperandType> outExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                                         OperandType::TENSOR_FLOAT32};
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_BATCH_TO_SPACE_ND: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SPACE_TO_BATCH_ND: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_PAD: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SQUEEZE: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_TRANSPOSE: {
            if (inputCount != 2 || outputCount != 1) {
                logInvalidInOutNumber(2, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_STRIDED_SLICE: {
            if (inputCount != 7 || outputCount != 1) {
                logInvalidInOutNumber(7, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32,
                                   OperandType::INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_DIV: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_SUB: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_FLOAT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        case ANEURALNETWORKS_MEAN: {
            if (inputCount != 3 || outputCount != 1) {
                logInvalidInOutNumber(3, 1);
                return ANEURALNETWORKS_BAD_DATA;
            }
            auto inputType = operands[inputIndexes[0]].type;
            std::vector<OperandType> inExpectedTypes;
            std::vector<OperandType> outExpectedTypes;
            if (inputType == OperandType::TENSOR_FLOAT32) {
                inExpectedTypes = {OperandType::TENSOR_FLOAT32,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_FLOAT32};
            } else if (inputType == OperandType::TENSOR_QUANT8_ASYMM) {
                inExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM,
                                   OperandType::TENSOR_INT32,
                                   OperandType::INT32};
                outExpectedTypes = {OperandType::TENSOR_QUANT8_ASYMM};
            } else {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Unsupported input tensor type for operation "
                           << kOperationNames[opType];
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
            return validateOperationOperandTypes(operands,
                                                 inputCount, inputIndexes,
                                                 inExpectedTypes,
                                                 outputCount, outputIndexes,
                                                 outExpectedTypes);
        }
        default:
            return ANEURALNETWORKS_BAD_DATA;
    }
}
#endif //HiFi_BUILD

ErrorStatus convertResultCodeToErrorStatus(int resultCode) {
    switch (resultCode) {
        case ANEURALNETWORKS_NO_ERROR:
            return ErrorStatus::NONE;

        case ANEURALNETWORKS_BAD_DATA:
        case ANEURALNETWORKS_UNEXPECTED_NULL:
            return ErrorStatus::INVALID_ARGUMENT;

        default:
#ifndef HIFI_BUILD
            LOG(ERROR) << "Unknown result code " << resultCode
                       << " mapped to ErrorStatus::GENERAL_FAILURE";
#endif //HIFI_BUILD
        case ANEURALNETWORKS_BAD_STATE:
        case ANEURALNETWORKS_INCOMPLETE:
        case ANEURALNETWORKS_OP_FAILED:
        case ANEURALNETWORKS_OUT_OF_MEMORY:
        case ANEURALNETWORKS_UNMAPPABLE:
            return ErrorStatus::GENERAL_FAILURE;
    }
}

int convertErrorStatusToResultCode(ErrorStatus status) {
    switch (status) {
        case ErrorStatus::NONE:
            return ANEURALNETWORKS_NO_ERROR;

        case ErrorStatus::INVALID_ARGUMENT:
            return ANEURALNETWORKS_BAD_DATA;

        default:
#ifndef HIFI_BUILD
            LOG(ERROR) << "Unknown ErrorStatus " << toString(status)
                       << " mapped to ANEURALNETWORKS_OP_FAILED";
#endif //HIFI_BUILD
        case ErrorStatus::DEVICE_UNAVAILABLE:
        case ErrorStatus::GENERAL_FAILURE:
        case ErrorStatus::OUTPUT_INSUFFICIENT_SIZE:
            return ANEURALNETWORKS_OP_FAILED;
    }
}

// Versioning

bool compliantWithV1_0(V1_0::OperationType) {
    return true;
}

bool compliantWithV1_0(V1_1::OperationType operation) {
    switch (static_cast<V1_0::OperationType>(operation)) {
        case V1_0::OperationType::ADD:
        case V1_0::OperationType::AVERAGE_POOL_2D:
        case V1_0::OperationType::CONCATENATION:
        case V1_0::OperationType::CONV_2D:
        case V1_0::OperationType::DEPTHWISE_CONV_2D:
        case V1_0::OperationType::DEPTH_TO_SPACE:
        case V1_0::OperationType::DEQUANTIZE:
        case V1_0::OperationType::EMBEDDING_LOOKUP:
        case V1_0::OperationType::FLOOR:
        case V1_0::OperationType::FULLY_CONNECTED:
        case V1_0::OperationType::HASHTABLE_LOOKUP:
        case V1_0::OperationType::L2_NORMALIZATION:
        case V1_0::OperationType::L2_POOL_2D:
        case V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case V1_0::OperationType::LOGISTIC:
        case V1_0::OperationType::LSH_PROJECTION:
        case V1_0::OperationType::LSTM:
        case V1_0::OperationType::MAX_POOL_2D:
        case V1_0::OperationType::MUL:
        case V1_0::OperationType::RELU:
        case V1_0::OperationType::RELU1:
        case V1_0::OperationType::RELU6:
        case V1_0::OperationType::RESHAPE:
        case V1_0::OperationType::RESIZE_BILINEAR:
        case V1_0::OperationType::RNN:
        case V1_0::OperationType::SOFTMAX:
        case V1_0::OperationType::SPACE_TO_DEPTH:
        case V1_0::OperationType::SVDF:
        case V1_0::OperationType::TANH:
        case V1_0::OperationType::OEM_OPERATION:
            return true;
        default:
            return false;
    }
}

bool compliantWithV1_1(V1_0::OperationType) {
    return true;
}

bool compliantWithV1_1(V1_1::OperationType) {
    return true;
}

bool compliantWithV1_0(V1_0::Capabilities) {
    return true;
}

bool compliantWithV1_0(const V1_1::Capabilities& capabilities) {
    return capabilities.relaxedFloat32toFloat16Performance.execTime ==
           capabilities.float32Performance.execTime
           &&
           capabilities.relaxedFloat32toFloat16Performance.powerUsage ==
           capabilities.float32Performance.powerUsage;
}

bool compliantWithV1_1(const V1_0::Capabilities&) {
    return true;
}

bool compliantWithV1_1(const V1_1::Capabilities&) {
    return true;
}

bool compliantWithV1_0(const V1_0::Operation&) {
    return true;
}

bool compliantWithV1_0(const V1_1::Operation& operation) {
    return compliantWithV1_0(operation.type);
}

bool compliantWithV1_1(const V1_0::Operation&) {
    return true;
}

bool compliantWithV1_1(const V1_1::Operation&) {
    return true;
}

static bool compliantWithV1_0(const hidl_vec<V1_1::Operation>& operations) {
    return std::all_of(operations.begin(), operations.end(),
                       [](const V1_1::Operation& operation) {
                           return compliantWithV1_0(operation);
                       });
}

bool compliantWithV1_0(const V1_0::Model&) {
    return true;
}

bool compliantWithV1_0(const V1_1::Model& model) {
    // In addition to new enumeration values being introduced in V1_1::Model, a
    // new flag was introduced to indicate whether or not float32 data can be
    // calculated using float16 units. This 'relaxComputationFloat32toFloat16'
    // flag is not relevant in whether a V1_1::Model is compliant with a
    // V1_0::Model because all 1.0 drivers require strict calculation by default
    // in the P NN runtime. Even if fp16 calculations are allowed, they can
    // still be computed by a strict fp32 driver.
    return compliantWithV1_0(model.operations);
}

bool compliantWithV1_1(const V1_0::Model&) {
    return true;
}

bool compliantWithV1_1(const V1_1::Model&) {
    return true;
}

V1_0::OperationType convertToV1_0(V1_0::OperationType type) {
    return type;
}

V1_0::OperationType convertToV1_0(V1_1::OperationType type) {
    if (!compliantWithV1_0(type)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Upcasting non-compliant type " << toString(type)
                   << " from V1_1::OperationType to V1_0::OperationType";
#endif //HIFI_BUILD
    }
    return static_cast<V1_0::OperationType>(type);
}

V1_1::OperationType convertToV1_1(V1_0::OperationType type) {
    return static_cast<V1_1::OperationType>(type);
}

V1_1::OperationType convertToV1_1(V1_1::OperationType type) {
    return type;
}

V1_0::Capabilities convertToV1_0(const V1_0::Capabilities& capabilities) {
    return capabilities;
}

V1_0::Capabilities convertToV1_0(const V1_1::Capabilities& capabilities) {
    if (!compliantWithV1_0(capabilities)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Upcasting non-compliant capabilities " << toString(capabilities)
                   << " from V1_1::Capabilities to V1_0::Capabilities";
#endif //HIFI_BUILD
    }
    return { .float32Performance = capabilities.float32Performance,
             .quantized8Performance = capabilities.quantized8Performance };
}

V1_1::Capabilities convertToV1_1(const V1_0::Capabilities& capabilities) {
    return { .float32Performance = capabilities.float32Performance,
             .quantized8Performance = capabilities.quantized8Performance,
             .relaxedFloat32toFloat16Performance = capabilities.float32Performance };
}

V1_1::Capabilities convertToV1_1(const V1_1::Capabilities& capabilities) {
    return capabilities;
}

V1_0::Operation convertToV1_0(const V1_0::Operation& operation) {
    return operation;
}

V1_0::Operation convertToV1_0(const V1_1::Operation& operation) {
    if (!compliantWithV1_0(operation)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Upcasting non-compliant operation " << toString(operation)
                   << " from V1_1::Operation to V1_0::Operation";
#endif //HIFI_BUILD
    }
    return {.type = convertToV1_0(operation.type),
            .inputs = operation.inputs,
            .outputs = operation.outputs};
}

V1_1::Operation convertToV1_1(const V1_0::Operation& operation) {
    return {.type = convertToV1_1(operation.type),
            .inputs = operation.inputs,
            .outputs = operation.outputs};
}

V1_1::Operation convertToV1_1(const V1_1::Operation& operation) {
    return operation;
}

static hidl_vec<V1_0::Operation> convertToV1_0(const hidl_vec<V1_1::Operation>& operations) {
    hidl_vec<V1_0::Operation> result(operations.size());
    std::transform(operations.begin(), operations.end(), result.begin(),
                   [](const V1_1::Operation& operation) { return convertToV1_0(operation); });
    return result;
}

static hidl_vec<V1_1::Operation> convertToV1_1(const hidl_vec<V1_0::Operation>& operations) {
    hidl_vec<V1_1::Operation> result(operations.size());
    std::transform(operations.begin(), operations.end(), result.begin(),
                   [](const V1_0::Operation& operation) { return convertToV1_1(operation); });
    return result;
}

V1_0::Model convertToV1_0(const V1_0::Model& model) {
    return model;
}

V1_0::Model convertToV1_0(const V1_1::Model& model) {
    if (!compliantWithV1_0(model)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Upcasting non-compliant model " << SHOW_IF_DEBUG(toString(model))
                   << " from V1_1::Model to V1_0::Model";
#endif //HIFI_BUILD
    }
    return {.operands = model.operands,
            .operations = convertToV1_0(model.operations),
            .inputIndexes = model.inputIndexes,
            .outputIndexes = model.outputIndexes,
            .operandValues = model.operandValues,
            .pools = model.pools};
}

V1_1::Model convertToV1_1(const V1_0::Model& model) {
    return {.operands = model.operands,
            .operations = convertToV1_1(model.operations),
            .inputIndexes = model.inputIndexes,
            .outputIndexes = model.outputIndexes,
            .operandValues = model.operandValues,
            .pools = model.pools,
            .relaxComputationFloat32toFloat16 = false};
}

V1_1::Model convertToV1_1(const V1_1::Model& model) {
    return model;
}

#ifdef NN_DEBUGGABLE
uint32_t getProp(const char* str, uint32_t defaultValue) {
    const std::string propStr = android::base::GetProperty(str, "");
    if (propStr.size() > 0) {
        return std::stoi(propStr);
    } else {
        return defaultValue;
    }
}
#endif  // NN_DEBUGGABLE
#endif //HIFI_BUILD

} // namespace nn
} // namespace android
