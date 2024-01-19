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

#define LOG_TAG "ValidateHal"

#include "ValidateHal.h"
#include "NeuralNetworks.h"
#include "Utils.h"

#ifndef HIFI_BUILD
#include <android-base/logging.h>
#endif //HIFI_BUILD

namespace android {
namespace nn {

class MemoryAccessVerifier {
public:
    MemoryAccessVerifier(const hidl_vec<hidl_memory>& pools)
        : mPoolCount(pools.size()), mPoolSizes(mPoolCount) {
        for (size_t i = 0; i < mPoolCount; i++) {
            mPoolSizes[i] = pools[i].size();
        }
    }
    bool validate(const DataLocation& location) {
        if (location.poolIndex >= mPoolCount) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid poolIndex " << location.poolIndex << "/" << mPoolCount;
#endif //HIFI_BUILD
            return false;
        }
        const size_t size = mPoolSizes[location.poolIndex];
        // Do the addition using size_t to avoid potential wrap-around problems.
        if (static_cast<size_t>(location.offset) + location.length > size) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Reference to pool " << location.poolIndex << " with offset "
                       << location.offset << " and length " << location.length
                       << " exceeds pool size of " << size;
#endif //HIFI_BUILD
            return false;
        }
        return true;
    }

private:
    size_t mPoolCount;
    std::vector<size_t> mPoolSizes;
};

static bool validateOperands(const hidl_vec<Operand>& operands,
                             const hidl_vec<uint8_t>& operandValues,
                             const hidl_vec<hidl_memory>& pools) {
    uint32_t index = 0;
    (void)index; /* To remove LLVM15 warning */
    MemoryAccessVerifier poolVerifier(pools);
    for (auto& operand : operands) {
        // Validate type and dimensions.
        switch (operand.type) {
            case OperandType::FLOAT32:
            case OperandType::INT32:
            case OperandType::UINT32:
            case OperandType::OEM: {
                size_t count = operand.dimensions.size();
                if (count != 0) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Scalar data has dimensions of rank "
                               << count;
#endif //HIFI_BUILD
                    return false;
                }
                break;
            }
            case OperandType::TENSOR_FLOAT32:
            case OperandType::TENSOR_INT32:
            case OperandType::TENSOR_QUANT8_ASYMM:
            case OperandType::TENSOR_OEM_BYTE: {
                if (operand.dimensions.size() == 0) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Tensor has dimensions of rank 0";
#endif //HIFI_BUILD
                    return false;
                }
                break;
            }
            default:
#ifndef HIFI_BUILD
                LOG(ERROR) << "Operand " << index << ": Invalid operand type "
                           << toString(operand.type);
#endif //HIFI_BUILD
                return false;
        }

        // TODO Validate the numberOfConsumers.
        // TODO Since we have to validate it, there was no point in including it. For the next
        // release, consider removing unless we have an additional process in system space
        // that creates this value. In that case, it would not have to be validated.

        // Validate the scale.
        switch (operand.type) {
            case OperandType::FLOAT32:
            case OperandType::INT32:
            case OperandType::UINT32:
            case OperandType::TENSOR_FLOAT32:
                if (operand.scale != 0.f) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Operand of type "
                               << getOperandTypeName(operand.type) << " with a non-zero scale ("
                               << operand.scale << ")";
#endif //HIFI_BUILD
                    return false;
                }
                break;
            case OperandType::TENSOR_INT32:
                // TENSOR_INT32 may be used with or without scale, depending on the operation.
                if (operand.scale < 0.f) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Operand of type "
                               << getOperandTypeName(operand.type) << " with a negative scale";
#endif //HIFI_BUILD
                    return false;
                }
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
                if (operand.scale <= 0.f) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Operand of type "
                               << getOperandTypeName(operand.type) << " with a non-positive scale";
#endif //HIFI_BUILD
                    return false;
                }
                break;
            default:
                // No validation for the OEM types.
                // TODO We should have had a separate type for TENSOR_INT32 that a scale
                // and those who don't.  Document now and fix in the next release.
                break;
        }

        // Validate the zeroPoint.
        switch (operand.type) {
            case OperandType::FLOAT32:
            case OperandType::INT32:
            case OperandType::UINT32:
            case OperandType::TENSOR_FLOAT32:
            case OperandType::TENSOR_INT32:
                if (operand.zeroPoint != 0) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Operand of type "
                               << getOperandTypeName(operand.type) << " with an non-zero zeroPoint "
                               << operand.zeroPoint;
#endif //HIFI_BUILD
                    return false;
                }
                break;
            case OperandType::TENSOR_QUANT8_ASYMM:
                if (operand.zeroPoint < 0 || operand.zeroPoint > 255) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Operand of type "
                               << getOperandTypeName(operand.type) << " with an invalid zeroPoint "
                               << operand.zeroPoint << ", must be in range [0, 255]";
#endif //HIFI_BUILD
                    return false;
                }
                break;
            default:
                // No validation for the OEM types.
                break;
        }

        // Validate the lifetime and the location.
        const DataLocation& location = operand.location;
        switch (operand.lifetime) {
            case OperandLifeTime::CONSTANT_COPY:
                if (location.poolIndex != 0) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index
                               << ": CONSTANT_COPY with a non-zero poolIndex "
                               << location.poolIndex;
#endif //HIFI_BUILD
                    return false;
                }
                // Do the addition using size_t to avoid potential wrap-around problems.
                if (static_cast<size_t>(location.offset) + location.length > operandValues.size()) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index
                               << ": OperandValue location out of range.  Starts at "
                               << location.offset << ", length " << location.length << ", max "
                               << operandValues.size();
#endif //HIFI_BUILD
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
                if (!poolVerifier.validate(location)) {
                    return false;
                }
                break;
            case OperandLifeTime::TEMPORARY_VARIABLE:
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                if (location.poolIndex != 0 || location.offset != 0 || location.length != 0) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": Unexpected poolIndex "
                               << location.poolIndex << ", offset " << location.offset
                               << ", or length " << location.length << " for operand of lifetime "
                               << toString(operand.lifetime);
#endif //HIFI_BUILD
                    return false;
                }
                break;
            default:
#ifndef HIFI_BUILD
                LOG(ERROR) << "Operand " << index << ": Invalid lifetime "
                           << toString(operand.lifetime);
#endif //HIFI_BUILD
                return false;
        }

        // For constants, validate that the length is as expected. The other lifetimes
        // expect the length to be 0. Don't validate for OEM types.
        if (operand.lifetime == OperandLifeTime::CONSTANT_REFERENCE ||
            operand.lifetime == OperandLifeTime::CONSTANT_COPY) {
            if (operand.type != OperandType::OEM &&
                operand.type != OperandType::TENSOR_OEM_BYTE) {
                uint32_t expectedLength = sizeOfData(operand.type, operand.dimensions);
                if (location.length != expectedLength) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Operand " << index << ": For operand " << toString(operand)
                               << " expected a size of " << expectedLength << " but got "
                               << location.length;
#endif //HIFI_BUILD
                    return false;
                }
            }
        }

        index++;
    }
    return true;
}

static bool validOperationType(V1_0::OperationType operation) {
    switch (operation) {
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

static bool validOperationType(V1_1::OperationType operation) {
    switch (operation) {
        case V1_1::OperationType::ADD:
        case V1_1::OperationType::AVERAGE_POOL_2D:
        case V1_1::OperationType::CONCATENATION:
        case V1_1::OperationType::CONV_2D:
        case V1_1::OperationType::DEPTHWISE_CONV_2D:
        case V1_1::OperationType::DEPTH_TO_SPACE:
        case V1_1::OperationType::DEQUANTIZE:
        case V1_1::OperationType::EMBEDDING_LOOKUP:
        case V1_1::OperationType::FLOOR:
        case V1_1::OperationType::FULLY_CONNECTED:
        case V1_1::OperationType::HASHTABLE_LOOKUP:
        case V1_1::OperationType::L2_NORMALIZATION:
        case V1_1::OperationType::L2_POOL_2D:
        case V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case V1_1::OperationType::LOGISTIC:
        case V1_1::OperationType::LSH_PROJECTION:
        case V1_1::OperationType::LSTM:
        case V1_1::OperationType::MAX_POOL_2D:
        case V1_1::OperationType::MUL:
        case V1_1::OperationType::RELU:
        case V1_1::OperationType::RELU1:
        case V1_1::OperationType::RELU6:
        case V1_1::OperationType::RESHAPE:
        case V1_1::OperationType::RESIZE_BILINEAR:
        case V1_1::OperationType::RNN:
        case V1_1::OperationType::SOFTMAX:
        case V1_1::OperationType::SPACE_TO_DEPTH:
        case V1_1::OperationType::SVDF:
        case V1_1::OperationType::TANH:
        case V1_1::OperationType::BATCH_TO_SPACE_ND:
        case V1_1::OperationType::DIV:
        case V1_1::OperationType::MEAN:
        case V1_1::OperationType::PAD:
        case V1_1::OperationType::SPACE_TO_BATCH_ND:
        case V1_1::OperationType::SQUEEZE:
        case V1_1::OperationType::STRIDED_SLICE:
        case V1_1::OperationType::SUB:
        case V1_1::OperationType::TRANSPOSE:
        case V1_1::OperationType::OEM_OPERATION:
            return true;
        default:
            return false;
    }
}

template<typename VersionedOperation>
static bool validateOperations(const hidl_vec<VersionedOperation>& operations,
                               const hidl_vec<Operand>& operands) {
    const size_t operandCount = operands.size();
    // This vector keeps track of whether there's an operation that writes to
    // each operand. It is used to validate that temporary variables and
    // model outputs will be written to.
    std::vector<bool> writtenTo(operandCount, false);
    for (auto& op : operations) {
        if (!validOperationType(op.type)) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid operation type " << toString(op.type);
#endif //HIFI_BUILD
            return false;
        }
        // TODO Validate the shapes and any known values. This is currently
        // done in CpuExecutor but should be done here for all drivers.
        int error =
            validateOperation(static_cast<int32_t>(op.type), op.inputs.size(),
                              op.inputs.size() > 0 ? op.inputs.data() : nullptr, op.outputs.size(),
                              op.outputs.size() > 0 ? op.outputs.data() : nullptr, operands);
        if (error != ANEURALNETWORKS_NO_ERROR) {
            return false;
        }

        for (uint32_t i : op.outputs) {
            const Operand& operand = operands[i];
            if (operand.lifetime != OperandLifeTime::TEMPORARY_VARIABLE &&
                operand.lifetime != OperandLifeTime::MODEL_OUTPUT) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Writing to an operand with incompatible lifetime "
                           << toString(operand.lifetime);
#endif //HIFI_BUILD
                return false;
            }

            // Check that we only write once to an operand.
            if (writtenTo[i]) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Operand " << i << " written a second time";
#endif //HIFI_BUILD
                return false;
            }
            writtenTo[i] = true;
        }
    }
    for (size_t i = 0; i < operandCount; i++) {
        if (!writtenTo[i]) {
            const Operand& operand = operands[i];
            if (operand.lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
                operand.lifetime == OperandLifeTime::MODEL_OUTPUT) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Operand " << i << " with lifetime " << toString(operand.lifetime)
                           << " is not being written to.";
#endif //HIFI_BUILD
                return false;
            }
        }
    }
    // TODO More whole graph verifications are possible, for example that an
    // operand is not use as input & output for the same op, and more
    // generally that it is acyclic.
    return true;
}

static bool validatePools(const hidl_vec<hidl_memory>& pools) {
    for (const hidl_memory& memory : pools) {
        const auto name = memory.name();
        if (name != "ashmem" && name != "mmap_fd") {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Unsupported memory type " << name;
#endif //HIFI_BUILD
            return false;
        }
        if (memory.handle() == nullptr) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Memory of type " << name << " is null";
#endif //HIFI_BUILD
            return false;
        }
    }
    return true;
}

static bool validateModelInputOutputs(const hidl_vec<uint32_t> indexes,
                                      const hidl_vec<Operand>& operands, OperandLifeTime lifetime) {
    const size_t operandCount = operands.size();
    for (uint32_t i : indexes) {
        if (i >= operandCount) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Model input or output index out of range: " << i << "/" << operandCount;
#endif //HIFI_BUILD
            return false;
        }
        const Operand& operand = operands[i];
        if (operand.lifetime != lifetime) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Model input or output has lifetime of " << toString(operand.lifetime)
                       << " instead of the expected " << toString(lifetime);
#endif //HIFI_BUILD
            return false;
        }
    }

    std::vector<uint32_t> sortedIndexes = indexes;
    std::sort(sortedIndexes.begin(), sortedIndexes.end());
    auto adjacentI = std::adjacent_find(sortedIndexes.begin(), sortedIndexes.end());
    if (adjacentI != sortedIndexes.end()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Model input or output occurs multiple times: " << *adjacentI;
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

template<typename VersionedModel>
static bool validateModelVersioned(const VersionedModel& model) {
    return (validateOperands(model.operands, model.operandValues, model.pools) &&
            validateOperations(model.operations, model.operands) &&
            validateModelInputOutputs(model.inputIndexes, model.operands,
                                      OperandLifeTime::MODEL_INPUT) &&
            validateModelInputOutputs(model.outputIndexes, model.operands,
                                      OperandLifeTime::MODEL_OUTPUT) &&
            validatePools(model.pools));
}

bool validateModel(const V1_0::Model& model) {
    return validateModelVersioned(model);
}

bool validateModel(const V1_1::Model& model) {
    return validateModelVersioned(model);
}

// Validates the arguments of a request. type is either "input" or "output" and is used
// for printing error messages. The operandIndexes is the appropriate array of input
// or output operand indexes that was passed to the ANeuralNetworksModel_identifyInputsAndOutputs.
static bool validateRequestArguments(const hidl_vec<RequestArgument>& requestArguments,
                                     const hidl_vec<uint32_t>& operandIndexes,
                                     const hidl_vec<Operand>& operands,
                                     const hidl_vec<hidl_memory>& pools, const char* type) {
    MemoryAccessVerifier poolVerifier(pools);
    // The request should specify as many arguments as were described in the model.
    const size_t requestArgumentCount = requestArguments.size();
    if (requestArgumentCount != operandIndexes.size()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Request specifies " << requestArgumentCount << " " << type
                   << "s but the model has " << operandIndexes.size();
#endif //HIFI_BUILD
        return false;
    }
    for (size_t requestArgumentIndex = 0; requestArgumentIndex < requestArgumentCount;
         requestArgumentIndex++) {
        const RequestArgument& requestArgument = requestArguments[requestArgumentIndex];
        const DataLocation& location = requestArgument.location;
        // Get the operand index for this argument. We extract it from the list
        // that was provided in the call to ANeuralNetworksModel_identifyInputsAndOutputs.
        // We assume in this function that the model has been validated already.
        const uint32_t operandIndex = operandIndexes[requestArgumentIndex];
        const Operand& operand = operands[operandIndex];
        if (requestArgument.hasNoValue) {
            if (location.poolIndex != 0 || location.offset != 0 || location.length != 0 ||
                requestArgument.dimensions.size() != 0) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Request " << type << " " << requestArgumentIndex
                           << " has no value yet has details.";
#endif //HIFI_BUILD
                return false;
            }
        } else {
            // Validate the location.
            if (!poolVerifier.validate(location)) {
                return false;
            }
            // If the argument specified a dimension, validate it.
            uint32_t rank = requestArgument.dimensions.size();
            if (rank == 0) {
                // Validate that all the dimensions are specified in the model.
                for (size_t i = 0; i < operand.dimensions.size(); i++) {
                    if (operand.dimensions[i] == 0) {
#ifndef HIFI_BUILD
                        LOG(ERROR) << "Model has dimension " << i
                                   << " set to 0 but the request does specify the dimension.";
#endif //HIFI_BUILD
                        return false;
                    }
                }
            } else {
                if (rank != operand.dimensions.size()) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << "Request " << type << " " << requestArgumentIndex
                               << " has number of dimensions (" << rank
                               << ") different than the model's (" << operand.dimensions.size()
                               << ")";
#endif //HIFI_BUILD
                    return false;
                }
                for (size_t i = 0; i < rank; i++) {
                    if (requestArgument.dimensions[i] != operand.dimensions[i] &&
                        operand.dimensions[i] != 0) {
#ifndef HIFI_BUILD
                        LOG(ERROR) << "Request " << type << " " << requestArgumentIndex
                                   << " has dimension " << i << " of "
                                   << requestArgument.dimensions[i]
                                   << " different than the model's " << operand.dimensions[i];
#endif //HIFI_BUILD
                        return false;
                    }
                    if (requestArgument.dimensions[i] == 0) {
#ifndef HIFI_BUILD
                        LOG(ERROR) << "Request " << type << " " << requestArgumentIndex
                                   << " has dimension " << i << " of zero";
#endif //HIFI_BUILD
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template<typename VersionedModel>
static bool validateRequestVersioned(const Request& request, const VersionedModel& model) {
    return (validateRequestArguments(request.inputs, model.inputIndexes, model.operands,
                                     request.pools, "input") &&
            validateRequestArguments(request.outputs, model.outputIndexes, model.operands,
                                     request.pools, "output") &&
            validatePools(request.pools));
}

bool validateRequest(const Request& request, const V1_0::Model& model) {
    return validateRequestVersioned(request, model);
}

bool validateRequest(const Request& request, const V1_1::Model& model) {
    return validateRequestVersioned(request, model);
}

bool validateExecutionPreference(ExecutionPreference preference) {
    return preference == ExecutionPreference::LOW_POWER ||
           preference == ExecutionPreference::FAST_SINGLE_ANSWER ||
           preference == ExecutionPreference::SUSTAINED_SPEED;
}

}  // namespace nn
}  // namespace android
