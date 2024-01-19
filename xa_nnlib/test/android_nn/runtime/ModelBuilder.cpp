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

#define LOG_TAG "ModelBuilder"

#include "ModelBuilder.h"

#include "CompilationBuilder.h"
#include "Utils.h"
#include "ValidateHal.h"

#include <map>
#include <utility>

namespace android {
namespace nn {

// The maximum number of operands and operations that a model may have.
const uint32_t MAX_NUMBER_OF_OPERANDS = 0xFFFFFFFE;
const uint32_t MAX_NUMBER_OF_OPERATIONS = 0xFFFFFFFE;

bool ModelBuilder::badState(const char* name) {
    if (mCompletedModel) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_" << name << " can't modify after model finished";
#endif //HIFI_BUILD
        return true;
    }
    if (mInvalidModel) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_" << name << " can't modify an invalid model";
#endif //HIFI_BUILD
        return true;
    }
    return false;
}

int ModelBuilder::addOperand(const ANeuralNetworksOperandType& type) {
    if (badState("addOperand")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    int n = validateOperandType(type, "ANeuralNetworksModel_addOperand", true);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    size_t idx = mOperands.size();
    if (idx >= MAX_NUMBER_OF_OPERANDS) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_addOperand exceed max operands";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    mOperands.push_back({
        .type = static_cast<OperandType>(type.type),
        .dimensions = hidl_vec<uint32_t>(type.dimensions, type.dimensions + type.dimensionCount),
        .numberOfConsumers = 0,
        .scale = type.scale,
        .zeroPoint = type.zeroPoint,
        .lifetime = OperandLifeTime::TEMPORARY_VARIABLE,
        .location = {.poolIndex = 0, .offset = 0, .length = 0},
    });
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValue(uint32_t index, const void* buffer, size_t length) {
#ifndef HIFI_BUILD
    VLOG(MODEL) << __func__ << " for operand " << index << " size " << length;
#endif //HIFI_BUILD
    if (badState("setOperandValue")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (index >= operandCount()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting operand " << index << " of "
                   << operandCount();
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    Operand& operand = mOperands[index];
    if (buffer == nullptr) {
        if (length) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "ANeuralNetworksModel_setOperandValue buffer is nullptr but length is "
                          "not 0";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
        operand.lifetime = OperandLifeTime::NO_VALUE;
        // The location is unused and is set to zeros.
        operand.location = {.poolIndex = 0,
                            .offset = 0,
                            .length = 0};
    } else {
        if (length > 0xFFFFFFFF) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "ANeuralNetworksModel_setOperandValue value length of " << length
                       << " exceeds max size";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
        uint32_t valueLength = static_cast<uint32_t>(length);
        uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
        if (operand.type != OperandType::OEM && neededLength != valueLength) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "ANeuralNetworksModel_setOperandValue setting " << valueLength
                       << " bytes when needing " << neededLength;
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }

#ifndef HIFI_BUILD
        if (valueLength <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) 
#else
        if(1) /* Avoiding shared memory use for large values, not supported as of now */
#endif //HIFI_BUILD
        {
            uint32_t existingSize = static_cast<uint32_t>(mSmallOperandValues.size());
            uint32_t extraBytes = alignBytesNeeded(existingSize, valueLength);
            mSmallOperandValues.resize(existingSize + extraBytes + valueLength);
            operand.lifetime = OperandLifeTime::CONSTANT_COPY;
            operand.location = {
                .poolIndex = 0, .offset = existingSize + extraBytes, .length = valueLength};
            memcpy(&mSmallOperandValues[operand.location.offset], buffer, valueLength);
#ifndef HIFI_BUILD
            VLOG(MODEL) << "Copied small value to offset " << operand.location.offset;
#endif //HIFI_BUILD
        } else {
#ifndef HIFI_BUILD
            VLOG(MODEL) << "Saving large value";
#endif //HIFI_BUILD
            operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
            // The values for poolIndex and offset will be set when the model is finished.
            typedef decltype(operand.location.poolIndex) PoolIndexType;
            typedef decltype(operand.location.offset) OffsetType;
            operand.location = {.poolIndex = ~PoolIndexType(0), .offset = ~OffsetType(0),
                                .length = valueLength};
            // We keep track of the buffers. We'll allocate the shared memory only
            // once we know the total size, to avoid needless copies.
            mLargeOperandValues.push_back(LargeValue{.operandIndex = index, .buffer = buffer});
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::copyLargeValuesToSharedMemory() {
#ifndef HIFI_BUILD
    VLOG(MODEL) << __func__ << " has " << mLargeOperandValues.size() << " values.";
#endif //HIFI_BUILD
#ifndef HIFI_BUILD
    //TODO: need to support this
    if (!mLargeOperandValues.empty()) {
        // Calculate the size of the shared memory needed for all the large values.
        // Also sets the offset for each value within the memory.
        size_t poolSize = 0;
        for (LargeValue& l: mLargeOperandValues) {
            Operand& operand = mOperands[l.operandIndex];
            nnAssert(operand.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
            poolSize += alignBytesNeeded(poolSize, operand.location.length);
            operand.location.offset = poolSize;
            poolSize += operand.location.length;
        }

        // Allocated the shared memory.
        int n = mLargeValueMemory.create(poolSize);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
        uint8_t* memoryPointer = nullptr;
        n = mLargeValueMemory.getPointer(&memoryPointer);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
        uint32_t poolIndex = mMemories.add(&mLargeValueMemory);
#ifndef HIFI_BUILD
        VLOG(MODEL) << "Allocated large value pool of size " << poolSize << " at index "
                    << poolIndex;
#endif //HIFI_BUILD

        // Copy the values to this memory.
        for (LargeValue& l: mLargeOperandValues) {
            Operand& operand = mOperands[l.operandIndex];
            operand.location.poolIndex = poolIndex;
            memcpy(memoryPointer + operand.location.offset, l.buffer, operand.location.length);
        }
    }
#endif //HIFI_BUILD
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::setOperandValueFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                                            size_t length) {
#ifndef HIFI_BUILD
    VLOG(MODEL) << __func__ << " for operand " << index << " offset " << offset << " size " << length;
#endif //HIFI_BUILD
    if (badState("setOperandValueFromMemory")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (index >= operandCount()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting operand " << index
                   << " of " << operandCount();
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    Operand& operand = mOperands[index];
    uint32_t neededLength = sizeOfData(operand.type, operand.dimensions);
    if (neededLength != length) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValueFromMemory setting " << length
                   << " bytes when needing " << neededLength;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (!memory->validateSize(offset, length)) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
    operand.location = {
                .poolIndex = mMemories.add(memory), .offset = offset, .length = neededLength};
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::addOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                               const uint32_t* inputs, uint32_t outputCount,
                               const uint32_t* outputs) {
    if (badState("addOperation")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (!validCode(kNumberOfOperationTypes, kNumberOfOperationTypesOEM, type)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_addOperation invalid operations type " << type;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    int n = validateOperation(type, inputCount, inputs,
                              outputCount, outputs, mOperands);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    uint32_t operationIndex = operationCount();
    if (operationIndex >= MAX_NUMBER_OF_OPERATIONS) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_addOperation exceed max operations";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }

    mOperations.push_back({
        .type = static_cast<OperationType>(type),
        .inputs = hidl_vec<uint32_t>(inputs, inputs + inputCount),
        .outputs = hidl_vec<uint32_t>(outputs, outputs + outputCount),
    });
    for (uint32_t i : mOperations.back().inputs) {
        mOperands[i].numberOfConsumers++;
    }
    mHasOEMOperation |= (mOperations.back().type == OperationType::OEM_OPERATION);

    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs,
                                      uint32_t outputCount, const uint32_t* outputs) {
    if (badState("identifyInputsAndOutputs")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    int n = validateOperandList(inputCount, inputs, operandCount(),
                                "ANeuralNetworksModel_identifyInputsAndOutputs inputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = validateOperandList(outputCount, outputs, operandCount(),
                            "ANeuralNetworksModel_identifyInputsAndOutputs outputs");
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // Makes a copy of the index list, validates the arguments, and changes
    // the lifetime info of the corresponding operand.
    auto setArguments = [&](std::vector<uint32_t>* indexVector, uint32_t indexCount,
                            const uint32_t* indexList, OperandLifeTime lifetime) -> bool {
        indexVector->resize(indexCount);
        for (uint32_t i = 0; i < indexCount; i++) {
            const uint32_t operandIndex = indexList[i];
            if (operandIndex >= mOperands.size()) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "ANeuralNetworksModel_identifyInputsAndOutputs Can't set input or output "
                              "to be "
                           << operandIndex << " as this exceeds the numbe of operands "
                           << mOperands.size();
#endif //HIFI_BUILD
                return false;
            }
            (*indexVector)[i] = operandIndex;
            Operand& operand = mOperands[operandIndex];
            if (operand.lifetime != OperandLifeTime::TEMPORARY_VARIABLE) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "ANeuralNetworksModel_identifyInputsAndOutputs Can't set operand "
                           << operandIndex
                           << " to be an input or output.  Check that it's not a constant or "
                              "already an input or output";
#endif //HIFI_BUILD
                return false;
            }
            operand.lifetime = lifetime;
        }
        return true;
    };

    if (!setArguments(&mInputIndexes, inputCount, inputs, OperandLifeTime::MODEL_INPUT) ||
        !setArguments(&mOutputIndexes, outputCount, outputs, OperandLifeTime::MODEL_OUTPUT)) {
        return ANEURALNETWORKS_BAD_DATA;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::relaxComputationFloat32toFloat16(bool allow) {
    if (badState("relaxComputationFloat32toFloat16")) {
        return ANEURALNETWORKS_BAD_STATE;
    }

    mRelaxComputationFloat32toFloat16 = allow;

    return ANEURALNETWORKS_NO_ERROR;
}

int ModelBuilder::createCompilation(CompilationBuilder** compilation) {
    if (!mCompletedModel || mInvalidModel) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksCompilation_create passed an unfinished or invalid model";
#endif //HIFI_BUILD
        *compilation = nullptr;
        return ANEURALNETWORKS_BAD_STATE;
    }
    *compilation = new (std::nothrow) CompilationBuilder(this);
    return (*compilation ? ANEURALNETWORKS_NO_ERROR : ANEURALNETWORKS_OUT_OF_MEMORY);
}

int ModelBuilder::finish() {
    if (mCompletedModel) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_finish called more than once";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_STATE;
    }
    if (mInvalidModel) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_finish called on an invalid model";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_STATE;
    }

    int n = copyLargeValuesToSharedMemory();
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // TODO: Modify validation so that it can be called without creating a HAL Model.
    // NOTE: Must copyLargeValuesToSharedMemory() before validation; otherwise,
    //       a CONSTANT_REFERENCE operand will not have correct .poolIndex, and
    //       validation will not work properly.
    Model modelForValidation;
    setHidlModel(&modelForValidation);
    if (!validateModel(modelForValidation)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_finish called on invalid model";
#endif //HIFI_BUILD
        mInvalidModel = true;
        return ANEURALNETWORKS_BAD_DATA;
    }

    // We sort the operations so that they will be in the appropriate
    // order for a single-threaded, op at a time execution.
    // TODO: we don't need this if we always run the partitioner.
    sortIntoRunOrder();
    mCompletedModel = true;
    return ANEURALNETWORKS_NO_ERROR;
}

void ModelBuilder::sortIntoRunOrder() {
    // Tracks the operations that can be executed.
    std::vector<uint32_t> opsReadyToRun;
    std::vector<Operation> runOrder;

    // Tracks how many inputs are needed for each operation to be ready to run.
    std::multimap<uint32_t, uint32_t> operandToOperations;
    std::vector<uint32_t> unknownInputCount(operationCount());
    for (uint32_t operationIndex = 0; operationIndex < operationCount(); operationIndex++) {
        uint32_t& count = unknownInputCount[operationIndex];
        count = 0;
        for (uint32_t operandIndex : mOperations[operationIndex].inputs) {
            auto lifetime = mOperands[operandIndex].lifetime;
            if (lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
                lifetime == OperandLifeTime::MODEL_OUTPUT) {
                count++;
                operandToOperations.insert(
                            std::pair<uint32_t, uint32_t>(operandIndex, operationIndex));
            }
        }
        if (count == 0) {
            opsReadyToRun.push_back(operationIndex);
        }
    }

    while (opsReadyToRun.size() > 0) {
        // Execute the next op
        int opIndex = opsReadyToRun.back();
        opsReadyToRun.pop_back();
        const Operation& operation = mOperations[opIndex];

        runOrder.push_back(mOperations[opIndex]);

        // Mark all its outputs as known.
        for (uint32_t operandIndex : operation.outputs) {
            auto range = operandToOperations.equal_range(operandIndex);
            for (auto i = range.first; i != range.second; i++) {
                uint32_t& count = unknownInputCount[i->second];
                if (--count == 0) {
                    opsReadyToRun.push_back(i->second);
                }
            }
        }
    }
    mOperations = runOrder;
}

void ModelBuilder::setHidlModel(Model* model) const {
    model->operands = mOperands;
    model->operations = mOperations;
    model->inputIndexes = mInputIndexes;
    model->outputIndexes = mOutputIndexes;
    model->operandValues = mSmallOperandValues;
    model->relaxComputationFloat32toFloat16 = mRelaxComputationFloat32toFloat16;

    uint32_t count = mMemories.size();
    model->pools.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        model->pools[i] = mMemories[i]->getHidlMemory();
    }
}

}  // namespace nn
}  // namespace android
