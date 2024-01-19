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

// Class used to build a model through a succession of successive calls
// to the NN API.

#ifndef ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H
#define ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H

#include "HalInterfaces.h"
#include "Memory.h"
#include "NeuralNetworks.h"
#include "Utils.h"

#ifndef HIFI_BUILD
#else
#include <memory>
#endif //HIFI_BUILD

namespace android {
namespace nn {

class CompilationBuilder;
class Device;
class ExecutionPlan;
class Memory;

class ModelBuilder {
public:
    // Adds an operand to the model.
    int addOperand(const ANeuralNetworksOperandType& type);
    int setOperandValue(uint32_t index, const void* buffer, size_t length);
    int setOperandValueFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                                  size_t length);

    int addOperation(ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs,
                     uint32_t outputCount, const uint32_t* outputs);
    int identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
                                 const uint32_t* outputs);
    int relaxComputationFloat32toFloat16(bool allow);
    bool isComputationFloat32RelaxedToFloat16() const { return mRelaxComputationFloat32toFloat16; }

    int finish();
    bool isFinished() const { return mCompletedModel; }

    bool hasOEMOperation() const { return mHasOEMOperation; }

    int createCompilation(CompilationBuilder** compilation);

    void setHidlModel(Model* model) const;

    uint32_t operandCount() const {
        // We don't allow more than uint32_t worth of operands
        return static_cast<uint32_t>(mOperands.size());
    }
    uint32_t operationCount() const {
        // We don't allow more than uint32_t worth of operations
        return static_cast<uint32_t>(mOperations.size());
    }
    uint32_t inputCount() const { return static_cast<uint32_t>(mInputIndexes.size()); }
    uint32_t outputCount() const { return static_cast<uint32_t>(mOutputIndexes.size()); }
    uint32_t getInputOperandIndex(uint32_t i) const { return mInputIndexes[i]; }
    const Operand& getInputOperand(uint32_t i) const {
        return mOperands[getInputOperandIndex(i)];
    }
    uint32_t getOutputOperandIndex(uint32_t i) const { return mOutputIndexes[i]; }
    const Operand& getOutputOperand(uint32_t i) const {
        return mOperands[getOutputOperandIndex(i)];
    }
    const Operand& getOperand(uint32_t index) const { return mOperands[index]; }
    const Operation& getOperation(uint32_t index) const { return mOperations[index]; }
    const MemoryTracker& getMemories() const { return mMemories; }
    const std::vector<Operation>& getOperations() const { return mOperations; }
    const uint8_t* getPointerToOperandValue(uint32_t offset) const {
        return mSmallOperandValues.data() + offset;
    }

    int partitionTheWork(const std::vector<std::shared_ptr<Device>>& devices,
                         uint32_t preference, ExecutionPlan* plan) const;

 private:
    // TODO: move partitionTheWork, findBestDeviceForEachOperation,
    // sortIntoRunOrder to CompilationBuilder?

#ifndef HIFI_BUILD
    int findBestDeviceForEachOperation(uint32_t preference,
                                       const std::vector<std::shared_ptr<Device>>& devices,
                                       const size_t deviceCount,
                                       std::vector<int>* bestDeviceForOperation) const;
    PerformanceInfo getPerformanceInfo(const std::shared_ptr<Device> device,
                                       uint32_t operationIndex) const;
#endif //HIFI_BUILD

    // Return true if either mCompleteModel or mInvalidModel is true.
    bool badState(const char* name);

    // Sorts the operations to be in the correct order for single threaded
    // node-at-a-time execution.
    void sortIntoRunOrder();

    // Copies the large values to a shared memory, if we have any.
    int copyLargeValuesToSharedMemory();

    // The operations of the graph.
    std::vector<Operation> mOperations;
    // Is at least one of those operations an OEM_OPERATION?
    bool mHasOEMOperation = false;
    // The description of the operands of the graph.
    std::vector<Operand> mOperands;
    // Specifies where to find the list of indexes identifying
    // the inputs and outputs of the model.  The offset is into
    // the mOperandIndexes table.
    std::vector<uint32_t> mInputIndexes;
    std::vector<uint32_t> mOutputIndexes;

    MemoryTracker mMemories;

    // The value of the small operands that are defined at model
    // creation time.
    std::vector<uint8_t> mSmallOperandValues;

    struct LargeValue {
        uint32_t operandIndex;
        const void* buffer;
    };
    // Operand index and buffer pointer for all the large operand values of this model.
    std::vector<LargeValue> mLargeOperandValues;
    // The shared memory region that will contain the large values.
    Memory mLargeValueMemory;

    // Once the model has been finished, we should not allow further
    // modifications to the model.
    bool mCompletedModel = false;

    // Any invalid manipulation of the model will mark the model invalid.
    // No further modifications are allowed to the model.
    bool mInvalidModel = false;

    // 'true' indicates TENSOR_FLOAT32 may be calculated with range and/or
    // precision as low as that of the IEEE 754 16-bit floating-point format.
    // 'false' indicates TENSOR_FLOAT32 must be calculated using at least the
    // range and precision of the IEEE 754 32-bit floating-point format.
    bool mRelaxComputationFloat32toFloat16 = false;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_MODEL_BUILDER_H
