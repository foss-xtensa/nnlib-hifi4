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

#define LOG_TAG "ExecutionBuilder"

#include "ExecutionBuilder.h"

#include "CompilationBuilder.h"
#include "CpuExecutor.h"
#include "HalInterfaces.h"
#include "Manager.h"
#include "ModelBuilder.h"
#include "Utils.h"

#include <mutex>
#include <thread>
#include <vector>

namespace android {
namespace nn {

int ModelArgumentInfo::setFromPointer(const Operand& operand,
                                      const ANeuralNetworksOperandType* type, void* data,
                                      uint32_t length) {
    if ((data == nullptr) != (length == 0)) {
#ifndef HIFI_BUILD
        const char* dataPtrMsg = data ? "NOT_NULLPTR" : "NULLPTR";
        LOG(ERROR) << "Data pointer must be nullptr if and only if length is zero (data = "
                   << dataPtrMsg << ", length = " << length << ")";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (data == nullptr) {
        state = ModelArgumentInfo::HAS_NO_VALUE;
    } else {
        int n = updateDimensionInfo(operand, type);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
        uint32_t neededLength = sizeOfData(operand.type, dimensions);
        if (operand.type != OperandType::OEM && neededLength != length) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Setting argument with invalid length: " << length
                       << ", expected length: " << neededLength;
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
        state = ModelArgumentInfo::POINTER;
    }
    buffer = data;
    locationAndLength = {.poolIndex = 0, .offset = 0, .length = length};
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::setFromMemory(const Operand& operand, const ANeuralNetworksOperandType* type,
                                     uint32_t poolIndex, uint32_t offset, uint32_t length) {
    int n = updateDimensionInfo(operand, type);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    uint32_t neededLength = sizeOfData(operand.type, dimensions);
    if (operand.type != OperandType::OEM && neededLength != length) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Setting argument with invalid length: " << length
                   << ", expected length: " << neededLength;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }

    state = ModelArgumentInfo::MEMORY;
    locationAndLength = {.poolIndex = poolIndex, .offset = offset, .length = length};
    buffer = nullptr;
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::setFromTemporaryMemory(const Operand& operand,
                                              uint32_t poolIndex, uint32_t offset) {
    int n = updateDimensionInfo(operand, nullptr);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    state = ModelArgumentInfo::MEMORY;
    locationAndLength =
            {.poolIndex = poolIndex, .offset = offset, .length = sizeOfData(operand)};
    buffer = nullptr;
    return ANEURALNETWORKS_NO_ERROR;
}

int ModelArgumentInfo::updateDimensionInfo(const Operand& operand,
                                           const ANeuralNetworksOperandType* newType) {
    nnAssert(dimensions.empty());
    if (newType == nullptr) {
        for (auto i : operand.dimensions) {
            if (i == 0) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Setting input/output with unspecified dimensions";
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
        }
        dimensions = operand.dimensions;
    } else {
        uint32_t count = newType->dimensionCount;
        if (static_cast<OperandType>(newType->type) != operand.type ||
            count != operand.dimensions.size()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Setting input/output with incompatible types";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }

        dimensions = hidl_vec<uint32_t>(count);
        for (uint32_t i = 0; i < count; i++) {
            if (operand.dimensions[i] != 0 && operand.dimensions[i] != newType->dimensions[i]) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Overriding a fully specified dimension is disallowed";
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            } else {
                dimensions[i] = newType->dimensions[i];
            }
        }
    }
    return ANEURALNETWORKS_NO_ERROR;
}

ExecutionBuilder::ExecutionBuilder(const CompilationBuilder* compilation) :
        mModel(compilation->mModel),
        mPlan(&compilation->mPlan),
        mPartitioning(compilation->mPartitioning),
        mInputs(mModel->inputCount()),
        mOutputs(mModel->outputCount()) {
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "ExecutionBuilder::ExecutionBuilder";
#endif //HIFI_BUILD
}

int ExecutionBuilder::setInput(uint32_t index, const ANeuralNetworksOperandType* type,
                               const void* buffer, size_t length) {
    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setInput bad index " << index << " " << count;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type != nullptr) {
        int n = validateOperandType(*type, "ANeuralNetworksExecution_setInput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setInput input exceeds max length " << length;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);
    return mInputs[index].setFromPointer(mModel->getInputOperand(index), type,
                                         const_cast<void*>(buffer), l);
}

int ExecutionBuilder::setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                         const Memory* memory, size_t offset, size_t length) {
    // Should be similar to StepExecutor::setInputOrOutputFromTemporaryMemory()

    uint32_t count = static_cast<uint32_t>(mInputs.size());
    if (index >= count) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setInputFromMemory bad index " << index << " "
                   << count;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (!memory->validateSize(offset, length)) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO validate the rest
    uint32_t poolIndex = mMemories.add(memory);
    return mInputs[index].setFromMemory(mModel->getInputOperand(index), type, poolIndex, offset,
                                        length);
}

int ExecutionBuilder::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                                size_t length) {
    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setOutput bad index " << index << " " << count;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (type != nullptr) {
        int n = validateOperandType(*type, "ANeuralNetworksExecution_setOutput", false);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    if (length > 0xFFFFFFFF) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setOutput input exceeds max length " << length;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    uint32_t l = static_cast<uint32_t>(length);
    return mOutputs[index].setFromPointer(mModel->getOutputOperand(index), type, buffer, l);
}

int ExecutionBuilder::setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                          const Memory* memory, size_t offset, size_t length) {
    // Should be similar to StepExecutor::setInputOrOutputFromTemporaryMemory()

    uint32_t count = static_cast<uint32_t>(mOutputs.size());
    if (index >= count) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setOutputFromMemory bad index " << index << " "
                   << count;
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    if (!memory->validateSize(offset, length)) {
        return ANEURALNETWORKS_BAD_DATA;
    }
    // TODO validate the rest
    uint32_t poolIndex = mMemories.add(memory);
    return mOutputs[index].setFromMemory(mModel->getOutputOperand(index), type, poolIndex, offset,
                                         length);
}

// Attempt synchronous execution of full model on CPU.
// Ensure that executionCallback->notify() is called.
#ifndef HIFI_BUILD
#ifndef DISABLE_PARTITIONED_EXECUTION
static void cpuFallbackFull(const ExecutionBuilder* executionBuilder,
                            const sp<ExecutionCallback>& executionCallback) {
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "cpuFallbackFull";
#endif //HIFI_BUILD
    StepExecutor executor(executionBuilder, executionBuilder->getModel(),
                          nullptr /* no VersionedIDevice, so CPU */,
                          nullptr /* no IPreparedModel */);
    executor.mapInputsAndOutputsTrivially();
    sp<ExecutionCallback> fallbackCallback;
    int n = executor.startCompute(&fallbackCallback);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        executionCallback->notify(convertResultCodeToErrorStatus(n));
        return;
    }
    fallbackCallback->wait();
    executionCallback->notify(fallbackCallback->getStatus());
}
#endif //#ifndef HIFI_BUILD
#endif //#ifndef DISABLE_PARTITIONED_EXECUTION

// Attempt synchronous execution on CPU.
// (1) First, attempt to execute this step on CPU.  If successful,
//     return true.  (Do not call executionCallback->notify().)
// (2) If unsuccessful, attempt to execute the full model on CPU,
//     ensure that executionCallback->notify() is called, and return
//     false.
#ifndef HIFI_BUILD
#ifndef DISABLE_PARTITIONED_EXECUTION
static bool cpuFallbackPartial(const ExecutionBuilder* executionBuilder,
                               const ExecutionPlan* plan,
                               std::shared_ptr<ExecutionPlan::Controller> controller,
                               const sp<ExecutionCallback>& executionCallback) {
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "cpuFallbackPartial";
#endif //HIFI_BUILD
    std::shared_ptr<StepExecutor> executor;
    int n = plan->fallback(controller, &executor);
    if (n != ANEURALNETWORKS_NO_ERROR || executor->isCpu()) {
        cpuFallbackFull(executionBuilder, executionCallback);
        return false;
    }
    sp<ExecutionCallback> fallbackCallback;
    if (executor->startComputeOnCpu(&fallbackCallback) != ANEURALNETWORKS_NO_ERROR) {
        cpuFallbackFull(executionBuilder, executionCallback);
        return false;
    }
    fallbackCallback->wait();
    if (fallbackCallback->getStatus() != ErrorStatus::NONE) {
        cpuFallbackFull(executionBuilder, executionCallback);
        return false;
    }
    return true;
}
#endif //#ifndef HIFI_BUILD
#endif //#ifndef DISABLE_PARTITIONED_EXECUTION

#ifndef HIFI_BUILD
#ifndef DISABLE_PARTITIONED_EXECUTION
static void asyncStartComputePartitioned(const ExecutionBuilder* executionBuilder,
                                         const ExecutionPlan* plan,
                                         std::shared_ptr<ExecutionPlan::Controller> controller,
                                         bool allowFallback,
                                         const sp<ExecutionCallback>& executionCallback) {
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "ExecutionBuilder::startCompute (from plan, iteratively)";
#endif //HIFI_BUILD
    while (true) {
        std::shared_ptr<StepExecutor> executor;
#ifndef HIFI_BUILD
        VLOG(EXECUTION) << "looking for next StepExecutor";
#endif //HIFI_BUILD
        int n = plan->next(controller, &executor);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            if (allowFallback) {
                cpuFallbackFull(executionBuilder, executionCallback);
            } 
#ifndef HIFI_BUILD
            else {
                executionCallback->notify(convertResultCodeToErrorStatus(n));
            }
#endif //HIFI_BUILD
            return;
        }
        if (executor == nullptr) {
            executionCallback->notify(ErrorStatus::NONE);
            return;
        }

        sp<ExecutionCallback> stepCallback;
        n = executor->startCompute(&stepCallback);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            if (allowFallback) {
                if (cpuFallbackPartial(executionBuilder, plan, controller, executionCallback)) {
                    // Successfully executed one step on CPU.
                    continue;
                } else {
                    // Either successfully executed entire plan on
                    // CPU, or tried and failed to do so.
                    return;
                }
            } else {
                executionCallback->notify(convertResultCodeToErrorStatus(n));
                return;
            }
        }
#ifndef HIFI_BUILD
        stepCallback->wait();
        ErrorStatus status = stepCallback->getStatus();
        if (status != ErrorStatus::NONE) {
            if (allowFallback) {
                if (cpuFallbackPartial(executionBuilder, plan, controller, executionCallback)) {
                    // Successfully executed one step on CPU.
                    continue;
                } else {
                    // Either successfully executed entire plan on
                    // CPU, or tried and failed to do so.
                    return;
                }
            } else {
                executionCallback->notify(status);
                return;
            }
        }
#endif //HIFI_BUILD
    }
}
#endif //#ifndef DISABLE_PARTITIONED_EXECUTION
#endif //#ifndef HIFI_BUILD

int ExecutionBuilder::startCompute(sp<ExecutionCallback>* synchronizationCallback) {
#ifndef HIFI_BUILD
    *synchronizationCallback = nullptr;
#endif //HIFI_BUILD

    // TODO validate that we have full types for all inputs and outputs,
    // that the graph is not cyclic,

    for (auto& p : mInputs) {
        if (p.state == ModelArgumentInfo::UNSPECIFIED) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "ANeuralNetworksExecution_startCompute not all inputs specified";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }
    for (auto& p : mOutputs) {
        if (p.state == ModelArgumentInfo::UNSPECIFIED) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "ANeuralNetworksExecution_startCompute not all outputs specified";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

#ifndef HIFI_BUILD
#ifndef DISABLE_PARTITIONED_EXECUTION
    {
        // TODO: Remove the non-plan-based path once we've fully integrated ExecutionPlan
        // with the compilation and execution phases of the NN API?  Or retain that path
        // as a fallback in the case of partitioning failure?
        //
        // TODO: Entire plan-based-path should run in an asynchronous thread --
        // take the asynchronous thread logic out of startComputeOnCpu() and use
        // it to wrap the plan-based-path.
        if (mPartitioning > 0) {
            const bool allowFallback = DeviceManager::partitioningAllowsFallback(mPartitioning);
            std::shared_ptr<ExecutionPlan::Controller> controller = mPlan->makeController(this);
            if (controller == nullptr) {
                if (!allowFallback) {
                    return ANEURALNETWORKS_OP_FAILED;
                }
            } else {
                // TODO: use a thread pool

                // Prepare the callback for asynchronous execution.
                // sp<ExecutionCallback> object is returned when the
                // execution has been successfully launched, otherwise a
                // nullptr is returned.  The executionCallback is
                // abstracted in the NN API as an "event".
                sp<ExecutionCallback> executionCallback = new ExecutionCallback();
                std::thread thread(asyncStartComputePartitioned, this, mPlan, controller,
                                   allowFallback,
                                   executionCallback);
                executionCallback->bind_thread(std::move(thread));
                *synchronizationCallback = executionCallback;
                return ANEURALNETWORKS_NO_ERROR;
            }
        }
    }
#else
    {
        // Find a driver that can handle all the operations.
        // TODO: Does not handle CPU fallback (which is tricky because
        //       StepExecutor::startCompute() is designed as
        //       asynchronous).
        // TODO: Does not actually behave asynchronously (because
        //       StepExecutor::startCompute() isn't actually asynchronous
        //       on a device as opposed to a CPU).
        Model hidlModel;
        mModel->setHidlModel(&hidlModel);
        const std::vector<std::shared_ptr<Device>>& devices = DeviceManager::get()->getDrivers();
        for (const auto& device : devices) {
            hidl_vec<bool> supports;
#ifndef HIFI_BUILD
            VLOG(EXECUTION) << "Checking " << device->getName();
#endif //HIFI_BUILD
            device->getSupportedOperations(hidlModel, &supports);
            if (std::find(supports.begin(), supports.end(), false) == supports.end()) {
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << "ExecutionBuilder::startCompute (without plan) on " << device->getName();
#endif //HIFI_BUILD
                StepExecutor executor(this, mModel, device->getInterface(),
                                      nullptr /* no IPreparedModel, so compile */);
                executor.mapInputsAndOutputsTrivially();
                return executor.startCompute(synchronizationCallback);
            }
        }
    }
#endif  // DISABLE_PARTITIONED_EXECUTION
#endif //HIFI_BUILD

    // Run on the CPU.
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "ExecutionBuilder::startCompute (without plan) on CPU";
#endif //HIFI_BUILD
    StepExecutor executor(this, mModel,
                          nullptr /* no VersionedIDevice, so CPU */,
                          nullptr /* no IPreparedModel */);
    executor.mapInputsAndOutputsTrivially();
    return executor.startCompute(synchronizationCallback);
}

// Figures out how to place each of the input or outputs in a buffer. This just does the layout,
// it does not copy data.  Aligns each input a bit.
#ifndef HIFI_BUILD
int StepExecutor::allocatePointerArgumentsToPool(std::vector<ModelArgumentInfo>* args,
                                                 Memory* memory) {
    uint32_t nextPoolIndex = mMemories.size();
    int64_t total = 0;
    for (auto& info : *args) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndLength;
            // TODO Good enough alignment?
            total += alignBytesNeeded(static_cast<uint32_t>(total), loc.length);
            loc.poolIndex = nextPoolIndex;
            loc.offset = static_cast<uint32_t>(total);
            total += loc.length;
        }
    };
    if (total > 0xFFFFFFFF) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_startCompute Size of all inputs or outputs exceeds "
                      "2^32.";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_BAD_DATA;
    }
    hidl_memory hidlMemory;
    if (total > 0) {
        memory->create(total);  // TODO check error
        mMemories.add(memory);
    }
    return ANEURALNETWORKS_NO_ERROR;
}
#endif //HIFI_BUILD

static void setRequestArgumentArray(const std::vector<ModelArgumentInfo>& argumentInfos,
                                     hidl_vec<RequestArgument>* ioInfos) {
    size_t count = argumentInfos.size();
    ioInfos->resize(count);
    for (size_t i = 0; i < count; i++) {
        const auto& info = argumentInfos[i];
        (*ioInfos)[i] = { .hasNoValue = info.state == ModelArgumentInfo::HAS_NO_VALUE,
                          .location = info.locationAndLength,
                          .dimensions = info.dimensions,
                        };
    }
}

StepExecutor::StepExecutor(const ExecutionBuilder* executionBuilder,
                           const ModelBuilder* model,
#ifndef HIFI_BUILD
                           VersionedIDevice* driver, sp<IPreparedModel> preparedModel) :
#else
                           void* driver, void* preparedModel) :
#endif //HIFI_BUILD
    mExecutionBuilder(executionBuilder), mModel(model),
    mDriver(driver), 
#ifndef HIFI_BUILD
    mPreparedModel(preparedModel),
#endif //HIFI_BUILD
    mInputs(model->inputCount()), mOutputs(model->outputCount()) {}

void StepExecutor::mapInputsAndOutputsTrivially() {
    mInputs = mExecutionBuilder->mInputs;
    mOutputs = mExecutionBuilder->mOutputs;
    mMemories = mExecutionBuilder->mMemories;
}

void StepExecutor::mapInputOrOutput(const ModelArgumentInfo& builderInputOrOutput,
                                    ModelArgumentInfo* executorInputOrOutput) {
    *executorInputOrOutput = builderInputOrOutput;
    switch (executorInputOrOutput->state) {
        default:
            nnAssert(!"unexpected ModelArgumentInfo::state");
            break;
        case ModelArgumentInfo::POINTER:
        case ModelArgumentInfo::UNSPECIFIED:
            break;
        case ModelArgumentInfo::MEMORY: {
            const uint32_t builderPoolIndex =
                    builderInputOrOutput.locationAndLength.poolIndex;
            const Memory* memory = mExecutionBuilder->mMemories[builderPoolIndex];
            const uint32_t executorPoolIndex = mMemories.add(memory);
            executorInputOrOutput->locationAndLength.poolIndex =
                    executorPoolIndex;
            break;
        }
    }
}

int StepExecutor::setInputOrOutputFromTemporaryMemory(const Operand& inputOrOutputOperand,
                                                      const Memory* memory, uint32_t offset,
                                                      ModelArgumentInfo* inputOrOutputInfo) {
    // Should be similar to
    //     ExecutionBuilder::setInputFromMemory()
    //     ExecutionBuilder::setOutputFromMemory()

    uint32_t poolIndex = mMemories.add(memory);
    return inputOrOutputInfo->setFromTemporaryMemory(inputOrOutputOperand, poolIndex, offset);
}

#ifndef HIFI_BUILD
static void logArguments(const char* kind, const std::vector<ModelArgumentInfo> &args) {
    for (unsigned i = 0; i < args.size(); i++) {
        const auto& arg = args[i];
        std::string prefix = kind + std::string("[") + std::to_string(i) + "] = ";
        switch (arg.state) {
            case ModelArgumentInfo::POINTER:
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << prefix << "POINTER(" << SHOW_IF_DEBUG(arg.buffer) << ")";
#endif //HIFI_BUILD
                break;
            case ModelArgumentInfo::MEMORY:
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << prefix << "MEMORY("
                                << "pool=" << arg.locationAndLength.poolIndex
                                << ", "
                                << "off=" << arg.locationAndLength.offset
                                << ")";
#endif //HIFI_BUILD
                break;
            case ModelArgumentInfo::HAS_NO_VALUE:
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << prefix << "HAS_NO_VALUE";
#endif //HIFI_BUILD
                break;
            case ModelArgumentInfo::UNSPECIFIED:
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << prefix << "UNSPECIFIED";
#endif //HIFI_BUILD
                break;
            default:
#ifndef HIFI_BUILD
                VLOG(EXECUTION) << prefix << "state(" << arg.state << ")";
#endif //HIFI_BUILD
                break;
        }
    }
}
#endif //#ifndef HIFI_BUILD

int StepExecutor::startCompute(sp<ExecutionCallback>* synchronizationCallback) {
#ifndef HIFI_BUILD
    if (VLOG_IS_ON(EXECUTION)) {
        logArguments("input", mInputs);
        logArguments("output", mOutputs);
    }
    if (mDriver == nullptr) {
        return startComputeOnCpu(synchronizationCallback);
    } else {
        return startComputeOnDevice(synchronizationCallback);
    }
#else
    return startComputeOnCpu(synchronizationCallback);
#endif //HIFI_BUILD
}

#ifndef HIFI_BUILD
int StepExecutor::startComputeOnDevice(sp<ExecutionCallback>* synchronizationCallback) {
    nnAssert(mDriver != nullptr);

    *synchronizationCallback = nullptr;

    // TODO: Remove the mPreparedModel == nullptr case once we've fully integrated
    // ExecutionPlan with the compilation and execution phases of the NN API
    if (mPreparedModel == nullptr) {
        Model model;
        mModel->setHidlModel(&model);

        // TODO Dangerous!  In async, the model will outlive it here. Safe for now
        sp<PreparedModelCallback> preparedModelCallback = new PreparedModelCallback();
        // TODO(butlermichael): Propagate user preference to this point instead of
        // using default value of ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER, or
        // remove this entire block of code since it is a stale path that is only
        // encountered on an #if-removed code.
        ExecutionPreference preference =
            static_cast<ExecutionPreference>(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
        ErrorStatus prepareLaunchStatus = mDriver->prepareModel(model, preference,
                                                                preparedModelCallback);
        if (prepareLaunchStatus != ErrorStatus::NONE) {
            return convertErrorStatusToResultCode(prepareLaunchStatus);
        }

        // Immediately synchronize with callback object for now
        // TODO: change to asynchronous later
        preparedModelCallback->wait();
        ErrorStatus prepareReturnStatus = preparedModelCallback->getStatus();
        mPreparedModel = preparedModelCallback->getPreparedModel();
        if (prepareReturnStatus != ErrorStatus::NONE) {
            return convertErrorStatusToResultCode(prepareReturnStatus);
        }
        if (mPreparedModel == nullptr) {
            return ANEURALNETWORKS_OP_FAILED;
        }
    }

    // We separate the input & output pools so that we reduce the copying done if we
    // do an eventual remoting (hidl_memory->update()).  We could also use it to set
    // protection on read only memory but that's not currently done.
    Memory inputPointerArguments;
    Memory outputPointerArguments;

    // Layout the input and output data
    int n = allocatePointerArgumentsToPool(&mInputs, &inputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    n = allocatePointerArgumentsToPool(&mOutputs, &outputPointerArguments);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    // Copy the input data that was specified via a pointer.
    // inputPointerArguments.update();
    for (auto& info : mInputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndLength;
            uint8_t* data = nullptr;
            int n = inputPointerArguments.getPointer(&data);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                return n;
            }
            memcpy(data + loc.offset, info.buffer, loc.length);
        }
    }
    // TODO: Add inputPointerArguments.commit() and .update() at all the right places

    Request request;
    setRequestArgumentArray(mInputs, &request.inputs);
    setRequestArgumentArray(mOutputs, &request.outputs);
    uint32_t count = mMemories.size();
    request.pools.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        request.pools[i] = mMemories[i]->getHidlMemory();
    }

    // Prepare the callback for asynchronous execution. sp<ExecutionCallback>
    // object is returned when the execution has been successfully launched,
    // otherwise a nullptr is returned. The executionCallback is abstracted in
    // the NN API as an "event".
    //
    // The sp is used for ref-counting purposes. Without it, the HIDL service
    // could attempt to communicate with a dead callback object.
    //
    // TODO: Explain the "dead callback" problem further, either here or
    // in the design document.
    sp<ExecutionCallback> executionCallback = new ExecutionCallback();

#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "Before mPreparedModel->execute() " << SHOW_IF_DEBUG(toString(request));
#endif //HIFI_BUILD
    // Execute.
    // TODO: What happens to the Callback if the service dies abnormally
    // -- won't that keep the Callback live forever, because the service
    // never has the opportunity to bump the reference count down? Or
    // maybe the HIDL infrastructure handles this magically? At worst,
    // it seems like this is a small memory leak, if the Callback stays
    // alive forever.
    Return<ErrorStatus> executeStatus = mPreparedModel->execute(request, executionCallback);
    if (!executeStatus.isOk() || executeStatus != ErrorStatus::NONE) {
#ifndef HIFI_BUILD
        VLOG(EXECUTION) << "**Execute failed**";
#endif //HIFI_BUILD
        return executeStatus.isOk()
                ? convertErrorStatusToResultCode(executeStatus)
                : ANEURALNETWORKS_OP_FAILED;
    }

    // TODO: Remove this synchronization point when the block of code below is
    // removed.
    executionCallback->wait();
    Return<ErrorStatus> callbackStatus = executionCallback->getStatus();
    if (!callbackStatus.isOk() || callbackStatus != ErrorStatus::NONE) {
#ifndef HIFI_BUILD
        VLOG(EXECUTION) << "**Execute async failed**";
#endif //HIFI_BUILD
        return callbackStatus.isOk()
                ? convertErrorStatusToResultCode(callbackStatus)
                : ANEURALNETWORKS_OP_FAILED;
    }

    // Copy the output data from shared memory to the output buffers.
    // TODO: Move this block of code somewhere else. It should not be in the
    // startCompute function.
    // TODO: outputMemory->update(); outputMemory->commit()
    for (auto& info : mOutputs) {
        if (info.state == ModelArgumentInfo::POINTER) {
            DataLocation& loc = info.locationAndLength;
            uint8_t* data = nullptr;
            int n = outputPointerArguments.getPointer(&data);
            if (n != ANEURALNETWORKS_NO_ERROR) {
                return n;
            }
            memcpy(info.buffer, data + loc.offset, loc.length);
        }
    }
#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "StepExecutor::startComputeOnDevice completed";
#endif //HIFI_BUILD

    *synchronizationCallback = executionCallback;
    return ANEURALNETWORKS_NO_ERROR;
}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
static void asyncStartComputeOnCpu(const Model& model, const Request& request,
                                   const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                   const std::vector<RunTimePoolInfo>& requestPoolInfos,
                                   const sp<IExecutionCallback>& executionCallback) {
    CpuExecutor executor;
    int err = executor.run(model, request, modelPoolInfos, requestPoolInfos);
    executionCallback->notify(convertResultCodeToErrorStatus(err));
}
#else
static void asyncStartComputeOnCpu(const Model& model, const Request& request,
                                   const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                   const std::vector<RunTimePoolInfo>& requestPoolInfos,
                                   const sp<ExecutionCallback>& executionCallback) {
    CpuExecutor executor;
    int err = executor.run(model, request, modelPoolInfos, requestPoolInfos);
    executionCallback->notify(convertResultCodeToErrorStatus(err));
}
#endif //HIFI_BUILD

int StepExecutor::startComputeOnCpu(sp<ExecutionCallback>* synchronizationCallback) {
    // TODO: use a thread pool

    Model model;
    mModel->setHidlModel(&model);

    // Prepare the callback for asynchronous execution. sp<ExecutionCallback>
    // object is returned when the execution has been successfully launched,
    // otherwise a nullptr is returned. The executionCallback is abstracted in
    // the NN API as an "event".
    sp<ExecutionCallback> executionCallback = new ExecutionCallback();
#ifndef HIFI_BUILD
    *synchronizationCallback = nullptr;
#endif //HIFI_BUILD

    std::vector<RunTimePoolInfo> modelPoolInfos;
#ifndef HIFI_BUILD
    if (!setRunTimePoolInfosFromHidlMemories(&modelPoolInfos, model.pools)) {
        return ANEURALNETWORKS_UNMAPPABLE;
    }
#endif //HIFI_BUILD

    std::vector<RunTimePoolInfo> requestPoolInfos;
#ifndef HIFI_BUILD //not supporting input from memory as of now
    requestPoolInfos.reserve(mMemories.size());
    bool fail = false;
    for (const Memory* mem : mMemories) {
        requestPoolInfos.emplace_back(mem->getHidlMemory(), &fail);
    }
    if (fail) {
        return ANEURALNETWORKS_UNMAPPABLE;
    }
#endif //HIFI_BUILD
    // Create as many pools as there are input / output.
    auto fixPointerArguments = [&requestPoolInfos](std::vector<ModelArgumentInfo>& argumentInfos) {
        for (ModelArgumentInfo& argumentInfo : argumentInfos) {
            if (argumentInfo.state == ModelArgumentInfo::POINTER) {
                argumentInfo.locationAndLength.poolIndex =
                            static_cast<uint32_t>(requestPoolInfos.size());
                argumentInfo.locationAndLength.offset = 0;
                requestPoolInfos.emplace_back(static_cast<uint8_t*>(argumentInfo.buffer));
            }
        }
    };
    fixPointerArguments(mInputs);
    fixPointerArguments(mOutputs);

    Request request;
    setRequestArgumentArray(mInputs, &request.inputs);
    setRequestArgumentArray(mOutputs, &request.outputs);

#ifndef HIFI_BUILD
    // TODO: should model be moved with a std::cref?
    std::thread thread(asyncStartComputeOnCpu, model, std::move(request),
                       std::move(modelPoolInfos), std::move(requestPoolInfos),
                       executionCallback);
    executionCallback->bind_thread(std::move(thread));
    *synchronizationCallback = executionCallback;
#else
    asyncStartComputeOnCpu(model, request, modelPoolInfos, requestPoolInfos, executionCallback);
#endif //HIFI_BUILD

    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace nn
}  // namespace android
