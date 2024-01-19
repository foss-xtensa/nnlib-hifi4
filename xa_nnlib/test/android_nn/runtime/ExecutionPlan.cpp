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

#define LOG_TAG "ExecutionPlan"

#include "ExecutionPlan.h"

#include "Callbacks.h"
#include "CompilationBuilder.h"
#include "ExecutionBuilder.h"
#include "Manager.h"
#include "ModelBuilder.h"
#include "Utils.h"

#include <functional>
#include <map>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

using ::android::hardware::neuralnetworks::V1_0::implementation::ExecutionCallback;
#if 0
using ::android::hardware::neuralnetworks::V1_0::implementation::PreparedModelCallback;
#endif

namespace android {
namespace nn {

#ifndef HIFI_BUILD
static int compile(std::shared_ptr<Device> device, const ModelBuilder* model,
                   int32_t executionPreference, sp<IPreparedModel>* preparedModel) {
    nnAssert(device != nullptr);  // nullptr indicates CPU
    // Compilation logic copied from ExecutionBuilder::startComputeOnDevice().
    Model hidlModel;
    model->setHidlModel(&hidlModel);

    sp<PreparedModelCallback> preparedModelCallback = new PreparedModelCallback();
    Return<ErrorStatus> prepareLaunchStatus = device->getInterface()->prepareModel(
        hidlModel, static_cast<ExecutionPreference>(executionPreference), preparedModelCallback);
    if (!prepareLaunchStatus.isOk()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ExecutionStep::finishSubModel compilation failed due to transport error: "
                   << prepareLaunchStatus.description();
#endif //HIFI_BUILD
        return ANEURALNETWORKS_OP_FAILED;
    }
    if (prepareLaunchStatus != ErrorStatus::NONE) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ExecutionStep::finishSubModel compilation failed with error: "
                   << toString(static_cast<ErrorStatus>(prepareLaunchStatus));
#endif //HIFI_BUILD
        return ANEURALNETWORKS_OP_FAILED;
    }

    preparedModelCallback->wait();
    ErrorStatus prepareReturnStatus = preparedModelCallback->getStatus();
    *preparedModel = preparedModelCallback->getPreparedModel();
    if (prepareReturnStatus != ErrorStatus::NONE || *preparedModel == nullptr) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ExecutionPlan compilation on " << device->getName() << " failed:"
                   << " prepareReturnStatus=" << toString(prepareReturnStatus)
                   << ", preparedModel=" << preparedModel->get();
#endif //HIFI_BUILD
        return ANEURALNETWORKS_OP_FAILED;
    }
    return ANEURALNETWORKS_NO_ERROR;
}
#else
int compile(std::shared_ptr<Device> device, const ModelBuilder* model, int32_t executionPreference,
            void* preparedModel) {
    nnAssert(device != nullptr);
    Model hidlModel;
    model->setHidlModel(&hidlModel);
    return device->prepareModel(hidlModel, static_cast<ExecutionPreference>(executionPreference),
                                preparedModel);
}
#endif //HIFI_BUILD

typedef std::function<void(uint32_t)> OperationReadyCallback;

// This class tracks whether we know the value of an operand as operations
// are processed.
#ifndef HIFI_BUILD
class OperandTracker {
public:
    // Creates the tracker for this model. Figure out which operations can be
    // executed right away and cb for each one of them.
    OperandTracker(const ModelBuilder* model, OperationReadyCallback cb);
    // Mark the specified operation as having been processed. The output
    // of the operation now being known, this may make new operations to be
    // able to run.  Call cb for each one of them.
    void markProcessed(uint32_t operationIndex, OperationReadyCallback cb);

private:
    const ModelBuilder* mModel;
    std::multimap<uint32_t, uint32_t> mOperandToOperations;
    std::vector<uint32_t> mUnknownInputCount;  // For each operation
};

OperandTracker::OperandTracker(const ModelBuilder* model, OperationReadyCallback cb) :
        mModel(model) {
    const auto& operations = mModel->getOperations();
    mUnknownInputCount.resize(operations.size());
    for (uint32_t operationIndex = 0; operationIndex < operations.size(); operationIndex++) {
        const Operation& operation = operations[operationIndex];
        uint32_t count = 0;
        for (uint32_t operandIndex : operation.inputs) {
            auto lifetime = mModel->getOperand(operandIndex).lifetime;
            if (lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
                lifetime == OperandLifeTime::MODEL_OUTPUT) {
                count++;
                mOperandToOperations.insert(
                        std::pair<uint32_t, uint32_t>(operandIndex, operationIndex));
            }
        }
        if (count == 0) {
            cb(operationIndex);
        }
        mUnknownInputCount[operationIndex] = count;
    }
}

void OperandTracker::markProcessed(uint32_t operationIndex, OperationReadyCallback cb) {
    // Mark all its outputs as known.
    const Operation& operation = mModel->getOperations()[operationIndex];
    for (uint32_t operandIndex : operation.outputs) {
        auto range = mOperandToOperations.equal_range(operandIndex);
        for (auto i = range.first; i != range.second; i++) {
            uint32_t& count = mUnknownInputCount[i->second];
            if (--count == 0) {
                cb(i->second);
            }
        }
    }
}
#endif //HIFI_BUILD

ExecutionStep::ExecutionStep(ExecutionPlan* plan, uint32_t stepIndex,
                             std::shared_ptr<Device> device)
        : mPlan(plan), mIndex(stepIndex), mSubModel(), mDevice(device) {}

// Adds an operand if it has not been added already.
// Sets the index in the submodel for the corresponding operand.
int ExecutionStep::addOperand(uint32_t fromOperandIndex, uint32_t* toOperandIndex,
                              const ModelBuilder& fromModel, OperandKind kind) {
    // Have we added this operand already?
    auto i = mOperandMap.find(fromOperandIndex);
    if (i != mOperandMap.end()) {
        nnAssert(kind == INPUT);
        *toOperandIndex = i->second;
        return ANEURALNETWORKS_NO_ERROR;
    }

    // First time we add this operand.
    *toOperandIndex = mSubModel.operandCount();
    mOperandMap.insert(std::pair<uint32_t, uint32_t>(fromOperandIndex, *toOperandIndex));

    // Add the operand to the submodel.
    const Operand& operand = fromModel.getOperand(fromOperandIndex);
    ANeuralNetworksOperandType type = {
        .type = static_cast<int32_t>(operand.type),
        .dimensionCount = static_cast<uint32_t>(operand.dimensions.size()),
        .dimensions = operand.dimensions.size() > 0 ? operand.dimensions.data() : nullptr,
        .scale = operand.scale,
        .zeroPoint = operand.zeroPoint
    };
    int n = mSubModel.addOperand(type);
    if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Previous error occurred when partitioning the graph";
#endif //HIFI_BUILD
        return n;
    }

    // Sets its value.
    switch (operand.lifetime) {
        case OperandLifeTime::CONSTANT_COPY: {
            const uint8_t* data = fromModel.getPointerToOperandValue(operand.location.offset);
            n = mSubModel.setOperandValue(*toOperandIndex, data, operand.location.length);
            if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Previous error occurred when partitioning the graph";
#endif //HIFI_BUILD
                return n;
            }
        } break;
        case OperandLifeTime::CONSTANT_REFERENCE: {
            const Memory* memory = fromModel.getMemories()[operand.location.poolIndex];
            n = mSubModel.setOperandValueFromMemory(*toOperandIndex, memory,
                                                     operand.location.offset,
                                                     operand.location.length);
            if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Previous error occurred when partitioning the graph";
#endif //HIFI_BUILD
                return n;
            }
        } break;
        case OperandLifeTime::NO_VALUE: {
            n = mSubModel.setOperandValue(*toOperandIndex, nullptr, 0);
            if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Previous error occurred when partitioning the graph";
#endif //HIFI_BUILD
                return n;
            }
        } break;
        case OperandLifeTime::TEMPORARY_VARIABLE:  // handled similarly to MODEL_OUTPUT
            if (kind == INPUT) {
                // The first time we've seen this operand is as an
                // input.  That means it must be defined by a
                // different partition, and is an input to this one.
                mTempsAsSubModelInputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            } else {
                // The first time we've seen this operand is as an
                // output.  It may be an input to a different
                // partition, so keep track of it.
                mPlan->recordTemporaryDef(fromOperandIndex, mIndex);
            }
            break;
        case OperandLifeTime::MODEL_INPUT:
            mModelInputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            break;
        case OperandLifeTime::MODEL_OUTPUT:  // handled similarly to TEMPORARY_VARIABLE
            if (kind == INPUT) {
                // The first time we've seen this operand is as an
                // input.  That means it must be defined by a
                // different partition, and is an input to this one.
                mOutputsAsSubModelInputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            } else {
                // The first time we've seen this operand is as an
                // output.
                mModelOutputs.push_back(std::make_pair(fromOperandIndex, *toOperandIndex));
            }
            break;
        default:
            nnAssert(false);
            break;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ExecutionStep::addOperation(int operationIndex, const ModelBuilder& fromModel) {
    const Operation& operation = fromModel.getOperation(operationIndex);

    // Convert the input and output operand indexes.
    //
    // We expect operations to be added in topological order.  Therefore:
    //
    // - We may not have seen an input if it is a model input, a
    //   constant, or an operand written by a different partition.
    //
    // - We should not have seen any outputs.
    const uint32_t inputCount = static_cast<uint32_t>(operation.inputs.size());
    const uint32_t outputCount = static_cast<uint32_t>(operation.outputs.size());
    std::vector<uint32_t> inputs(inputCount);
    std::vector<uint32_t> outputs(outputCount);

    auto addOperands = [this, &fromModel](const hidl_vec<uint32_t>& globalOperands,
                                          std::vector<uint32_t>& localOperands,
                                          OperandKind kind) -> int {
        const uint32_t operandCount = static_cast<uint32_t>(globalOperands.size());
        for (uint32_t i = 0; i < operandCount; i++) {
            uint32_t localOperand = ~0U;
            int n = addOperand(globalOperands[i], &localOperand, fromModel, kind);
            if (n != ANEURALNETWORKS_NO_ERROR)
                return n;
            localOperands[i] = localOperand;
        }
        return ANEURALNETWORKS_NO_ERROR;
    };

    int n;
    if ((n = addOperands(operation.inputs, inputs, INPUT)) != ANEURALNETWORKS_NO_ERROR ||
        (n = addOperands(operation.outputs, outputs, OUTPUT)) != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }

    return mSubModel.addOperation(static_cast<uint32_t>(operation.type), inputCount, inputs.data(),
                                   outputCount, outputs.data());
}

void ExecutionStep::mapInputsAndOutputs(std::shared_ptr<StepExecutor> stepExecutor) const {
    for (uint32_t i = 0, e = mInputIndexSubModelToFromModel.size(); i < e; i++) {
        stepExecutor->mapInput(mInputIndexSubModelToFromModel[i], i);
    }
    for (uint32_t i = 0, e = mOutputIndexSubModelToFromModel.size(); i < e; i++) {
        stepExecutor->mapOutput(mOutputIndexSubModelToFromModel[i], i);
    }
}

void ExecutionPlan::CompoundBody::findTempsAsSubModelOutputs() {
    for (const auto& step : mSteps) {
        for (const auto& input : step->getTempsAsSubModelInputs()) {
            const uint32_t fromModelIndex = input.first;
            const auto it = mTemporaryToDefiningStep.find(fromModelIndex);
            nnAssert(it != mTemporaryToDefiningStep.end());
            const uint32_t stepIndex = it->second;
            nnAssert(stepIndex < mSteps.size());
            mSteps[stepIndex]->recordTempAsSubModelOutput(fromModelIndex);
        }
    }
}

#ifndef HIFI_BUILD
void ExecutionStep::logSubModel() const {
#ifndef HIFI_BUILD
    VLOG(COMPILATION) << "ExecutionStep::finishSubModel, step " << mIndex;
#endif //HIFI_BUILD

    auto logRemapEntry = [](std::string &toLog, const std::pair<uint32_t, uint32_t>& e) {
        if (!toLog.empty()) {
            toLog += ", ";
        }
        toLog += "(";
        toLog += std::to_string(e.first);
        toLog += "->";
        toLog += std::to_string(e.second);
        toLog += ")";
    };

    auto logRemapVector = [&logRemapEntry](const char* name, const RemapVectorType& map) {
        std::string toLog;
        for (const auto& e : map) {
            logRemapEntry(toLog, e);
        }
#ifndef HIFI_BUILD
        VLOG(COMPILATION) << name << ": " << toLog;
#endif //HIFI_BUILD
    };
    auto logRemapSet = [&logRemapEntry](const char* name, const SubModelOutputSetType& set) {
        std::string toLog;
        for (const auto& e : set) {
            logRemapEntry(toLog, e);
        }
#ifndef HIFI_BUILD
        VLOG(COMPILATION) << name << ": " << toLog;
#endif //HIFI_BUILD
    };

    logRemapVector("model inputs", mModelInputs);
    logRemapVector("model outputs", mModelOutputs);
    logRemapVector("temps as submodel inputs", mTempsAsSubModelInputs);
    logRemapSet("temps as submodel outputs", mTempsAsSubModelOutputs);
    logRemapVector("outputs as submodel inputs", mOutputsAsSubModelInputs);
}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
static void convertModelInputsOrOutputs(
        // IN: mModel{Inputs|Outputs}
        const ExecutionStep::RemapVectorType& myModelInputsOrOutputs,
        // IN: fromModel->{input|output}Count()
        uint32_t                              fromModelInputOrOutputCount,
        // IN: fromModel->get{Input|Output}OperandIndex
        std::function<uint32_t(uint32_t)>     fromModelGetInputOrOutputOperandIndex,
        // OUT: for v : mModel{Inputs|Outputs} : v.second
        std::vector<uint32_t>*                inputsOrOutputs,
        // OUT: submodel input-or-output index to original model input-or-output index
        std::vector<uint32_t>*                inputOrOutputIndexSubModelToFromModel) {
    std::map<uint32_t, uint32_t> fromModelIndexMap;  // operand index to input-or-output index
    for (uint32_t i = 0; i < fromModelInputOrOutputCount; i++) {
        fromModelIndexMap[fromModelGetInputOrOutputOperandIndex(i)] = i;
    }
    for (const auto& myInputOrOutput : myModelInputsOrOutputs) {
        inputsOrOutputs->push_back(myInputOrOutput.second);
        const uint32_t fromModelInputOrOutputIndex = fromModelIndexMap[myInputOrOutput.first];
        inputOrOutputIndexSubModelToFromModel->push_back(fromModelInputOrOutputIndex);
    }
}
#else
static void convertModelInputs(
        // IN: mModel{Inputs|Outputs}
        const ExecutionStep::RemapVectorType& myModelInputsOrOutputs,
        // IN: fromModel->{input|output}Count()
        uint32_t                              fromModelInputOrOutputCount,
        // IN: fromModel->get{Input|Output}OperandIndex
        const ModelBuilder *fromModel,
        // OUT: for v : mModel{Inputs|Outputs} : v.second
        std::vector<uint32_t>*                inputsOrOutputs,
        // OUT: submodel input-or-output index to original model input-or-output index
        std::vector<uint32_t>*                inputOrOutputIndexSubModelToFromModel) {
    std::map<uint32_t, uint32_t> fromModelIndexMap;  // operand index to input-or-output index
    for (uint32_t i = 0; i < fromModelInputOrOutputCount; i++) {
        fromModelIndexMap[fromModel->getInputOperandIndex(i)] = i;
    }
    for (const auto& myInputOrOutput : myModelInputsOrOutputs) {
        inputsOrOutputs->push_back(myInputOrOutput.second);
        const uint32_t fromModelInputOrOutputIndex = fromModelIndexMap[myInputOrOutput.first];
        inputOrOutputIndexSubModelToFromModel->push_back(fromModelInputOrOutputIndex);
    }
}

static void convertModelOutputs(
        // IN: mModel{Inputs|Outputs}
        const ExecutionStep::RemapVectorType& myModelInputsOrOutputs,
        // IN: fromModel->{input|output}Count()
        uint32_t                              fromModelInputOrOutputCount,
        // IN: fromModel->get{Input|Output}OperandIndex
        const ModelBuilder *fromModel,
        // OUT: for v : mModel{Inputs|Outputs} : v.second
        std::vector<uint32_t>*                inputsOrOutputs,
        // OUT: submodel input-or-output index to original model input-or-output index
        std::vector<uint32_t>*                inputOrOutputIndexSubModelToFromModel) {
    std::map<uint32_t, uint32_t> fromModelIndexMap;  // operand index to input-or-output index
    for (uint32_t i = 0; i < fromModelInputOrOutputCount; i++) {
        fromModelIndexMap[fromModel->getOutputOperandIndex(i)] = i;
    }
    for (const auto& myInputOrOutput : myModelInputsOrOutputs) {
        inputsOrOutputs->push_back(myInputOrOutput.second);
        const uint32_t fromModelInputOrOutputIndex = fromModelIndexMap[myInputOrOutput.first];
        inputOrOutputIndexSubModelToFromModel->push_back(fromModelInputOrOutputIndex);
    }
}
#endif //HIFI_BUILD

int ExecutionStep::finishSubModel(const ModelBuilder* fromModel, bool* hasOutputOfUnknownSize,
                                  int32_t executionPreference) {
#ifndef HIFI_BUILD
    if (VLOG_IS_ON(COMPILATION)) {
        logSubModel();
    }
#endif //HIFI_BUILD

    mSubModel.relaxComputationFloat32toFloat16(fromModel->isComputationFloat32RelaxedToFloat16());

    // Input order: mModelInputs, mTempsAsSubModelInputs, mOutputsAsSubModelInputs
    // Output order: mModelOutputs, mTempsAsSubModelOutputs
    //
    // ExecutionPlan::next() depends on these orderings.

    std::vector<uint32_t> inputs;
#ifndef HIFI_BUILD
    convertModelInputsOrOutputs(mModelInputs,
                                fromModel->inputCount(),
                                [=](uint32_t i) { return fromModel->getInputOperandIndex(i); },
                                &inputs,
                                &mInputIndexSubModelToFromModel);
#else
    convertModelInputs(mModelInputs,
                                fromModel->inputCount(),
                                fromModel,
                                &inputs,
                                &mInputIndexSubModelToFromModel);
#endif //HIFI_BUILD

    for (const auto& subModelInput : mTempsAsSubModelInputs) {
        inputs.push_back(subModelInput.second);
    }
    for (const auto& subModelInput : mOutputsAsSubModelInputs) {
        inputs.push_back(subModelInput.second);
    }

    std::vector<uint32_t> outputs;
#ifndef HIFI_BUILD
    convertModelInputsOrOutputs(mModelOutputs,
                                fromModel->outputCount(),
                                [=](uint32_t i) { return fromModel->getOutputOperandIndex(i); },
                                &outputs,
                                &mOutputIndexSubModelToFromModel);
#else
    convertModelOutputs(mModelOutputs,
                                fromModel->outputCount(),
                                fromModel,
                                &outputs,
                                &mOutputIndexSubModelToFromModel);
#endif //HIFI_BUILD
    for (const auto& subModelOutput : mTempsAsSubModelOutputs) {
        outputs.push_back(subModelOutput.second);
        const Operand& operand = mSubModel.getOperand(subModelOutput.second);
        for (uint32_t dimension : operand.dimensions) {
            if (dimension == 0) {
                *hasOutputOfUnknownSize = true;
#ifndef HIFI_BUILD
                VLOG(COMPILATION) << "SubModelOutput (operand#" << subModelOutput.first
                                << " of original graph) has unknown size: "
                                << toString(operand);
#endif //HIFI_BUILD
                break;
            }
        }
    }

    {
        int n = mSubModel.identifyInputsAndOutputs(inputs.size(), &inputs[0], outputs.size(), &outputs[0]);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
        n = mSubModel.finish();
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }

    {
        // Compute mOutputsAsSubModelInputsIndexToFromModel.

        std::map<uint32_t, uint32_t> fromModelOperandIndexToOutputIndex;
        for (unsigned i = 0, e = fromModel->outputCount(); i < e; ++i) {
            fromModelOperandIndexToOutputIndex[fromModel->getOutputOperandIndex(i)] = i;
        }

        for (unsigned i = 0, e = mOutputsAsSubModelInputs.size(); i < e; i++) {
            const uint32_t fromModelOperandIndex = mOutputsAsSubModelInputs[i].first;
            const auto it = fromModelOperandIndexToOutputIndex.find(fromModelOperandIndex);
            if (it == fromModelOperandIndexToOutputIndex.end()) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "Could not find main model output operand " << fromModelOperandIndex
                           << " in main model output operand list";
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_STATE;
            }
            mOutputsAsSubModelInputsIndexToFromModel.push_back(it->second);
        }
    }

    // TODO: Move compilation elsewhere?

    if (mDevice == nullptr) {
        return ANEURALNETWORKS_NO_ERROR;
    }

#ifndef HIFI_BUILD
    VLOG(COMPILATION) << "ExecutionStep::finishSubModel, compilation";
#endif //HIFI_BUILD
#ifndef HIFI_BUILD
    return compile(mDevice, &mSubModel, executionPreference, &mPreparedSubModel);
#else
    return compile(mDevice, &mSubModel, executionPreference, nullptr);
#endif //HIFI_BUILD
}

void ExecutionStep::dump() const {
    Model model;
    mSubModel.setHidlModel(&model);
#ifndef HIFI_BUILD
    if (VLOG_IS_ON(COMPILATION)) {
        VLOG(COMPILATION) << "ExecutionStep#" << mIndex
                          << " for " << (mDevice == nullptr ? "CPU" : mDevice->getName());
        logModelToInfo(model);
    }
#endif //HIFI_BUILD
}

int ExecutionPlan::CompoundBody::finish(const ModelBuilder* fromModel,
                                        int32_t executionPreference) {
    findTempsAsSubModelOutputs();
    for (const auto& step : mSteps) {
        int n = step->finishSubModel(fromModel, &mHasSubModelOutputOfUnknownSize,
                                     executionPreference);
        if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
            VLOG(COMPILATION) << "ExecutionPlan::CompoundBody::finish -- finishSubModel failed";
#endif //HIFI_BUILD
            return n;
        }
    }
    if (mHasSubModelOutputOfUnknownSize) {
#ifndef HIFI_BUILD
        VLOG(COMPILATION) << "ExecutionPlan::CompoundBody::finish -- mHasSubModelOutputOfUnknownSize";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_OP_FAILED;
    }

    mSuccessfulFinish = true;
    return ANEURALNETWORKS_NO_ERROR;
}

#ifndef HIFI_BUILD
int ExecutionPlan::SimpleBody::finish([[maybe_unused]] const ModelBuilder* fromModel,
#else
int ExecutionPlan::SimpleBody::finish(const ModelBuilder* fromModel,
#endif // HIFI_BUILD
                                      int32_t executionPreference) {
    if (mDevice == nullptr) {
        mSuccessfulFinish = true;
        return ANEURALNETWORKS_NO_ERROR;
    }

#ifndef HIFI_BUILD
    VLOG(COMPILATION) << "ExecutionPlan::SimpleBody::finish, compilation";
#endif //HIFI_BUILD
#ifndef HIFI_BUILD
    const int n = compile(mDevice, mModel, executionPreference, &mPreparedModel);
#else
    const int n = compile(mDevice, mModel, executionPreference, nullptr);
#endif //HIFI_BUILD
    mSuccessfulFinish = (n == ANEURALNETWORKS_NO_ERROR);
    return n;
}

int ExecutionPlan::finish(const ModelBuilder* fromModel, int32_t executionPreference) {
    nnAssert(mBody != nullptr);
    return mBody->finish(fromModel, executionPreference);
}

ExecutionPlan::Controller::Controller(
    const ExecutionPlan* plan,
    const ExecutionBuilder* executionBuilder,
    std::shared_ptr<const SubModelInputsAndOutputsType> subModelInputsAndOutputs,
    uint32_t totalSizeOfTemporaries) :
        mPlan(plan), mExecutionBuilder(executionBuilder),
        mSubModelInputsAndOutputs(subModelInputsAndOutputs), mNextStepIndex(0) {
#ifndef HIFI_BUILD
    if (totalSizeOfTemporaries) {
        if (mTemporaries.create(totalSizeOfTemporaries) != ANEURALNETWORKS_NO_ERROR) {
            LOG(ERROR) << "ExecutionPlan::Controller failed to allocate temporaries";
            mNextStepIndex = kBadStepIndex;
        }
    }
#endif //HIFI_BUILD
}

std::shared_ptr<ExecutionPlan::Controller> ExecutionPlan::makeController(
    const ExecutionBuilder* executionBuilder) const {
    nnAssert((mState == EMPTY) == (mBody == nullptr));
    if (mBody && !mBody->mSuccessfulFinish) {
#ifndef HIFI_BUILD
        VLOG(EXECUTION) << "ExecutionPlan::makeController -- unsuccessful finish";
#endif //HIFI_BUILD
        return std::shared_ptr<Controller>(nullptr);
    }

    // Create the layout for a Memory object big enough for to hold
    // every TEMPORARY in the original model that is live across
    // partition boundaries.
    //
    // TODO: Rethink this approach for managing temporaries.  Some
    // alternatives:
    //
    // 1) Adopt a memory layout scheme analogous to stack allocation,
    // where objects of non-overlapping lifetime can occupy the same
    // storage.  We would still have a single Memory object in this
    // case.
    //
    // 2) Do something like what CpuExecutor does, and do allocations
    // and deallocations on the fly (during execution) before first
    // reference and after last reference, respectively.  This would
    // mean having one Memory object per TEMPORARY; or, in a more
    // complicated implementation, one Memory object per set of
    // temporaries that have the same lifetime.  Note that the Android
    // system limits the number of shared memory objects, which are
    // what our Memory objects represent.
    //
    uint32_t totalSizeOfTemporaries = 0;
    std::shared_ptr<Controller::SubModelInputsAndOutputsType> subModelInputsAndOutputs;
    if (mState == COMPOUND) {
        const ModelBuilder* fromModel = executionBuilder->getModel();
        for (const auto& step : compound()->mSteps) {
            for (const auto& output: step->getTempsAsSubModelOutputs()) {
                const uint32_t fromModelOperandIndex = output.first;
                const Operand& fromModelOperand = fromModel->getOperand(fromModelOperandIndex);
                if (subModelInputsAndOutputs == nullptr) {
                    subModelInputsAndOutputs =
                            std::make_shared<Controller::SubModelInputsAndOutputsType>();
                }
                const uint32_t size = sizeOfData(fromModelOperand);
                totalSizeOfTemporaries += alignBytesNeeded(totalSizeOfTemporaries, size);
                subModelInputsAndOutputs->insert(std::make_pair(fromModelOperandIndex, totalSizeOfTemporaries));
                totalSizeOfTemporaries += size;
            }
        }
#ifndef HIFI_BUILD
        if (VLOG_IS_ON(EXECUTION) && (subModelInputsAndOutputs != nullptr)) {
            for (const auto& io : *subModelInputsAndOutputs) {
                VLOG(EXECUTION) << "temp: origOpndIdx = " << io.first
                                << ", offset = " << io.second;
            }
        }
#endif //HIFI_BUILD
    }

    return std::shared_ptr<Controller>(new Controller(this, executionBuilder,
                                                      subModelInputsAndOutputs,
                                                      totalSizeOfTemporaries));
}


// TODO: Find a better way to provide this functionality.
int ExecutionPlan::fallback(std::shared_ptr<Controller> controller,
                            std::shared_ptr<StepExecutor>* executor) const {
    *executor = nullptr;

#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "ExecutionPlan::fallback(" << controller << ", " << executor
                    << "): mNextStepIndex = " << controller->mNextStepIndex;
#endif //HIFI_BUILD

    if (controller->mNextStepIndex == 0) {
        // We haven't called next().
        return ANEURALNETWORKS_OP_FAILED;
    }

    if (controller->mNextStepIndex == Controller::kBadStepIndex) {
        // The last call to next() did not produce an executor.
        return ANEURALNETWORKS_OP_FAILED;
    }

    --controller->mNextStepIndex;
    return next(controller, executor);
}

int ExecutionPlan::next(std::shared_ptr<Controller> controller,
                        std::shared_ptr<StepExecutor>* executor) const {
    *executor = nullptr;

#ifndef HIFI_BUILD
    VLOG(EXECUTION) << "ExecutionPlan::next("
                    << SHOW_IF_DEBUG(controller << ", " << executor)
                    << "): mNextStepIndex = " << controller->mNextStepIndex;
#endif //HIFI_BUILD

    if (controller->mNextStepIndex == Controller::kBadStepIndex) {
        return ANEURALNETWORKS_OP_FAILED;
    }

    if (mState == EMPTY) {
        nnAssert(controller->mNextStepIndex == 0);  // end
        controller->mNextStepIndex = Controller::kBadStepIndex;
        return ANEURALNETWORKS_NO_ERROR;
    }

    if (mState == SIMPLE) {
        if (controller->mNextStepIndex == 0) {
            // First (and only) step.
            auto simpleBody = static_cast<const SimpleBody*>(mBody);
            *executor = std::make_shared<StepExecutor>(
                controller->mExecutionBuilder,
                simpleBody->mModel,
#ifndef HIFI_BUILD
                (simpleBody->mDevice == nullptr ? nullptr : simpleBody->mDevice->getInterface()),
                simpleBody->mPreparedModel);
#else
                nullptr,
                nullptr);
#endif //HIFI_BUILD
            (*executor)->mapInputsAndOutputsTrivially();
            controller->mNextStepIndex = 1;
            return ANEURALNETWORKS_NO_ERROR;
        }

        nnAssert(controller->mNextStepIndex == 1);  // end
        controller->mNextStepIndex = Controller::kBadStepIndex;
        return ANEURALNETWORKS_NO_ERROR;
    }

#ifndef HIFI_BUILD
    auto compoundBody = compound();

    if (controller->mNextStepIndex == compoundBody->mSteps.size()) {
        // end
        controller->mNextStepIndex = Controller::kBadStepIndex;
        return ANEURALNETWORKS_NO_ERROR;
    }

    // Input order: model inputs, temps as submodel inputs, outputs as submodel inputs
    // Output order: model outputs, temps as submodel outputs
    //
    // ExecutionStep::finishSubModel() establishes these orderings.

    const auto step = compoundBody->mSteps[controller->mNextStepIndex];
    *executor = std::make_shared<StepExecutor>(
        controller->mExecutionBuilder,
        step->getSubModel(),
        (step->getDevice() == nullptr ? nullptr : step->getDevice()->getInterface()),
        step->getPreparedSubModel());
    step->mapInputsAndOutputs(*executor);
    if (controller->mSubModelInputsAndOutputs != nullptr) {
        {
            // Tell executor about temps as submodel outputs.

            const size_t firstSubModelOutputIndex = step->getModelOutputs().size();
            const auto& subModelOutputs = step->getTempsAsSubModelOutputs();

            uint32_t idx = 0;
            for (auto I = subModelOutputs.begin(), E = subModelOutputs.end(); I != E; I++, idx++) {
                const uint32_t fromModelOperandIndex = I->first;
                const uint32_t offsetOfTemporary =
                    controller->mSubModelInputsAndOutputs->at(fromModelOperandIndex);
                int n = (*executor)->setOutputFromTemporaryMemory(
                    firstSubModelOutputIndex + idx,
                    &controller->mTemporaries,
                    offsetOfTemporary);
                if (n != ANEURALNETWORKS_NO_ERROR) {
                    controller->mNextStepIndex = Controller::kBadStepIndex;
                    return n;
                }
            }
        }
        {
            // Tell executor about temps as submodel inputs.

            const size_t firstSubModelInputIndex = step->getModelInputs().size();
            const auto& subModelInputs = step->getTempsAsSubModelInputs();

            uint32_t idx = 0;
            for (auto I = subModelInputs.begin(), E = subModelInputs.end(); I != E; I++, idx++) {
                const uint32_t fromModelOperandIndex = I->first;
                const uint32_t offsetOfTemporary =
                    controller->mSubModelInputsAndOutputs->at(fromModelOperandIndex);
                int n = (*executor)->setInputFromTemporaryMemory(
                    firstSubModelInputIndex + idx,
                    &controller->mTemporaries,
                    offsetOfTemporary);
                if (n != ANEURALNETWORKS_NO_ERROR) {
                    controller->mNextStepIndex = Controller::kBadStepIndex;
                    return n;
                }
            }
        }
    }
    {
        // Tell executor about outputs as submodel inputs.

        const size_t firstOutputsAsSubModelInputIndex =
                step->getModelInputs().size() + step->getTempsAsSubModelInputs().size();
        const auto& outputsAsSubModelInputsIndexToFromModel =
                step->getOutputsAsSubModelInputsIndexToFromModel();
        for (uint32_t i = 0, e = outputsAsSubModelInputsIndexToFromModel.size(); i < e; i++) {
            uint32_t o = outputsAsSubModelInputsIndexToFromModel[i];
            (*executor)->mapOutputToInput(o, firstOutputsAsSubModelInputIndex + i);
        }
    }

    controller->mNextStepIndex++;
#endif //HIFI_BUILD
    return ANEURALNETWORKS_NO_ERROR;
}

std::shared_ptr<ExecutionStep> ExecutionPlan::createNewStep(const std::shared_ptr<Device> device) {
    nnAssert(mState != SIMPLE);
    if (mState == EMPTY) {
        mBody = new CompoundBody();
        mState = COMPOUND;
    }
    auto& steps = compound()->mSteps;
    auto step = std::make_shared<ExecutionStep>(this, steps.size(), device);
    steps.push_back(step);
    return step;
}

void ExecutionPlan::becomeSingleStep(const std::shared_ptr<Device> device,
                                     const ModelBuilder* model) {
    nnAssert(mState == EMPTY);
    mBody = new SimpleBody(device, model);
    mState = SIMPLE;
}

void ExecutionPlan::dump() const {
    if (mBody) {
        mBody->dump();
    } else {
#ifndef HIFI_BUILD
        VLOG(COMPILATION) << "EMPTY";
#endif //HIFI_BUILD
    }
}

#ifndef HIFI_BUILD
#else
void ExecutionPlan::reset() {
    if (mBody) {
        delete mBody;
        mBody = nullptr;
    }
    mState = EMPTY;
}
#endif //HIFI_BUILD

ExecutionPlan::Kind ExecutionPlan::forTest_getKind() const {
    switch (mState) {
        case EMPTY:
            return Kind::EMPTY;
        case SIMPLE:
            nnAssert(mBody);
            return mBody->mSuccessfulFinish ? Kind::SIMPLE : Kind::ERROR;
        case COMPOUND:
            nnAssert(mBody);
            return mBody->mSuccessfulFinish ? Kind::COMPOUND : Kind::ERROR;
        default:
            nnAssert(!"unexpected state");
            return Kind::ERROR;
    }
}

std::shared_ptr<const Device> ExecutionPlan::forTest_simpleGetDevice() const {
    nnAssert(mState == SIMPLE);
    return static_cast<const SimpleBody*>(mBody)->mDevice;
}

const std::vector<std::shared_ptr<ExecutionStep>>& ExecutionPlan::forTest_compoundGetSteps() const {
    return compound()->mSteps;
}

bool ExecutionPlan::forTest_hasSubModelOutputsOfUnknownSize() const {
    return mBody->hasSubModelOutputsOfUnknownSize();
}

void ExecutionPlan::SimpleBody::dump() const {
#ifndef HIFI_BUILD
    VLOG(COMPILATION) << "SIMPLE for " << (mDevice == nullptr ? "CPU" : mDevice->getName());
#endif //HIFI_BUILD
}

void ExecutionPlan::CompoundBody::dump() const {
    for (const auto& step : mSteps) {
        step->dump();
    }
}

#ifndef HIFI_BUILD
int ModelBuilder::partitionTheWork(const std::vector<std::shared_ptr<Device>>& devices,
                                   uint32_t preference, ExecutionPlan* plan) const {
    // This function uses a heuristic approach to partitioning the graph.
    // It should be good enough for the first release.

    const size_t nonCpuDeviceCount = devices.size();
    // The device count is the number of HAL devices + 1. The +1 is for the CPU.
    // Note that deviceCount includes CPU, which has no entry in devices[].
    const size_t deviceCount = nonCpuDeviceCount + 1;
    const size_t operationCount = mOperations.size();

    VLOG(COMPILATION) << "ModelBuilder::partitionTheWork: deviceCount = " << deviceCount
                      << ", operationCount = " << operationCount;

    // If we only have the CPU, or if the graph has no operations, no need to try to partition.
    if (nonCpuDeviceCount == 0 || operationCount == 0) {
        // Make sure no op is an OEM operation.
        if (mHasOEMOperation) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "No driver can do the OEM op";
#endif //HIFI_BUILD
            return ANEURALNETWORKS_BAD_DATA;
        }
        plan->becomeSingleStep(nullptr /* CPU */, this);
        return plan->finish(this, preference);
    }

    // Figure out where each operation will best execute.
    // The value of the vector is the index in the devices vector, with devices.size()
    // representing the CPU.
    std::vector<int> bestDeviceForOperation(operationCount);
    int status = findBestDeviceForEachOperation(preference, devices, deviceCount,
                                                &bestDeviceForOperation);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        return status;
    }

    // If one device will run all the operations, we don't need to split the work.
    if (std::adjacent_find(bestDeviceForOperation.begin(), bestDeviceForOperation.end(),
                           std::not_equal_to<int>()) == bestDeviceForOperation.end()) {
        const int bestDeviceIndex = bestDeviceForOperation[0];
        const bool cpu = (size_t(bestDeviceIndex) == deviceCount - 1);
        VLOG(COMPILATION) << "ModelBuilder::partitionTheWork: only one best device: "
                          << bestDeviceIndex << " = "
                          << (cpu ? "CPU" : devices[bestDeviceIndex]->getName());
        plan->becomeSingleStep(cpu ? nullptr : devices[bestDeviceIndex], this);
        return plan->finish(this, preference);
    }

    // No easy solution, we need to split the work.

    // We keep track of the operations that are ready to run for each device.
    std::vector<std::queue<uint32_t>> perDeviceQueue(deviceCount);

    // This helper function enqueues the operation on the appropriate queue.
    auto enqueueOnAppropriateDevice = [&](uint32_t operationIndex) {
        int deviceIndex = bestDeviceForOperation[operationIndex];
        perDeviceQueue[deviceIndex].push(operationIndex);
        VLOG(COMPILATION) << "enqueueOnAppropriateDevice " << operationIndex << " onto "
                          << deviceIndex;
    };

    // This helper function finds a device that has operations ready to process.
    // We start by looking at the CPU. We do this to try to maximize the
    // size of the graph we'll send to non-CPU devices. If the CPU runs first,
    // it will have the chance to prepare more of the inputs required by the
    // other devices. This function returns -1 if all queues are empty.
    auto findNextDeviceToProcess = [&]() -> int {
        for (int i = deviceCount - 1; i >= 0; i--) {
            if (!perDeviceQueue[i].empty()) {
                return i;
            }
        }
        return -1;
    };

    OperandTracker tracker(this, enqueueOnAppropriateDevice);
    // For each iteration of this loop, we'll create an execution step.
    while (true) {
        // Find the device we'll do this step for.
        int deviceIndex = findNextDeviceToProcess();
        VLOG(COMPILATION) << "findNextDeviceToProcess: " << deviceIndex;
        if (deviceIndex < 0) {
            break;
        }
        // nullptr represents the CPU.
        std::shared_ptr<Device> device =
                static_cast<size_t>(deviceIndex) < nonCpuDeviceCount
                        ? devices[deviceIndex] : nullptr;

        // Assign as much as possible to this device.
        std::shared_ptr<ExecutionStep> step = plan->createNewStep(device);
        auto& queue = perDeviceQueue[deviceIndex];
        while (!queue.empty()) {
            uint32_t operationIndex = queue.front();
            queue.pop();
            int n = step->addOperation(operationIndex, *this);
            if (n != ANEURALNETWORKS_NO_ERROR) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "failed to add operation " << operationIndex << " to step";
#endif //HIFI_BUILD
                return n;
            }
            tracker.markProcessed(operationIndex, enqueueOnAppropriateDevice);
        }
    }

    int n = plan->finish(this, preference);
    if (VLOG_IS_ON(COMPILATION)) {
        Model model;
        setHidlModel(&model);
        VLOG(COMPILATION) << "ModelBuilder::partitionTheWork: original model: ";
        logModelToInfo(model);
        plan->dump();
    }
    return n;
}

PerformanceInfo ModelBuilder::getPerformanceInfo(const std::shared_ptr<Device> device,
                                                 uint32_t operationIndex) const {
    const Operation& operation = getOperation(operationIndex);
    // TODO This assumes that the type is dictated by the first operand. This is
    // currently the case but is not a safe assumption to make in the long term.
    const uint32_t operandIndex = operation.inputs[0];
    const OperandType operandType = mOperands[operandIndex].type;
    switch(operandType) {
        case OperandType::FLOAT32:
        case OperandType::TENSOR_FLOAT32:
            if (mRelaxComputationFloat32toFloat16) {
                return device->getRelaxedFloat32toFloat16Performance();
            } else {
                return device->getFloat32Performance();
            }
        case OperandType::INT32:
        case OperandType::UINT32:
        case OperandType::TENSOR_INT32:
        case OperandType::TENSOR_QUANT8_ASYMM:
            // For OEM, the real selection will be made from who can run the operand.
        case OperandType::OEM:
        case OperandType::TENSOR_OEM_BYTE:
            return device->getQuantized8Performance();
        default:
            nnAssert(false);
            return device->getQuantized8Performance();
    }
}
#endif //HIFI_BUILD

namespace {
// This class determines whether a given device can execute a given operation
class CanDo {
public:
    CanDo() {}

    void initialize(const ModelBuilder* model, std::shared_ptr<Device> device) {
        Model hidlModel;
        model->setHidlModel(&hidlModel);
        device->getSupportedOperations(hidlModel, &mSupportsOperationByIndex);
    }

    bool check(size_t operationIndex) const { return mSupportsOperationByIndex[operationIndex]; }

private:
    hidl_vec<bool> mSupportsOperationByIndex;
};
};  // anonymous namespace

#ifndef HIFI_BUILD
int ModelBuilder::findBestDeviceForEachOperation(
        uint32_t preference,
        const std::vector<std::shared_ptr<Device>>& devices,
        const size_t deviceCount,
        std::vector<int>* bestDeviceForOperation) const {

    // Note that deviceCount includes CPU, which has no entry in devices[]
    const size_t nonCpuDeviceCount = deviceCount - 1;

    std::vector<CanDo> canDo(nonCpuDeviceCount);
    for (size_t deviceIndex = 0; deviceIndex < nonCpuDeviceCount; deviceIndex++) {
        canDo[deviceIndex].initialize(this, devices[deviceIndex]);
    }

    // Figure out the best driver for each operation.
    const size_t operationCount = mOperations.size();
    for (size_t operationIndex = 0; operationIndex < operationCount; operationIndex++) {
        // Find which non-CPU device gives the best performance for this operation.
        int bestChoice = -1;
        float bestPerfVal = 0.0;  // Do not check bestPerfVal if bestChoice < 0.
        for (size_t deviceIndex = 0; deviceIndex < nonCpuDeviceCount; deviceIndex++) {
            const auto& device = devices[deviceIndex];
            if (canDo[deviceIndex].check(operationIndex)) {
                const PerformanceInfo perf = getPerformanceInfo(device, operationIndex);
                const float perfVal =
                            (preference == ANEURALNETWORKS_PREFER_LOW_POWER ? perf.powerUsage
                                                                            : perf.execTime);
                if (bestChoice < 0 || perfVal < bestPerfVal) {
                    bestChoice = deviceIndex;
                    bestPerfVal = perfVal;
                }
            } else {
                // Somewhat noisy logging, but only place where the user of
                // NNAPI can get feedback on why an operation was not run on a
                // specific device.
                // Logs O(operationCount * nonCpuDeviceCount) times, but
                // typically nonCpuDeviceCount is very small.
#ifndef HIFI_BUILD
                VLOG(COMPILATION) << "Device " << device->getName()
                                  << " can't do operation "
                                  << toString(getOperation(operationIndex).type);
#endif //HIFI_BUILD
            }
        }
        // If it's the OEM op, we'd better have a device able to do it.
        if (mOperations[operationIndex].type == OperationType::OEM_OPERATION) {
            if (bestChoice < 0) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "No driver can do the OEM op";
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }
        } else {
            // If no driver has been found, or if the best driver is not better than the CPU,
            // prefer the CPU. Since the performance is a ratio compared to the CPU performance,
            // by definition the performance of the CPU is 1.0.
            if (bestChoice < 0 || bestPerfVal >= 1.0) {
                bestChoice = nonCpuDeviceCount;  // The ID of the CPU.
            }
        }

        (*bestDeviceForOperation)[operationIndex] = bestChoice;
#ifndef HIFI_BUILD
        VLOG(COMPILATION) << "ModelBuilder::findBestDeviceForEachOperation("
                          << toString(getOperation(operationIndex).type)
                          << ") = "
                          << (*bestDeviceForOperation)[operationIndex];
#endif //HIFI_BUILD
    }
    return ANEURALNETWORKS_NO_ERROR;
}
#endif //HIFI_BUILD

} // namespace nn
} // namespace android
