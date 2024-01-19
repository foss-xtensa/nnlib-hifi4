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

#define LOG_TAG "CpuExecutor"

#include "CpuExecutor.h"

#include "NeuralNetworks.h"
#ifdef HIFI_NNLIB_OPT
#include "xa_nnlib_ann_api.h"
#else
#include "Operations.h"
#endif

#ifndef HIFI_BUILD
#include "Eigen/Core"
#endif //HiFi_BUILD
// b/109953668, disable OpenMP
#ifdef NNAPI_OPENMP
#include <omp.h>
#endif  // NNAPI_OPENMP

#ifndef HIFI_BUILD
#include <sys/mman.h>
#else
#include <cstring>
#define PROF_ALLOCATE
#include "xt_profiler.h"

#define PROFILER_START(op) \
    XTPWR_PROFILER_OPEN(0, op, "", 1, NULL, 0); \
    XTPWR_PROFILER_START(0);

#define PROFILER_STOP \
    XTPWR_PROFILER_STOP(0); \
    XTPWR_PROFILER_UPDATE(0); \
    XTPWR_PROFILER_CLOSE(0, 1, 0);

#endif //HIFI_BUILD


namespace android {
namespace nn {

// TODO: short term, make share memory mapping and updating a utility function.
// TODO: long term, implement mmap_fd as a hidl IMemory service.
#ifndef HIFI_BUILD
RunTimePoolInfo::RunTimePoolInfo(const hidl_memory& hidlMemory, bool* fail) {
    sp<IMemory> memory;
    uint8_t* buffer = nullptr;

    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory = mapMemory(hidlMemory);
        if (memory == nullptr) {
            LOG(ERROR) << "Can't map shared memory.";
            if (fail) *fail = true;
            return;
        }
        memory->update();
        buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
        if (buffer == nullptr) {
            LOG(ERROR) << "Can't access shared memory.";
            if (fail) *fail = true;
            return;
        }
    } else if (memType == "mmap_fd") {
        size_t size = hidlMemory.size();
        int fd = hidlMemory.handle()->data[0];
        int prot = hidlMemory.handle()->data[1];
        size_t offset = getSizeFromInts(hidlMemory.handle()->data[2],
                                        hidlMemory.handle()->data[3]);
        buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
        if (buffer == MAP_FAILED) {
            LOG(ERROR) << "RunTimePoolInfo::set(): Can't mmap the file descriptor.";
            if (fail) *fail = true;
            return;
        }
    } else {
        LOG(ERROR) << "RunTimePoolInfo::set(): unsupported hidl_memory type";
        if (fail) *fail = true;
        return;
    }

    mHidlMemory = hidlMemory;
    mBuffer     = buffer;
    mMemory     = memory;
}
#endif //HIFI_BUILD

RunTimePoolInfo::RunTimePoolInfo(uint8_t* buffer) {
    mBuffer = buffer;
}

RunTimePoolInfo::RunTimePoolInfo(RunTimePoolInfo&& other) {
    moveFrom(std::move(other));
    other.mBuffer = nullptr;
}

RunTimePoolInfo& RunTimePoolInfo::operator=(RunTimePoolInfo&& other) {
    if (this != &other) {
        release();
        moveFrom(std::move(other));
        other.mBuffer = nullptr;
    }
    return *this;
}

void RunTimePoolInfo::moveFrom(RunTimePoolInfo &&other) {
    mHidlMemory = std::move(other.mHidlMemory);
    mBuffer     = std::move(other.mBuffer);
    mMemory     = std::move(other.mMemory);
}

void RunTimePoolInfo::release() {
    if (mBuffer == nullptr) {
        return;
    }

    auto memType = mHidlMemory.name();
    if (memType == "ashmem") {
        // nothing to do
    } else if (memType == "mmap_fd") {
#ifndef HIFI_BUILD
        size_t size = mHidlMemory.size();
        if (munmap(mBuffer, size)) {
            LOG(ERROR) << "RunTimePoolInfo::release(): Can't munmap";
        }
#endif //HIFI_BUILD
    } else if (memType == "") {
        // Represents a POINTER argument; nothing to do
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "RunTimePoolInfo::release(): unsupported hidl_memory type";
#endif //HIFI_BUILD
    }

    mHidlMemory = hidl_memory();
    mMemory     = nullptr;
    mBuffer     = nullptr;
}

// Making sure the output data are correctly updated after execution.
bool RunTimePoolInfo::update() const {
    auto memType = mHidlMemory.name();
    if (memType == "ashmem") {
#ifndef HIFI_BUILD
        mMemory->commit();
#endif //HIFI_BUILD
        return true;
    } else if (memType == "mmap_fd") {
#ifndef HIFI_BUILD
        int prot = mHidlMemory.handle()->data[1];
        if (prot & PROT_WRITE) {
            size_t size = mHidlMemory.size();
            return msync(mBuffer, size, MS_SYNC) == 0;
        }
#endif //HIFI_BUILD
    }
    // No-op for other types of memory.
    return true;
}

#ifndef HIFI_BUILD
bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->clear();
    poolInfos->reserve(pools.size());
    bool fail = false;
    for (const auto& pool : pools) {
        poolInfos->emplace_back(pool, &fail);
    }
    if (fail) {
        LOG(ERROR) << "Could not map pools";
        poolInfos->clear();
        return false;
    }
    return true;
}
#endif //HIFI_BUILD

// Updates the RunTimeOperandInfo with the newly calculated shape.
// Allocate the buffer if we need to.
static bool setInfoAndAllocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    // For user-provided model output operands, the parameters must match the Shape
    // calculated from the preparation step.
    if (info->lifetime == OperandLifeTime::MODEL_OUTPUT) {
        if (info->type != shape.type ||
            info->dimensions != shape.dimensions) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid type or dimensions for model output";
#endif //HIFI_BUILD
            return false;
        }
        if (info->type == OperandType::TENSOR_QUANT8_ASYMM &&
            (info->scale != shape.scale || info->zeroPoint != shape.offset)) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "Invalid scale or zeroPoint for model output";
#endif //HIFI_BUILD
            return false;
        }
    }
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    info->scale = shape.scale;
    info->zeroPoint = shape.offset;
    if (info->lifetime == OperandLifeTime::TEMPORARY_VARIABLE && info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

// Ignore the .pools entry in model and request.  This will have been taken care of
// by the caller.
int CpuExecutor::run(const V1_0::Model& model, const Request& request,
                     const std::vector<RunTimePoolInfo>& modelPoolInfos,
                     const std::vector<RunTimePoolInfo>& requestPoolInfos) {
    return run(convertToV1_1(model), request, modelPoolInfos, requestPoolInfos);
}

int CpuExecutor::run(const V1_1::Model& model, const Request& request,
                     const std::vector<RunTimePoolInfo>& modelPoolInfos,
                     const std::vector<RunTimePoolInfo>& requestPoolInfos) {
#ifndef HIFI_BUILD
    VLOG(CPUEXE) << "CpuExecutor::run() with request("
                 << SHOW_IF_DEBUG(toString(request)) << ")";
#endif //HIFI_BUILD

    // b/109953668, disable OpenMP
#ifdef NNAPI_OPENMP
    ScopedOpenmpSettings openMpSettings;
#endif  // NNAPI_OPENMP

    mModel = &model;
    mRequest = &request; // TODO check if mRequest is needed
    initializeRunTimeInfo(modelPoolInfos, requestPoolInfos);
    // The model has serialized the operation in execution order.
    for (const auto& operation : model.operations) {
        int n = executeOperation(operation);
        if (n != ANEURALNETWORKS_NO_ERROR) {
            return n;
        }
    }
    for (auto& runtimeInfo : modelPoolInfos) {
        runtimeInfo.update();
    }
    for (auto& runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }
    mModel = nullptr;
    mRequest = nullptr;
#ifndef HIFI_BUILD
    VLOG(CPUEXE) << "Completed run normally";
#endif //HIFI_BUILD
    return ANEURALNETWORKS_NO_ERROR;
}

bool CpuExecutor::initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                        const std::vector<RunTimePoolInfo>& requestPoolInfos) {
#ifndef HIFI_BUILD
    VLOG(CPUEXE) << "CpuExecutor::initializeRunTimeInfo";
#endif //HIFI_BUILD
    const size_t count = mModel->operands.size();
    mOperands.resize(count);

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel->operands[i];
        RunTimeOperandInfo& to = mOperands[i];
        to.type = from.type;
        to.dimensions = from.dimensions;
        to.scale = from.scale;
        to.zeroPoint = from.zeroPoint;
        to.length = from.location.length;
        to.lifetime = from.lifetime;
        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel->operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE: {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < modelPoolInfos.size());
                auto& r = modelPoolInfos[poolIndex];
                to.buffer = r.getBuffer() + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.numberOfUsesLeft = 0;
                break;
            default:
                nnAssert(false);
                break;
        }
    }

    // Adjust the runtime info for the arguments passed to the model,
    // modifying the buffer location, and possibly the dimensions.
    auto updateForArguments = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                  const hidl_vec<RequestArgument>& arguments) {
        nnAssert(indexes.size() == arguments.size());
        for (size_t i = 0; i < indexes.size(); i++) {
            const uint32_t operandIndex = indexes[i];
            const RequestArgument& from = arguments[i];
            RunTimeOperandInfo& to = mOperands[operandIndex];
            if (from.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // from.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                // TODO make sure that's the case for the default CPU path.
                to.dimensions = from.dimensions;
            }
            if (from.hasNoValue) {
                to.lifetime = OperandLifeTime::NO_VALUE;
                nnAssert(to.buffer == nullptr);
            } else {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < requestPoolInfos.size());
                auto& r = requestPoolInfos[poolIndex];
                to.buffer = r.getBuffer() + from.location.offset;
            }
        }
    };
    updateForArguments(mModel->inputIndexes, mRequest->inputs);
    updateForArguments(mModel->outputIndexes, mRequest->outputs);

    return true;
}

void CpuExecutor::freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs) {
    for (uint32_t i : inputs) {
        auto& info = mOperands[i];
        // Check if it's a static or model input/output.
        if (info.numberOfUsesLeft == 0) {
            continue;
        }
        info.numberOfUsesLeft--;
        if (info.numberOfUsesLeft == 0) {
            nnAssert(info.buffer != nullptr);
            delete[] info.buffer;
            info.buffer = nullptr;
        }
    }
}

#ifdef HIFI_NNLIB_OPT
static void pad_shape(const RunTimeOperandInfo& input, RunTimeOperandInfo& output, Shape input_shape, Shape output_shape)
{
    int i, j;
    if(input.type == OperandType::TENSOR_FLOAT32)
    {
        const float *inp = reinterpret_cast<const float*>(input.buffer);
        float *out = reinterpret_cast<float*>(output.buffer);
        for(i = 0; i < getSizeOfDimension(input_shape, 0)*getSizeOfDimension(input_shape, 1)*getSizeOfDimension(input_shape, 2); i++)
        {
            for(j = 0; j < getSizeOfDimension(input_shape, 3); j++)
            {
                out[i*getSizeOfDimension(output_shape, 3)+j] = inp[i*getSizeOfDimension(input_shape, 3)+j];
            }
            for( ; j < getSizeOfDimension(output_shape, 3); j++)
            {
                out[i*getSizeOfDimension(output_shape, 3)+j] = 0.0f;
            }
        }
    }
    else if(input.type == OperandType::TENSOR_QUANT8_ASYMM)
    {
        const uint8_t *inp = reinterpret_cast<const uint8_t*>(input.buffer);
        uint8_t *out = reinterpret_cast<uint8_t*>(output.buffer);
        for(i = 0; i < getSizeOfDimension(input_shape, 0)*getSizeOfDimension(input_shape, 1)*getSizeOfDimension(input_shape, 2); i++)
        {
            for(j = 0; j < getSizeOfDimension(input_shape, 3); j++)
            {
                out[i*getSizeOfDimension(output_shape, 3)+j] = inp[i*getSizeOfDimension(input_shape, 3)+j];
            }
            for( ; j < getSizeOfDimension(output_shape, 3); j++)
            {
                out[i*getSizeOfDimension(output_shape, 3)+j] = output_shape.offset;
            }
        }
    }
}
#endif

int CpuExecutor::executeOperation(const Operation& operation) {
    // VLOG(CPUEXE) << "CpuExecutor::executeOperation(" << toString(operation) << ")";
    const hidl_vec<uint32_t>& ins = operation.inputs;
    const hidl_vec<uint32_t>& outs = operation.outputs;
    bool success = false;

    // Function to verify that the number of input and output parameters
    // matches what is expected.  Also checks that all the parameters have
    // values. This function is to be used only for operations that do not
    // accept optional arguments.
    // TODO Have a version that works for optional arguments.
#ifndef HIFI_BUILD
    auto allParametersPresent = [&operation, &ins, &outs, this](size_t requiredIns,
                                                                size_t requiredOuts) -> bool {
        auto verify = [&operation, this](size_t requiredCount, const hidl_vec<uint32_t>& indexes,
#endif //HIFI_BUILD
    auto allParametersPresent = [&ins, &outs, this](size_t requiredIns,
                                                                size_t requiredOuts) -> bool {
        auto verify = [this](size_t requiredCount, const hidl_vec<uint32_t>& indexes,
                          const char* type) -> bool {
            size_t actualCount = indexes.size();
            if (actualCount != requiredCount) {
#ifndef HIFI_BUILD
                LOG(ERROR) << getOperationName(operation.type)
                           << ": Invalid number of " << type << " operands. Got " << actualCount
                           << " of " << requiredCount;
#endif //HIFI_BUILD
                return false;
            }
            for (size_t i = 0; i < actualCount; i++) {
                if (mOperands[indexes[i]].lifetime == OperandLifeTime::NO_VALUE) {
#ifndef HIFI_BUILD
                    LOG(ERROR) << getOperationName(operation.type) << " " << type
                               << " operand " << i << " is required but missing.";
#endif //HIFI_BUILD
                    return false;
                }
            }
            return true;
        };
        return verify(requiredIns, ins, "in") && verify(requiredOuts, outs, "out");
    };

    switch (operation.type) {
        case OperationType::OEM_OPERATION: {
#ifndef HIFI_BUILD
            LOG(ERROR) << "OEM operation not supported for CPU execution";
#endif //HIFI_BUILD
            success = false;
        } break;
        case OperationType::ADD: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (in1.type == OperandType::TENSOR_FLOAT32) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("ADD float32");
                if(success) success = addFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     in1.shape(),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     in2.shape(),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
                PROFILER_STOP;
            } else if (in1.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("ADD aym8");
                if(success) success = addQuant8(reinterpret_cast<const uint8_t*>(in1.buffer),
                                    in1.shape(),
                                    reinterpret_cast<const uint8_t*>(in2.buffer),
                                    in2.shape(),
                                    activation,
                                    reinterpret_cast<uint8_t*>(out.buffer),
                                    outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::MUL: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (in1.type == OperandType::TENSOR_FLOAT32) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("MUL float32");
                if(success) success = mulFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     in1.shape(),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     in2.shape(),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
                PROFILER_STOP;
            } else if (in1.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("MUL aym8");
                if(success) success = 
                          mulQuant8(reinterpret_cast<const uint8_t*>(in1.buffer),
                                    in1.shape(),
                                    reinterpret_cast<const uint8_t*>(in2.buffer),
                                    in2.shape(),
                                    activation,
                                    reinterpret_cast<uint8_t*>(out.buffer),
                                    outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::FLOOR: {
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = floorPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("FLOOR float32");
                if(success) success = 
                          floorFloat32(reinterpret_cast<const float*>(input.buffer),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::DEQUANTIZE: {
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = dequantizePrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("DEQUANTIZE aym8");
                if(success) success = 
                          dequantizeQuant8ToFloat32(
                                  reinterpret_cast<const uint8_t*>(input.buffer),
                                  reinterpret_cast<float*>(output.buffer),
                                  input.shape());
                PROFILER_STOP;
            }
        } break;
        case OperationType::DEPTHWISE_CONV_2D: {
            const size_t inCount = ins.size();
            if ((inCount != 11 && inCount != 8) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding_left, padding_right;
            int32_t padding_top, padding_bottom;
            int32_t stride_width, stride_height;
            int32_t depth_multiplier;
            int32_t activation;

            if (inCount == 11) {
                padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
                padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
                padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
                padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
                depth_multiplier = getScalarData<int32_t>(mOperands[ins[9]]);
                activation       = getScalarData<int32_t>(mOperands[ins[10]]);
            } else {
                int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
                depth_multiplier = getScalarData<int32_t>(mOperands[ins[6]]);
                activation       = getScalarData<int32_t>(mOperands[ins[7]]);

                Shape inputShape = input.shape();
                Shape filterShape = filter.shape();
                int32_t input_width  = getSizeOfDimension(inputShape, 2);
                int32_t input_height = getSizeOfDimension(inputShape, 1);
                int32_t filter_width  = getSizeOfDimension(filterShape, 2);
                int32_t filter_height = getSizeOfDimension(filterShape, 1);
                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, padding_implicit,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, padding_implicit,
                                         &padding_top, &padding_bottom);
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();
#ifdef HIFI_NNLIB_OPT
            Shape filterShapePadded = filter.shape();
#endif

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifdef HIFI_NNLIB_OPT
                int32_t scratch_size;
                filterShapePadded.dimensions[3] = ((filterShapePadded.dimensions[3]+1)&(~1));
#endif
                success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                               padding_left, padding_right,
                                               padding_top, padding_bottom,
                                               stride_width, stride_height,
#ifndef HIFI_NNLIB_OPT
                                               &outShape);
#else
                                               &outShape, scratch_size);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("DEPTHWISE_CONV_2D float32");
                if(success) success = 
                          setInfoAndAllocateIfNeeded(&output, outShape) &&
                          depthwiseConvFloat32(reinterpret_cast<const float*>(input.buffer),
                                               input.shape(),
                                               reinterpret_cast<const float*>(filter.buffer),
                                               filter.shape(),
                                               reinterpret_cast<const float*>(bias.buffer),
                                               bias.shape(),
                                               padding_left, padding_right,
                                               padding_top, padding_bottom,
                                               stride_width, stride_height,
                                               depth_multiplier, activation,
                                               reinterpret_cast<float*>(output.buffer),
                                               outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);
                PROFILER_START("DEPTHWISE_CONV_2D float32");
                if(success) success = 
                          setInfoAndAllocateIfNeeded(&output, outShape) &&
                          depthwiseConvFloat32(reinterpret_cast<const float*>(input.buffer),
                                               input.shape(),
                                               reinterpret_cast<const float*>(filter.buffer),
                                               filter.shape(),
                                               reinterpret_cast<const float*>(bias.buffer),
                                               bias.shape(),
                                               padding_left, padding_right,
                                               padding_top, padding_bottom,
                                               stride_width, stride_height,
                                               depth_multiplier, activation,
                                               reinterpret_cast<float*>(output.buffer),
                                               outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifdef HIFI_NNLIB_OPT
                int32_t scratch_size;
                //filterShapePadded.dimensions[3] = ((filterShapePadded.dimensions[3]+3)&(~3));
#endif
                success = depthwiseConvPrepare(input.shape(), filter.shape(), bias.shape(),
                                               padding_left, padding_right,
                                               padding_top, padding_bottom,
                                               stride_width, stride_height,
#ifndef HIFI_NNLIB_OPT
                                               &outShape) &&
#else
                                               &outShape, scratch_size) &&
#endif
                          setInfoAndAllocateIfNeeded(&output, outShape);
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("DEPTHWISE_CONV_2D aym8");
                if(success) success = 
                          depthwiseConvQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                              input.shape(),
                                              reinterpret_cast<const uint8_t*>(filter.buffer),
                                              filter.shape(),
                                              reinterpret_cast<const int32_t*>(bias.buffer),
                                              bias.shape(),
                                              padding_left, padding_right,
                                              padding_top, padding_bottom,
                                              stride_width, stride_height,
                                              depth_multiplier, activation,
                                              reinterpret_cast<uint8_t*>(output.buffer),
                                              outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);
                PROFILER_START("DEPTHWISE_CONV_2D aym8");
                if(success) success = 
                          depthwiseConvQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                              input.shape(),
                                              reinterpret_cast<const uint8_t*>(filter.buffer),
                                              filter.shape(),
                                              reinterpret_cast<const int32_t*>(bias.buffer),
                                              bias.shape(),
                                              padding_left, padding_right,
                                              padding_top, padding_bottom,
                                              stride_width, stride_height,
                                              depth_multiplier, activation,
                                              reinterpret_cast<uint8_t*>(output.buffer),
                                              outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            }

        } break;
        case OperationType::CONV_2D: {
            const size_t inCount = ins.size();
            if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input  = mOperands[ins[0]];
            const RunTimeOperandInfo& filter = mOperands[ins[1]];
#ifdef HIFI_NNLIB_OPT
            RunTimeOperandInfo filter_padded = mOperands[ins[1]];
            filter_padded.buffer = NULL;
            filter_padded.lifetime = OperandLifeTime::TEMPORARY_VARIABLE;
#endif
            const RunTimeOperandInfo& bias   = mOperands[ins[2]];

            int32_t padding_left, padding_right;
            int32_t padding_top, padding_bottom;
            int32_t stride_width, stride_height;
            int32_t activation;

            if (inCount == 10) {
                padding_left     = getScalarData<int32_t>(mOperands[ins[3]]);
                padding_right    = getScalarData<int32_t>(mOperands[ins[4]]);
                padding_top      = getScalarData<int32_t>(mOperands[ins[5]]);
                padding_bottom   = getScalarData<int32_t>(mOperands[ins[6]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[7]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[8]]);
                activation       = getScalarData<int32_t>(mOperands[ins[9]]);
            } else {
                int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[3]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[4]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[5]]);
                activation       = getScalarData<int32_t>(mOperands[ins[6]]);

                Shape inputShape = input.shape();
                Shape filterShape = filter.shape();
                int32_t input_width  = getSizeOfDimension(inputShape, 2);
                int32_t input_height = getSizeOfDimension(inputShape, 1);
                int32_t filter_width  = getSizeOfDimension(filterShape, 2);
                int32_t filter_height = getSizeOfDimension(filterShape, 1);
                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, padding_implicit,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, padding_implicit,
                                         &padding_top, &padding_bottom);
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();
#ifdef HIFI_NNLIB_OPT
            Shape filterShapePadded = filter.shape();
#endif

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifdef HIFI_NNLIB_OPT
                int32_t scratch_size;
                filterShapePadded.dimensions[3] = ((filterShapePadded.dimensions[3]+1)&(~1));
#endif
                success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                      padding_left, padding_right,
                                      padding_top, padding_bottom,
                                      stride_width, stride_height,
#ifndef HIFI_NNLIB_OPT
                                      &outShape) &&
#else
                                      &outShape, scratch_size) &&
#endif
                          setInfoAndAllocateIfNeeded(&output, outShape);
#ifdef HIFI_NNLIB_OPT
                filter_padded.type = filterShapePadded.type;
                filter_padded.dimensions = filterShapePadded.dimensions;
                filter_padded.scale = filterShapePadded.scale;
                filter_padded.zeroPoint = filterShapePadded.offset;
                filter_padded.buffer = (uint8_t *)malloc(sizeOfData(filter_padded.type, filter_padded.dimensions));
#endif

#ifndef HIFI_NNLIB_OPT
                PROFILER_START("CONV_2D float32");
                if(success) success = 
                          convFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                      reinterpret_cast<const float*>(filter.buffer), filter.shape(),
                                      reinterpret_cast<const float*>(bias.buffer), bias.shape(),
                                      padding_left, padding_right,
                                      padding_top, padding_bottom,
                                      stride_width, stride_height, activation,
                                      reinterpret_cast<float*>(output.buffer), outShape);
                PROFILER_STOP;
#else
                pad_shape(filter, filter_padded, filter.shape(), filterShapePadded);
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);
                PROFILER_START("CONV_2D float32");
                if(success) success = 
                          convFloat32(reinterpret_cast<const float*>(input.buffer), input.shape(),
                                      reinterpret_cast<const float*>(filter.buffer), filter.shape(),
                                      reinterpret_cast<const float*>(bias.buffer), bias.shape(),
                                      padding_left, padding_right,
                                      padding_top, padding_bottom,
                                      stride_width, stride_height, activation,
                                      reinterpret_cast<float*>(output.buffer), outShape, p_scratch);
                PROFILER_STOP;
                free(filter_padded.buffer);
                free(p_scratch);
#endif
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifdef HIFI_NNLIB_OPT
                int32_t scratch_size;
                filterShapePadded.dimensions[3] = ((filterShapePadded.dimensions[3]+3)&(~3));
#endif
                success = convPrepare(input.shape(), filter.shape(), bias.shape(),
                                      padding_left, padding_right,
                                      padding_top, padding_bottom,
                                      stride_width, stride_height,
#ifndef HIFI_NNLIB_OPT
                                      &outShape) &&
#else
                                      &outShape, scratch_size) &&
#endif
                          setInfoAndAllocateIfNeeded(&output, outShape);
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("CONV_2D asym8");
                if(success) success = 
                          convQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<const uint8_t*>(filter.buffer),
                                     filter.shape(),
                                     reinterpret_cast<const int32_t*>(bias.buffer),
                                     bias.shape(),
                                     padding_left, padding_right,
                                     padding_top, padding_bottom,
                                     stride_width, stride_height, activation,
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);
                PROFILER_START("CONV_2D asym8");
                if(success) success = 
                          convQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<const uint8_t*>(filter.buffer),
                                     filter.shape(),
                                     reinterpret_cast<const int32_t*>(bias.buffer),
                                     bias.shape(),
                                     padding_left, padding_right,
                                     padding_top, padding_bottom,
                                     stride_width, stride_height, activation,
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            }
        } break;
        case OperationType::AVERAGE_POOL_2D: {
            const size_t inCount = ins.size();
            if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding_left, padding_right;
            int32_t padding_top, padding_bottom;
            int32_t stride_width, stride_height;
            int32_t filter_width, filter_height;
            int32_t activation;
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif

            if (inCount == 10) {
                padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
                padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
                padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
                padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
                activation       = getScalarData<int32_t>(mOperands[ins[9]]);
            } else {
                int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
                activation       = getScalarData<int32_t>(mOperands[ins[6]]);

                Shape inputShape = input.shape();
                int32_t input_width  = getSizeOfDimension(inputShape, 2);
                int32_t input_height = getSizeOfDimension(inputShape, 1);
                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, padding_implicit,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, padding_implicit,
                                         &padding_top, &padding_bottom);
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding_left, padding_right,
                                                padding_top, padding_bottom,
                                                stride_width, stride_height,
                                                filter_width, filter_height,
#ifndef HIFI_NNLIB_OPT
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                                                &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("AVERAGE_POOL_2D float32");
                if(success) success = 
                          averagePoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                             input.shape(),
                                             padding_left, padding_right,
                                             padding_top, padding_bottom,
                                             stride_width, stride_height,
                                             filter_width, filter_height, activation,
                                             reinterpret_cast<float*>(output.buffer),
                                             outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);
                
                PROFILER_START("AVERAGE_POOL_2D float32");
                if(success) success = 
                          averagePoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                             input.shape(),
                                             padding_left, padding_right,
                                             padding_top, padding_bottom,
                                             stride_width, stride_height,
                                             filter_width, filter_height, activation,
                                             reinterpret_cast<float*>(output.buffer),
                                             outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericPoolingPrepare(input.shape(),
                                                padding_left, padding_right,
                                                padding_top, padding_bottom,
                                                stride_width, stride_height,
                                                filter_width, filter_height,
#ifndef HIFI_NNLIB_OPT
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                                                &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("AVERAGE_POOL_2D asym8");
                if(success) success = 
                          averagePoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                            input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height, activation,
                                            reinterpret_cast<uint8_t*>(output.buffer),
                                            outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);

                PROFILER_START("AVERAGE_POOL_2D asym8");
                if(success) success = 
                          averagePoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                            input.shape(),
                                            padding_left, padding_right,
                                            padding_top, padding_bottom,
                                            stride_width, stride_height,
                                            filter_width, filter_height, activation,
                                            reinterpret_cast<uint8_t*>(output.buffer),
                                            outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            }
        } break;
        case OperationType::L2_POOL_2D: {
            const size_t inCount = ins.size();
            if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding_left, padding_right;
            int32_t padding_top, padding_bottom;
            int32_t stride_width, stride_height;
            int32_t filter_width, filter_height;
            int32_t activation;
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif

            if (inCount == 10) {
                padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
                padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
                padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
                padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
                activation       = getScalarData<int32_t>(mOperands[ins[9]]);
            } else {
                int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
                activation       = getScalarData<int32_t>(mOperands[ins[6]]);

                Shape inputShape = input.shape();
                int32_t input_width  = getSizeOfDimension(inputShape, 2);
                int32_t input_height = getSizeOfDimension(inputShape, 1);
                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, padding_implicit,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, padding_implicit,
                                         &padding_top, &padding_bottom);
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding_left, padding_right,
                                                padding_top, padding_bottom,
                                                stride_width, stride_height,
                                                filter_width, filter_height,
#ifndef HIFI_NNLIB_OPT
                                                &outShape);
#else
                                                &outShape, operation, scratch_size);
#endif
                PROFILER_START("L2_POOL_2D float32");
                if(success) success = 
                          setInfoAndAllocateIfNeeded(&output, outShape) &&
                          l2PoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                        input.shape(),
                                        padding_left, padding_right,
                                        padding_top, padding_bottom,
                                        stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<float*>(output.buffer),
                                        outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::MAX_POOL_2D: {
            const size_t inCount = ins.size();
            if ((inCount != 10 && inCount != 7) ||
                    !allParametersPresent(inCount, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];

            int32_t padding_left, padding_right;
            int32_t padding_top, padding_bottom;
            int32_t stride_width, stride_height;
            int32_t filter_width, filter_height;
            int32_t activation;
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif

            if (inCount == 10) {
                padding_left     = getScalarData<int32_t>(mOperands[ins[1]]);
                padding_right    = getScalarData<int32_t>(mOperands[ins[2]]);
                padding_top      = getScalarData<int32_t>(mOperands[ins[3]]);
                padding_bottom   = getScalarData<int32_t>(mOperands[ins[4]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[5]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[6]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[7]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[8]]);
                activation       = getScalarData<int32_t>(mOperands[ins[9]]);
            } else {
                int32_t padding_implicit = getScalarData<int32_t>(mOperands[ins[1]]);
                stride_width     = getScalarData<int32_t>(mOperands[ins[2]]);
                stride_height    = getScalarData<int32_t>(mOperands[ins[3]]);
                filter_width     = getScalarData<int32_t>(mOperands[ins[4]]);
                filter_height    = getScalarData<int32_t>(mOperands[ins[5]]);
                activation       = getScalarData<int32_t>(mOperands[ins[6]]);

                Shape inputShape = input.shape();
                int32_t input_width  = getSizeOfDimension(inputShape, 2);
                int32_t input_height = getSizeOfDimension(inputShape, 1);
                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, padding_implicit,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, padding_implicit,
                                         &padding_top, &padding_bottom);
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = genericPoolingPrepare(input.shape(),
                                                padding_left, padding_right,
                                                padding_top, padding_bottom,
                                                stride_width, stride_height,
                                                filter_width, filter_height,
#ifndef HIFI_NNLIB_OPT
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                                                &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("MAX_POOL_2D float32");
                if(success) success = 
                          maxPoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         padding_left, padding_right,
                                         padding_top, padding_bottom,
                                         stride_width, stride_height,
                                         filter_width, filter_height, activation,
                                         reinterpret_cast<float*>(output.buffer),
                                         outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);

                PROFILER_START("MAX_POOL_2D float32");
                if(success) success = 
                          maxPoolFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         padding_left, padding_right,
                                         padding_top, padding_bottom,
                                         stride_width, stride_height,
                                         filter_width, filter_height, activation,
                                         reinterpret_cast<float*>(output.buffer),
                                         outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericPoolingPrepare(input.shape(),
                                                padding_left, padding_right,
                                                padding_top, padding_bottom,
                                                stride_width, stride_height,
                                                filter_width, filter_height,
#ifndef HIFI_NNLIB_OPT
                                                &outShape);
#else
                                                &outShape, operation, scratch_size);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("MAX_POOL_2D asym8");
                if(success) success = 
                          setInfoAndAllocateIfNeeded(&output, outShape) &&
                          maxPoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        padding_left, padding_right,
                                        padding_top, padding_bottom,
                                        stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        outShape);
                PROFILER_STOP;
#else
                printf("Scratch %d \n", scratch_size);
                void *p_scratch = malloc(scratch_size);

                PROFILER_START("MAX_POOL_2D asym8");
                if(success) success = 
                          setInfoAndAllocateIfNeeded(&output, outShape) &&
                          maxPoolQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        padding_left, padding_right,
                                        padding_top, padding_bottom,
                                        stride_width, stride_height,
                                        filter_width, filter_height, activation,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        outShape, p_scratch);
                PROFILER_STOP;
                free(p_scratch);
#endif
            }

        } break;
        case OperationType::RELU: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU float32");
                if(success) success = 
                          reluFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU asym8");
                if(success) success = 
                          reluQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<uint8_t*>(output.buffer),
                                     outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::RELU1: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU1 float32");
                if(success) success = 
                          relu1Float32(reinterpret_cast<const float*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU1 asym8");
                if(success) success = 
                          relu1Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<uint8_t*>(output.buffer),
                                      outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::RELU6: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU6 float32");
                if(success) success = 
                          relu6Float32(reinterpret_cast<const float*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<float*>(output.buffer),
                                       outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("RELU6 asym8");
                if(success) success = 
                          relu6Quant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<uint8_t*>(output.buffer),
                                      outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::TANH: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("TANH float32");
                if(success) success = 
                          tanhFloat32(reinterpret_cast<const float*>(input.buffer),
                                      input.shape(),
                                      reinterpret_cast<float*>(output.buffer),
                                      outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::LOGISTIC: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("LOGISTIC float32");
                if(success) success = 
                          logisticFloat32(reinterpret_cast<const float*>(input.buffer),
                                          input.shape(),
                                          reinterpret_cast<float*>(output.buffer),
                                          outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("LOGISTIC asym8");
                if(success) success = 
                          logisticQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                         input.shape(),
                                         reinterpret_cast<uint8_t*>(output.buffer),
                                         outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::SOFTMAX: {
#ifdef HIFI_NNLIB_OPT
            int32_t scratch_size;
#endif
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            RunTimeOperandInfo& input = mOperands[ins[0]];
            float beta = getScalarData<float>(mOperands[ins[1]]);
            if (beta <= 0.0f) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "beta must be positive for softmax";
#endif //HIFI_BUILD
                return ANEURALNETWORKS_BAD_DATA;
            }

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
                PROFILER_START("SOFTMAX float32");
                if(success) success = 
                          softmaxFloat32(reinterpret_cast<const float*>(input.buffer),
                                         input.shape(),
                                         beta,
                                         reinterpret_cast<float*>(output.buffer),
                                         output.shape());
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_NNLIB_OPT
                success = genericActivationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#else
                success = genericActivationPrepare(input.shape(), &outShape, operation, scratch_size) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
#endif
#ifndef HIFI_NNLIB_OPT
                PROFILER_START("SOFTMAX asym8");
                if(success) success = 
                          softmaxQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                        input.shape(),
                                        beta,
                                        reinterpret_cast<uint8_t*>(output.buffer),
                                        output.shape());
                PROFILER_STOP;
#else
                {
                    void *p_scratch = malloc(scratch_size);
                    PROFILER_START("SOFTMAX asym8");
                    if(success) success = 
                        softmaxQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                      input.shape(),
                                      beta,
                                      reinterpret_cast<uint8_t*>(output.buffer),
                                      output.shape(), p_scratch);
                    PROFILER_STOP;
                }
#endif
            }
        } break;
        case OperationType::FULLY_CONNECTED: {
            if (!allParametersPresent(4, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            RunTimeOperandInfo& input   = mOperands[ins[0]];
            RunTimeOperandInfo& weights = mOperands[ins[1]];
            RunTimeOperandInfo& bias    = mOperands[ins[2]];

            int32_t activation = getScalarData<int32_t>(mOperands[ins[3]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("FULLY_CONNECTED float32");
                if(success) success = 
                          fullyConnectedFloat32(reinterpret_cast<const float*>(input.buffer),
                                                input.shape(),
                                                reinterpret_cast<const float*>(weights.buffer),
                                                weights.shape(),
                                                reinterpret_cast<const float*>(bias.buffer),
                                                bias.shape(),
                                                activation,
                                                reinterpret_cast<float*>(output.buffer),
                                                outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = fullyConnectedPrepare(input.shape(), weights.shape(), bias.shape(),
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("FULLY_CONNECTED asym8");
                if(success) success = 
                          fullyConnectedQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                               input.shape(),
                                               reinterpret_cast<const uint8_t*>(weights.buffer),
                                               weights.shape(),
                                               reinterpret_cast<const int32_t*>(bias.buffer),
                                               bias.shape(),
                                               activation,
                                               reinterpret_cast<uint8_t*>(output.buffer),
                                               outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::CONCATENATION: {
            if (outs.size() != 1 || ins.size() < 2) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            int numInputTensors = ins.size() - 1;
            int32_t axis = getScalarData<int32_t>(mOperands[ins[numInputTensors]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            const RunTimeOperandInfo& firstInput = mOperands[ins[0]];
            if (firstInput.type == OperandType::TENSOR_FLOAT32) {
                std::vector<Shape> inputShapes(numInputTensors);
                std::vector<const float*> inputDataPtrs(numInputTensors);

                for (int i=0; i<numInputTensors; i++) {
                    RunTimeOperandInfo& input = mOperands[ins[i]];
                    inputShapes[i] = input.shape();
                    inputDataPtrs[i] = reinterpret_cast<const float*>(input.buffer);
                }
                success = concatenationPrepare(inputShapes, axis, &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("CONCATENATION float32");
                if(success) success = 
                          concatenationFloat32(inputDataPtrs, inputShapes, axis,
                                               reinterpret_cast<float*>(output.buffer), outShape);
                PROFILER_STOP;
            } else if (firstInput.type == OperandType::TENSOR_QUANT8_ASYMM) {
                std::vector<Shape> inputShapes(numInputTensors);
                std::vector<const uint8_t*> inputDataPtrs(numInputTensors);

                for (int i=0; i<numInputTensors; i++) {
                    RunTimeOperandInfo& input = mOperands[ins[i]];
                    inputShapes[i] = input.shape();
                    inputDataPtrs[i] = reinterpret_cast<const uint8_t*>(input.buffer);
                }
                success = concatenationPrepare(inputShapes, axis, &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("CONCATENATION asym8");
                if(success) success = 
                          concatenationQuant8(inputDataPtrs, inputShapes, axis,
                                              reinterpret_cast<uint8_t*>(output.buffer),
                                              outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::L2_NORMALIZATION: {
            if (!allParametersPresent(1, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("L2_NORMALIZATION float32");
                if(success) success = 
                          l2normFloat32(reinterpret_cast<const float*>(input.buffer),
                                        input.shape(),
                                        reinterpret_cast<float*>(output.buffer),
                                        outShape);
                PROFILER_STOP;
            } else if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("L2_NORMALIZATION asym8");
                if(success) success = 
                          l2normQuant8(reinterpret_cast<const uint8_t*>(input.buffer),
                                       input.shape(),
                                       reinterpret_cast<uint8_t*>(output.buffer),
                                       outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::LOCAL_RESPONSE_NORMALIZATION: {
            if (!allParametersPresent(5, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t radius = getScalarData<int32_t>(mOperands[ins[1]]);
            float bias = getScalarData<float>(mOperands[ins[2]]);
            float alpha = getScalarData<float>(mOperands[ins[3]]);
            float beta = getScalarData<float>(mOperands[ins[4]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = genericNormalizationPrepare(input.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("LOCAL_RESPONSE_NORMALIZATION float32");
                if(success) success = 
                          localResponseNormFloat32(reinterpret_cast<const float*>(input.buffer),
                                                   input.shape(),
                                                   radius, bias, alpha, beta,
                                                   reinterpret_cast<float*>(output.buffer),
                                                   outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::RESHAPE: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& targetShape = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = reshapePrepare(input.shape(),
                                     reinterpret_cast<const int32_t*>(targetShape.buffer),
                                     getNumberOfElements(targetShape.shape()),
                                     &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
            PROFILER_START("RESHAPE");
            if(success) success = 
                      reshapeGeneric(reinterpret_cast<const void*>(input.buffer),
                                     input.shape(),
                                     reinterpret_cast<void*>(output.buffer),
                                     outShape);
            PROFILER_STOP;
        } break;
        case OperationType::RESIZE_BILINEAR: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t width = getScalarData<int32_t>(mOperands[ins[1]]);
            int32_t height = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            if (input.type == OperandType::TENSOR_FLOAT32) {
                success = resizeBilinearPrepare(input.shape(),
                                                width, height,
                                                &outShape) &&
                          setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("RESIZE_BILINEAR");
                if(success) success = 
                          resizeBilinearFloat32(reinterpret_cast<const float*>(input.buffer),
                                                input.shape(),
                                                reinterpret_cast<float*>(output.buffer),
                                                outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::DEPTH_TO_SPACE: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = depthToSpacePrepare(input.shape(),
                                          blockSize,
                                          &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
            PROFILER_START("DEPTH_TO_SPACE");
            if(success) success = 
                      depthToSpaceGeneric(input.buffer,
                                          input.shape(),
                                          blockSize,
                                          output.buffer,
                                          outShape);
            PROFILER_STOP;
        } break;
        case OperationType::SPACE_TO_DEPTH: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            int32_t blockSize = getScalarData<int32_t>(mOperands[ins[1]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = spaceToDepthPrepare(input.shape(),
                                          blockSize,
                                          &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
            PROFILER_START("SPACE_TO_DEPTH");
            if(success) success = 
                      spaceToDepthGeneric(input.buffer,
                                          input.shape(),
                                          blockSize,
                                          output.buffer,
                                          outShape);
            PROFILER_STOP;
        } break;
        case OperationType::EMBEDDING_LOOKUP: {
            const RunTimeOperandInfo &values =
                mOperands[ins[EmbeddingLookup::kValueTensor]];
            const RunTimeOperandInfo &lookups =
                mOperands[ins[EmbeddingLookup::kLookupTensor]];
            RunTimeOperandInfo &output =
                mOperands[outs[EmbeddingLookup::kOutputTensor]];

            Shape outputShape;
            EmbeddingLookup lookup(operation, mOperands);

            success = embeddingLookupPrepare(values.shape(), lookups.shape(), &outputShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape);
            PROFILER_START("EMBEDDING_LOOKUP");
            if(success) success = 
                lookup.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::HASHTABLE_LOOKUP: {
            const RunTimeOperandInfo &lookups =
                mOperands[ins[HashtableLookup::kLookupTensor]];
            const RunTimeOperandInfo &keys =
                mOperands[ins[HashtableLookup::kKeyTensor]];
            const RunTimeOperandInfo &values =
                mOperands[ins[HashtableLookup::kValueTensor]];

            RunTimeOperandInfo &output =
                mOperands[outs[HashtableLookup::kOutputTensor]];
            RunTimeOperandInfo &hits =
                mOperands[outs[HashtableLookup::kHitsTensor]];

            Shape outputShape, hitShape;
            HashtableLookup lookup(operation, mOperands);

            success = hashtableLookupPrepare(lookups.shape(), keys.shape(), values.shape(),
                                             &outputShape, &hitShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape) &&
                setInfoAndAllocateIfNeeded(&hits, hitShape);
            PROFILER_START("HASHTABLE_LOOKUP");
            if(success) success = 
                lookup.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::LSH_PROJECTION: {
            RunTimeOperandInfo &output =
                mOperands[outs[LSHProjection::kOutputTensor]];

            Shape outputShape;
            LSHProjection lsh(operation, mOperands);

            success = LSHProjection::Prepare(operation, mOperands,
                                             &outputShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape);
            PROFILER_START("LSH_PROJECTION");
            if(success) success = 
                lsh.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::LSTM: {
            RunTimeOperandInfo &scratch =
                mOperands[outs[LSTMCell::kScratchBufferTensor]];
            RunTimeOperandInfo &outputStateOut =
                mOperands[outs[LSTMCell::kOutputStateOutTensor]];
            RunTimeOperandInfo &cellStateOut =
                mOperands[outs[LSTMCell::kCellStateOutTensor]];
            RunTimeOperandInfo &output =
                mOperands[outs[LSTMCell::kOutputTensor]];

            Shape scratchShape, outputStateShape, cellStateShape, outputShape;
            LSTMCell lstm_cell(operation, mOperands);

            success = LSTMCell::Prepare(operation, mOperands,
                                        &scratchShape, &outputStateShape,
                                        &cellStateShape, &outputShape) &&
                setInfoAndAllocateIfNeeded(&scratch, scratchShape) &&
                setInfoAndAllocateIfNeeded(&outputStateOut, outputStateShape) &&
                setInfoAndAllocateIfNeeded(&cellStateOut, cellStateShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape);
            PROFILER_START("LSTM");
            if(success) success = 
                lstm_cell.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::RNN: {
            RunTimeOperandInfo &hiddenStateOut =
                mOperands[outs[RNN::kHiddenStateOutTensor]];
            RunTimeOperandInfo &output =
                mOperands[outs[RNN::kOutputTensor]];

            Shape hiddenStateShape, outputShape;
            RNN rnn_cell(operation, mOperands);

            success = RNN::Prepare(operation, mOperands,
                                   &hiddenStateShape, &outputShape) &&
                setInfoAndAllocateIfNeeded(&hiddenStateOut, hiddenStateShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape);
            PROFILER_START("RNN");
            if(success) success = 
                rnn_cell.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::SVDF: {
            RunTimeOperandInfo &stateOut =
                mOperands[outs[SVDF::kStateOutTensor]];
            RunTimeOperandInfo &output =
                mOperands[outs[SVDF::kOutputTensor]];

            Shape stateShape, outputShape;
            SVDF svdf(operation, mOperands);

            success = SVDF::Prepare(operation, mOperands,
                                    &stateShape, &outputShape) &&
                setInfoAndAllocateIfNeeded(&stateOut, stateShape) &&
                setInfoAndAllocateIfNeeded(&output, outputShape);
            PROFILER_START("SVDF");
            if(success) success = 
                svdf.Eval();
            PROFILER_STOP;
        } break;
        case OperationType::BATCH_TO_SPACE_ND: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& blockSize = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = batchToSpacePrepare(input.shape(),
                                          reinterpret_cast<const int32_t*>(blockSize.buffer),
                                          blockSize.shape(),
                                          &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("BATCH_TO_SPACE_ND");
                if(success) success = 
                      batchToSpaceGeneric(input.buffer,
                                          input.shape(),
                                          reinterpret_cast<const int32_t*>(blockSize.buffer),
                                          output.buffer,
                                          outShape);
                PROFILER_STOP;
        } break;
        case OperationType::SPACE_TO_BATCH_ND: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& blockSize = mOperands[ins[1]];
            const RunTimeOperandInfo& paddings = mOperands[ins[2]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = spaceToBatchPrepare(input.shape(),
                                          reinterpret_cast<const int32_t*>(blockSize.buffer),
                                          blockSize.shape(),
                                          reinterpret_cast<const int32_t*>(paddings.buffer),
                                          paddings.shape(),
                                          &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("SPACE_TO_BATCH_ND");
                if(success) success = 
                      spaceToBatchGeneric(input.buffer,
                                          input.shape(),
                                          reinterpret_cast<const int32_t*>(blockSize.buffer),
                                          reinterpret_cast<const int32_t*>(paddings.buffer),
                                          paddings.shape(),
                                          output.buffer,
                                          outShape);
                PROFILER_STOP;
        } break;
        case OperationType::PAD: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& paddings = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = padPrepare(input.shape(),
                                 reinterpret_cast<const int32_t*>(paddings.buffer),
                                 paddings.shape(),
                                 &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("PAD");
                if(success) success = 
                      padGeneric(input.buffer,
                                 input.shape(),
                                 reinterpret_cast<const int32_t*>(paddings.buffer),
                                 output.buffer,
                                 outShape);
                PROFILER_STOP;
        } break;
        case OperationType::SQUEEZE: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& squeezeDims = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = squeezePrepare(input.shape(),
                                     reinterpret_cast<const int32_t*>(squeezeDims.buffer),
                                     squeezeDims.shape(),
                                     &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("SQUEEZE");
                if(success) success = 
                      squeezeGeneric(input.buffer,
                                     input.shape(),
                                     output.buffer,
                                     outShape);
                PROFILER_STOP;
        } break;
        case OperationType::TRANSPOSE: {
            if (!allParametersPresent(2, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& perms = mOperands[ins[1]];

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = transposePrepare(input.shape(),
                                       reinterpret_cast<const int32_t*>(perms.buffer),
                                       perms.shape(),
                                       &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("TRANSPOSE");
                if(success) success = 
                      transposeGeneric(input.buffer,
                                       input.shape(),
                                       reinterpret_cast<const int32_t*>(perms.buffer),
                                       perms.shape(),
                                       output.buffer,
                                       outShape);
                PROFILER_STOP;
        } break;
        case OperationType::STRIDED_SLICE: {
            if (!allParametersPresent(7, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& begins = mOperands[ins[1]];
            const RunTimeOperandInfo& ends = mOperands[ins[2]];
            const RunTimeOperandInfo& strides = mOperands[ins[3]];
            int32_t beginMask = getScalarData<int32_t>(mOperands[ins[4]]);
            int32_t endMask = getScalarData<int32_t>(mOperands[ins[5]]);
            int32_t shrinkAxisMask = getScalarData<int32_t>(mOperands[ins[6]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = stridedSlicePrepare(input.shape(),
                                          reinterpret_cast<const int32_t*>(begins.buffer),
                                          begins.shape(),
                                          reinterpret_cast<const int32_t*>(ends.buffer),
                                          ends.shape(),
                                          reinterpret_cast<const int32_t*>(strides.buffer),
                                          strides.shape(),
                                          beginMask, endMask, shrinkAxisMask,
                                          &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("STRIDED_SLICE");
                if(success) success = 
                      stridedSliceGeneric(input.buffer,
                                          input.shape(),
                                          reinterpret_cast<const int32_t*>(begins.buffer),
                                          reinterpret_cast<const int32_t*>(ends.buffer),
                                          reinterpret_cast<const int32_t*>(strides.buffer),
                                          beginMask, endMask, shrinkAxisMask,
                                          output.buffer,
                                          outShape);
                PROFILER_STOP;
        } break;
        case OperationType::DIV: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (in1.type == OperandType::TENSOR_FLOAT32) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("DIV float32");
                if(success) success = 
                          divFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     in1.shape(),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     in2.shape(),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::SUB: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& in1 = mOperands[ins[0]];
            const RunTimeOperandInfo& in2 = mOperands[ins[1]];
            int32_t activation = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& out = mOperands[outs[0]];
            Shape outShape = out.shape();

            if (in1.type == OperandType::TENSOR_FLOAT32) {
                success = addMulPrepare(in1.shape(), in2.shape(), &outShape) &&
                          setInfoAndAllocateIfNeeded(&out, outShape);
                PROFILER_START("SUB float32");
                if(success) success = 
                          subFloat32(reinterpret_cast<const float*>(in1.buffer),
                                     in1.shape(),
                                     reinterpret_cast<const float*>(in2.buffer),
                                     in2.shape(),
                                     activation,
                                     reinterpret_cast<float*>(out.buffer),
                                     outShape);
                PROFILER_STOP;
            }
        } break;
        case OperationType::MEAN: {
            if (!allParametersPresent(3, 1)) {
                return ANEURALNETWORKS_BAD_DATA;
            }
            const RunTimeOperandInfo& input = mOperands[ins[0]];
            const RunTimeOperandInfo& axis = mOperands[ins[1]];
            int32_t keepDims = getScalarData<int32_t>(mOperands[ins[2]]);

            RunTimeOperandInfo& output = mOperands[outs[0]];
            Shape outShape = output.shape();

            success = meanPrepare(input.shape(),
                                  reinterpret_cast<const int32_t*>(axis.buffer),
                                  axis.shape(),
                                  keepDims > 0,
                                  &outShape) &&
                      setInfoAndAllocateIfNeeded(&output, outShape);
                PROFILER_START("MEAN");
                if(success) success = 
                      meanGeneric(input.buffer,
                                  input.shape(),
                                  reinterpret_cast<const int32_t*>(axis.buffer),
                                  axis.shape(),
                                  keepDims > 0,
                                  output.buffer,
                                  outShape);
                PROFILER_STOP;
        } break;
        default:
            nnAssert(false);
            break;
    }
    if (!success) {
#ifndef HIFI_BUILD
        LOG(ERROR) << getOperationName(operation.type) << " failed.";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_OP_FAILED;
    }

    freeNoLongerUsedOperands(ins);
    return ANEURALNETWORKS_NO_ERROR;
}

// b/109953668, disable OpenMP
#ifdef NNAPI_OPENMP
ScopedOpenmpSettings::ScopedOpenmpSettings() {
    mBlocktimeInitial = kmp_get_blocktime();
    kmp_set_blocktime(20);  // ms, see b/109645291

#if NNAPI_LIMIT_CPU_THREADS
    // Code not yet enabled. Choosing the number of threads to be based on
    // benchmarking. See longer comment by the class declaration.
    mMaxThreadsInitial = Eigen::nbThreads();
    const int nProcs = omp_get_num_procs();
    int threads = nProcs;
    if (nProcs >= 8) {
        threads = nProcs - 4;
    } else if (nProcs >= 4) {
        threads = nProcs - 2;
    }
    Eigen::setNbThreads(threads);
#endif
}

ScopedOpenmpSettings::~ScopedOpenmpSettings() {
    kmp_set_blocktime(mBlocktimeInitial);
#if NNAPI_LIMIT_CPU_THREADS
    Eigen::setNbThreads(mMaxThreadsInitial);
#endif
}
#endif  // NNAPI_OPENMP

} // namespace nn
} // namespace android
