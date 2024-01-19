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

// Contains all the entry points to the C Neural Networks API.
// We do basic validation of the operands and then call the class
// that implements the functionality.

#define LOG_TAG "NeuralNetworks"

#include "NeuralNetworks.h"

#ifndef HIFI_BUILD
#include "Callbacks.h"
#endif //HIFI_BUILD
#include "CompilationBuilder.h"
#include "ExecutionBuilder.h"
#include "Manager.h"
#include "Memory.h"
#include "NeuralNetworksOEM.h"
#include "ModelBuilder.h"
#include "Utils.h"

#include <memory>
#include <vector>

// Make sure the constants defined in the header files have not changed values.
// IMPORTANT: When adding new values, update kNumberOfDataTypes or kNumberOfDataTypesOEM
// in Utils.h.
static_assert(ANEURALNETWORKS_FLOAT32 == 0, "ANEURALNETWORKS_FLOAT32 has changed");
static_assert(ANEURALNETWORKS_INT32 == 1, "ANEURALNETWORKS_INT32 has changed");
static_assert(ANEURALNETWORKS_UINT32 == 2, "ANEURALNETWORKS_UINT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_FLOAT32 == 3,
              "ANEURALNETWORKS_TENSOR_FLOAT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_INT32 == 4, "ANEURALNETWORKS_TENSOR_INT32 has changed");
static_assert(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == 5,
              "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM has changed");
static_assert(ANEURALNETWORKS_OEM_SCALAR == 10000, "ANEURALNETWORKS_OEM_SCALAR has changed");
static_assert(ANEURALNETWORKS_TENSOR_OEM_BYTE == 10001,
              "ANEURALNETWORKS_TENSOR_OEM_BYTE has changed");

// IMPORTANT: When adding new values, update kNumberOfOperationTypes or
// kNumberOfOperationTypesOEMin Utils.h.
static_assert(ANEURALNETWORKS_ADD == 0, "ANEURALNETWORKS_ADD has changed");
static_assert(ANEURALNETWORKS_AVERAGE_POOL_2D == 1,
              "ANEURALNETWORKS_AVERAGE_POOL_2D has changed");
static_assert(ANEURALNETWORKS_CONCATENATION == 2, "ANEURALNETWORKS_CONCATENATION has changed");
static_assert(ANEURALNETWORKS_CONV_2D == 3, "ANEURALNETWORKS_CONV_2D has changed");
static_assert(ANEURALNETWORKS_DEPTHWISE_CONV_2D == 4,
              "ANEURALNETWORKS_DEPTHWISE_CONV_2D has changed");
static_assert(ANEURALNETWORKS_DEPTH_TO_SPACE == 5,
              "ANEURALNETWORKS_DEPTH_TO_SPACE has changed");
static_assert(ANEURALNETWORKS_DEQUANTIZE == 6, "ANEURALNETWORKS_DEQUANTIZE has changed");
static_assert(ANEURALNETWORKS_EMBEDDING_LOOKUP == 7,
              "ANEURALNETWORKS_EMBEDDING_LOOKUP has changed");
static_assert(ANEURALNETWORKS_FLOOR == 8, "ANEURALNETWORKS_FLOOR has changed");
static_assert(ANEURALNETWORKS_FULLY_CONNECTED == 9,
              "ANEURALNETWORKS_FULLY_CONNECTED has changed");
static_assert(ANEURALNETWORKS_HASHTABLE_LOOKUP == 10,
              "ANEURALNETWORKS_HASHTABLE_LOOKUP has changed");
static_assert(ANEURALNETWORKS_L2_NORMALIZATION == 11,
              "ANEURALNETWORKS_L2_NORMALIZATION has changed");
static_assert(ANEURALNETWORKS_L2_POOL_2D == 12, "ANEURALNETWORKS_L2_POOL has changed");
static_assert(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION == 13,
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION has changed");
static_assert(ANEURALNETWORKS_LOGISTIC == 14, "ANEURALNETWORKS_LOGISTIC has changed");
static_assert(ANEURALNETWORKS_LSH_PROJECTION == 15,
              "ANEURALNETWORKS_LSH_PROJECTION has changed");
static_assert(ANEURALNETWORKS_LSTM == 16, "ANEURALNETWORKS_LSTM has changed");
static_assert(ANEURALNETWORKS_MAX_POOL_2D == 17, "ANEURALNETWORKS_MAX_POOL has changed");
static_assert(ANEURALNETWORKS_MUL == 18, "ANEURALNETWORKS_MUL has changed");
static_assert(ANEURALNETWORKS_RELU == 19, "ANEURALNETWORKS_RELU has changed");
static_assert(ANEURALNETWORKS_RELU1 == 20, "ANEURALNETWORKS_RELU1 has changed");
static_assert(ANEURALNETWORKS_RELU6 == 21, "ANEURALNETWORKS_RELU6 has changed");
static_assert(ANEURALNETWORKS_RESHAPE == 22, "ANEURALNETWORKS_RESHAPE has changed");
static_assert(ANEURALNETWORKS_RESIZE_BILINEAR == 23,
              "ANEURALNETWORKS_RESIZE_BILINEAR has changed");
static_assert(ANEURALNETWORKS_RNN == 24, "ANEURALNETWORKS_RNN has changed");
static_assert(ANEURALNETWORKS_SOFTMAX == 25, "ANEURALNETWORKS_SOFTMAX has changed");
static_assert(ANEURALNETWORKS_SPACE_TO_DEPTH == 26,
              "ANEURALNETWORKS_SPACE_TO_DEPTH has changed");
static_assert(ANEURALNETWORKS_SVDF == 27, "ANEURALNETWORKS_SVDF has changed");
static_assert(ANEURALNETWORKS_TANH == 28, "ANEURALNETWORKS_TANH has changed");

static_assert(ANEURALNETWORKS_BATCH_TO_SPACE_ND == 29, "ANEURALNETWORKS_BATCH_TO_SPACE_ND has changed");
static_assert(ANEURALNETWORKS_DIV == 30, "ANEURALNETWORKS_DIV has changed");
static_assert(ANEURALNETWORKS_MEAN == 31, "ANEURALNETWORKS_MEAN has changed");
static_assert(ANEURALNETWORKS_PAD == 32, "ANEURALNETWORKS_PAD has changed");
static_assert(ANEURALNETWORKS_SPACE_TO_BATCH_ND == 33, "ANEURALNETWORKS_SPACE_TO_BATCH_ND has changed");
static_assert(ANEURALNETWORKS_SQUEEZE == 34, "ANEURALNETWORKS_SQUEEZE has changed");
static_assert(ANEURALNETWORKS_STRIDED_SLICE == 35, "ANEURALNETWORKS_STRIDED_SLICE has changed");
static_assert(ANEURALNETWORKS_SUB == 36, "ANEURALNETWORKS_TANH has changed");
static_assert(ANEURALNETWORKS_TRANSPOSE == 37, "ANEURALNETWORKS_TRANSPOSE has changed");

static_assert(ANEURALNETWORKS_OEM_OPERATION == 10000,
              "ANEURALNETWORKS_OEM_OPERATION has changed");

static_assert(ANEURALNETWORKS_FUSED_NONE == 0, "ANEURALNETWORKS_FUSED_NONE has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU == 1, "ANEURALNETWORKS_FUSED_RELU has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU1 == 2, "ANEURALNETWORKS_FUSED_RELU1 has changed");
static_assert(ANEURALNETWORKS_FUSED_RELU6 == 3, "ANEURALNETWORKS_FUSED_RELU6 has changed");

static_assert(ANEURALNETWORKS_PREFER_LOW_POWER == 0,
              "ANEURALNETWORKS_PREFER_LOW_POWER has changed");
static_assert(ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER == 1,
              "ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER has changed");
static_assert(ANEURALNETWORKS_PREFER_SUSTAINED_SPEED == 2,
              "ANEURALNETWORKS_PREFER_SUSTAINED_SPEED has changed");

static_assert(ANEURALNETWORKS_NO_ERROR == 0, "ANEURALNETWORKS_NO_ERROR has changed");
static_assert(ANEURALNETWORKS_OUT_OF_MEMORY == 1, "ANEURALNETWORKS_OUT_OF_MEMORY has changed");
static_assert(ANEURALNETWORKS_INCOMPLETE == 2, "ANEURALNETWORKS_INCOMPLETE has changed");
static_assert(ANEURALNETWORKS_UNEXPECTED_NULL == 3,
              "ANEURALNETWORKS_UNEXPECTED_NULL has changed");
static_assert(ANEURALNETWORKS_BAD_DATA == 4, "ANEURALNETWORKS_BAD_DATA has changed");
static_assert(ANEURALNETWORKS_OP_FAILED == 5, "ANEURALNETWORKS_OP_FAILED has changed");
static_assert(ANEURALNETWORKS_BAD_STATE == 6, "ANEURALNETWORKS_BAD_STATE has changed");

static_assert(ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES == 128,
              "ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES has changed");

// Make sure that the constants are compatible with the values defined in
// hardware/interfaces/neuralnetworks/1.0/types.hal.
static_assert(static_cast<int32_t>(OperandType::OEM) == ANEURALNETWORKS_OEM_SCALAR,
              "OEM != ANEURALNETWORKS_OEM");
static_assert(static_cast<int32_t>(OperandType::FLOAT32) == ANEURALNETWORKS_FLOAT32,
              "FLOAT32 != ANEURALNETWORKS_FLOAT32");
static_assert(static_cast<int32_t>(OperandType::INT32) == ANEURALNETWORKS_INT32,
              "INT32 != ANEURALNETWORKS_INT32");
static_assert(static_cast<int32_t>(OperandType::UINT32) == ANEURALNETWORKS_UINT32,
              "UINT32 != ANEURALNETWORKS_UINT32");
static_assert(static_cast<int32_t>(OperandType::TENSOR_OEM_BYTE) == ANEURALNETWORKS_TENSOR_OEM_BYTE,
              "TENSOR_OEM_BYTE != ANEURALNETWORKS_TENSOR_OEM_BYTE");
static_assert(static_cast<int32_t>(OperandType::TENSOR_FLOAT32) == ANEURALNETWORKS_TENSOR_FLOAT32,
              "TENSOR_FLOAT32 != ANEURALNETWORKS_TENSOR_FLOAT32");
static_assert(static_cast<int32_t>(OperandType::TENSOR_QUANT8_ASYMM) ==
                          ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
              "TENSOR_QUANT8_ASYMM != ANEURALNETWORKS_TENSOR_QUANT8_ASYMM");

static_assert(static_cast<int32_t>(OperationType::ADD) == ANEURALNETWORKS_ADD,
              "OperationType::ADD != ANEURALNETWORKS_ADD");
static_assert(static_cast<int32_t>(OperationType::AVERAGE_POOL_2D) ==
                          ANEURALNETWORKS_AVERAGE_POOL_2D,
              "OperationType::AVERAGE_POOL_2D != ANEURALNETWORKS_AVERAGE_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::CONV_2D) == ANEURALNETWORKS_CONV_2D,
              "OperationType::CONV_2D != ANEURALNETWORKS_CONV_2D");
static_assert(static_cast<int32_t>(OperationType::DEPTHWISE_CONV_2D) ==
                          ANEURALNETWORKS_DEPTHWISE_CONV_2D,
              "OperationType::DEPTHWISE_CONV_2D != ANEURALNETWORKS_DEPTHWISE_CONV_2D");
static_assert(static_cast<int32_t>(OperationType::DEPTH_TO_SPACE) ==
                          ANEURALNETWORKS_DEPTH_TO_SPACE,
              "OperationType::DEPTH_TO_SPACE != ANEURALNETWORKS_DEPTH_TO_SPACE");
static_assert(static_cast<int32_t>(OperationType::DEQUANTIZE) == ANEURALNETWORKS_DEQUANTIZE,
              "OperationType::DEQUANTIZE != ANEURALNETWORKS_DEQUANTIZE");
static_assert(static_cast<int32_t>(OperationType::EMBEDDING_LOOKUP) ==
                          ANEURALNETWORKS_EMBEDDING_LOOKUP,
              "OperationType::EMBEDDING_LOOKUP != ANEURALNETWORKS_EMBEDDING_LOOKUP");
static_assert(static_cast<int32_t>(OperationType::FLOOR) == ANEURALNETWORKS_FLOOR,
              "OperationType::FLOOR != ANEURALNETWORKS_FLOOR");
static_assert(static_cast<int32_t>(OperationType::FULLY_CONNECTED) ==
                          ANEURALNETWORKS_FULLY_CONNECTED,
              "OperationType::FULLY_CONNECTED != ANEURALNETWORKS_FULLY_CONNECTED");
static_assert(static_cast<int32_t>(OperationType::HASHTABLE_LOOKUP) ==
                          ANEURALNETWORKS_HASHTABLE_LOOKUP,
              "OperationType::HASHTABLE_LOOKUP != ANEURALNETWORKS_HASHTABLE_LOOKUP");
static_assert(static_cast<int32_t>(OperationType::L2_NORMALIZATION) ==
                          ANEURALNETWORKS_L2_NORMALIZATION,
              "OperationType::L2_NORMALIZATION != ANEURALNETWORKS_L2_NORMALIZATION");
static_assert(static_cast<int32_t>(OperationType::L2_POOL_2D) == ANEURALNETWORKS_L2_POOL_2D,
              "OperationType::L2_POOL_2D != ANEURALNETWORKS_L2_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::LOCAL_RESPONSE_NORMALIZATION) ==
                          ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION,
              "OperationType::LOCAL_RESPONSE_NORMALIZATION != "
              "ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION");
static_assert(static_cast<int32_t>(OperationType::LOGISTIC) == ANEURALNETWORKS_LOGISTIC,
              "OperationType::LOGISTIC != ANEURALNETWORKS_LOGISTIC");
static_assert(static_cast<int32_t>(OperationType::LSH_PROJECTION) ==
                          ANEURALNETWORKS_LSH_PROJECTION,
              "OperationType::LSH_PROJECTION != ANEURALNETWORKS_LSH_PROJECTION");
static_assert(static_cast<int32_t>(OperationType::LSTM) == ANEURALNETWORKS_LSTM,
              "OperationType::LSTM != ANEURALNETWORKS_LSTM");
static_assert(static_cast<int32_t>(OperationType::MAX_POOL_2D) == ANEURALNETWORKS_MAX_POOL_2D,
              "OperationType::MAX_POOL_2D != ANEURALNETWORKS_MAX_POOL_2D");
static_assert(static_cast<int32_t>(OperationType::MUL) == ANEURALNETWORKS_MUL,
              "OperationType::MUL != ANEURALNETWORKS_MUL");
static_assert(static_cast<int32_t>(OperationType::RELU) == ANEURALNETWORKS_RELU,
              "OperationType::RELU != ANEURALNETWORKS_RELU");
static_assert(static_cast<int32_t>(OperationType::RELU1) == ANEURALNETWORKS_RELU1,
              "OperationType::RELU1 != ANEURALNETWORKS_RELU1");
static_assert(static_cast<int32_t>(OperationType::RELU6) == ANEURALNETWORKS_RELU6,
              "OperationType::RELU6 != ANEURALNETWORKS_RELU6");
static_assert(static_cast<int32_t>(OperationType::RESHAPE) == ANEURALNETWORKS_RESHAPE,
              "OperationType::RESHAPE != ANEURALNETWORKS_RESHAPE");
static_assert(static_cast<int32_t>(OperationType::RESIZE_BILINEAR) ==
                          ANEURALNETWORKS_RESIZE_BILINEAR,
              "OperationType::RESIZE_BILINEAR != ANEURALNETWORKS_RESIZE_BILINEAR");
static_assert(static_cast<int32_t>(OperationType::RNN) == ANEURALNETWORKS_RNN,
              "OperationType::RNN != ANEURALNETWORKS_RNN");
static_assert(static_cast<int32_t>(OperationType::SOFTMAX) == ANEURALNETWORKS_SOFTMAX,
              "OperationType::SOFTMAX != ANEURALNETWORKS_SOFTMAX");
static_assert(static_cast<int32_t>(OperationType::SPACE_TO_DEPTH) ==
                          ANEURALNETWORKS_SPACE_TO_DEPTH,
              "OperationType::SPACE_TO_DEPTH != ANEURALNETWORKS_SPACE_TO_DEPTH");
static_assert(static_cast<int32_t>(OperationType::SVDF) == ANEURALNETWORKS_SVDF,
              "OperationType::SVDF != ANEURALNETWORKS_SVDF");
static_assert(static_cast<int32_t>(OperationType::TANH) == ANEURALNETWORKS_TANH,
              "OperationType::TANH != ANEURALNETWORKS_TANH");

static_assert(static_cast<int32_t>(OperationType::BATCH_TO_SPACE_ND) == ANEURALNETWORKS_BATCH_TO_SPACE_ND,
              "OperationType::BATCH_TO_SPACE_ND != ANEURALNETWORKS_BATCH_TO_SPACE_ND");
static_assert(static_cast<int32_t>(OperationType::DIV) == ANEURALNETWORKS_DIV,
              "OperationType::DIV != ANEURALNETWORKS_DIV");
static_assert(static_cast<int32_t>(OperationType::MEAN) == ANEURALNETWORKS_MEAN,
              "OperationType::MEAN != ANEURALNETWORKS_MEAN");
static_assert(static_cast<int32_t>(OperationType::PAD) == ANEURALNETWORKS_PAD,
              "OperationType::PAD != ANEURALNETWORKS_PAD");
static_assert(static_cast<int32_t>(OperationType::SPACE_TO_BATCH_ND) ==
                          ANEURALNETWORKS_SPACE_TO_BATCH_ND,
              "OperationType::SPACE_TO_BATCH_ND != ANEURALNETWORKS_SPACE_TO_BATCH_ND");
static_assert(static_cast<int32_t>(OperationType::SQUEEZE) == ANEURALNETWORKS_SQUEEZE,
              "OperationType::SQUEEZE != ANEURALNETWORKS_SQUEEZE");
static_assert(static_cast<int32_t>(OperationType::STRIDED_SLICE) ==
                          ANEURALNETWORKS_STRIDED_SLICE,
              "OperationType::STRIDED_SLICE != ANEURALNETWORKS_STRIDED_SLICE");
static_assert(static_cast<int32_t>(OperationType::SUB) == ANEURALNETWORKS_SUB,
              "OperationType::SUB != ANEURALNETWORKS_SUB");
static_assert(static_cast<int32_t>(OperationType::TRANSPOSE) == ANEURALNETWORKS_TRANSPOSE,
              "OperationType::TRANSPOSE != ANEURALNETWORKS_TRANSPOSE");

static_assert(static_cast<int32_t>(FusedActivationFunc::NONE) == ANEURALNETWORKS_FUSED_NONE,
              "FusedActivationFunc::NONE != ANEURALNETWORKS_FUSED_NONE");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU) == ANEURALNETWORKS_FUSED_RELU,
              "FusedActivationFunc::RELU != ANEURALNETWORKS_FUSED_RELU");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU1) == ANEURALNETWORKS_FUSED_RELU1,
              "FusedActivationFunc::RELU1 != ANEURALNETWORKS_FUSED_RELU1");
static_assert(static_cast<int32_t>(FusedActivationFunc::RELU6) == ANEURALNETWORKS_FUSED_RELU6,
              "FusedActivationFunc::RELU6 != ANEURALNETWORKS_FUSED_RELU6");

using android::sp;
using namespace android::nn;

#ifndef HIFI_BUILD
int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory) {
    *memory = nullptr;
#ifndef HIFI_BUILD
    std::unique_ptr<MemoryFd> m = std::make_unique<MemoryFd>();
#else
    std::unique_ptr<MemoryFd> m = std::unique_ptr<MemoryFd>(new MemoryFd());
#endif //HIFI_BUILD
    if (m == nullptr) {
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    int n = m->set(size, prot, fd, offset);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m.release());
    return ANEURALNETWORKS_NO_ERROR;
}
#endif //HIFI_BUILD

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
    // No validation.  Free of nullptr is valid.
    Memory* m = reinterpret_cast<Memory*>(memory);
    delete m;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
#ifndef HIFI_BUILD
    initVLogMask();
#endif //HIFI_BUILD
    if (!model) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_create passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = new (std::nothrow) ModelBuilder();
    if (m == nullptr) {
        *model = nullptr;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    *model = reinterpret_cast<ANeuralNetworksModel*>(m);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
    // No validation.  Free of nullptr is valid.
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    delete m;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
    if (!model) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_finish passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->finish();
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
    if (!model || !type) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_addOperand passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->addOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
    if (!model || (!buffer && length != 0)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->setOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length) {
    if (!model || !memory) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_setOperandValue passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    const Memory* mem = reinterpret_cast<const Memory*>(memory);
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->setOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs) {
    if (!model || !inputs || !outputs) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksModel_addOperation passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->addOperation(type, inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
    if (!model || !inputs || !outputs) {
#ifndef HIFI_BUILD
        LOG(ERROR) << ("ANeuralNetworksModel_identifyInputsAndOutputs passed a nullptr");
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->identifyInputsAndOutputs(inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model,
                                                          bool allow) {
    if (!model) {
#ifndef HIFI_BUILD
        LOG(ERROR) << ("ANeuralNetworksModel_relaxComputationFloat32toFloat16 passed a nullptr");
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    return m->relaxComputationFloat32toFloat16(allow);
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
    if (!model || !compilation) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksCompilation_create passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    ModelBuilder* m = reinterpret_cast<ModelBuilder*>(model);
    CompilationBuilder* c = nullptr;
    int result = m->createCompilation(&c);
    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return result;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
    // No validation.  Free of nullptr is valid.
    // TODO specification says that a compilation-in-flight can be deleted
    CompilationBuilder* c = reinterpret_cast<CompilationBuilder*>(compilation);
    delete c;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
    if (!compilation) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksCompilation_setPreference passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    CompilationBuilder* c = reinterpret_cast<CompilationBuilder*>(compilation);
    return c->setPreference(preference);
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
    if (!compilation) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksCompilation_finish passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    CompilationBuilder* c = reinterpret_cast<CompilationBuilder*>(compilation);
    return c->finish();
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
    if (!compilation || !execution) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_create passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    CompilationBuilder* c = reinterpret_cast<CompilationBuilder*>(compilation);
    ExecutionBuilder* r = nullptr;
    int result = c->createExecution(&r);
    *execution = reinterpret_cast<ANeuralNetworksExecution*>(r);
    return result;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
    // TODO specification says that an execution-in-flight can be deleted
    // No validation.  Free of nullptr is valid.
    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    delete r;
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
    if (!execution || (!buffer && length != 0)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setInput passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    return r->setInput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
    if (!execution || !memory) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setInputFromMemory passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const Memory* m = reinterpret_cast<const Memory*>(memory);
    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    return r->setInputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
    if (!execution || (!buffer && length != 0)) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setOutput passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    return r->setOutput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
    if (!execution || !memory) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_setOutputFromMemory passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);
    const Memory* m = reinterpret_cast<const Memory*>(memory);
    return r->setOutputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
    if (!execution || !event) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "ANeuralNetworksExecution_startCompute passed a nullptr";
#endif //HIFI_BUILD
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    // TODO validate the rest

    ExecutionBuilder* r = reinterpret_cast<ExecutionBuilder*>(execution);

    // Dynamically allocate an sp to wrap an ExecutionCallback, seen in the NN
    // API as an abstract event object. The sp<ExecutionCallback> object is
    // returned when the execution has been successfully launched, otherwise a
    // nullptr is returned. The sp is used for ref-counting purposes. Without
    // it, the HIDL service could attempt to communicate with a dead callback
    // object.
#ifndef HIFI_BUILD
    std::unique_ptr<sp<ExecutionCallback>> e = std::make_unique<sp<ExecutionCallback>>();
    *event = nullptr;

    int n = r->startCompute(e.get());
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
    *event = reinterpret_cast<ANeuralNetworksEvent*>(e.release());
#else
    int n = r->startCompute(nullptr);
    if (n != ANEURALNETWORKS_NO_ERROR) {
        return n;
    }
#endif //HIFI_BUILD
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
#ifndef HIFI_BUILD
    if (event == nullptr) {
        LOG(ERROR) << "ANeuralNetworksEvent_wait passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    sp<ExecutionCallback>* e = reinterpret_cast<sp<ExecutionCallback>*>(event);
    (*e)->wait();
    return convertErrorStatusToResultCode((*e)->getStatus());
#else
    return ANEURALNETWORKS_NO_ERROR;
#endif //HIFI_BUILD
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
#ifndef HIFI_BUILD
    // No validation.  Free of nullptr is valid.
    if (event) {
        sp<ExecutionCallback>* e = reinterpret_cast<sp<ExecutionCallback>*>(event);
        (*e)->wait();
        delete e;
    }
#endif //HIFI_BUILD
}
