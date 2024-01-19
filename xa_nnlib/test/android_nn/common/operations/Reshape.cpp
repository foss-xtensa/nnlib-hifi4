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

// Contains the implementation of the operations.

#define LOG_TAG "Operations"

#ifndef HIFI_BUILD
#include "Operations.h"
#else
#include "xa_nnlib_ann_api.h"
#endif //HiFi_BUILD

#include "CpuOperationUtils.h"

#ifndef HIFI_BUILD
#else
#include "Utils_lib.h"
#endif //HiFi_BUILD

#ifndef HIFI_BUILD
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#else
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#endif //HiFi_BUILD

namespace android {
namespace nn {

bool reshapeGeneric(const void* inputData, const Shape& inputShape,
                    void* outputData, const Shape& outputShape) {
    size_t count = sizeOfData(inputShape.type, inputShape.dimensions);
    memcpy(outputData, inputData, count);
    return true;
}

bool resizeBilinearFloat32(const float* inputData, const Shape& inputShape,
                           float* outputData, const Shape& outputShape) {
    int32_t height = (int32_t) getSizeOfDimension(outputShape, 1);
    int32_t width  = (int32_t) getSizeOfDimension(outputShape, 2);

    int32_t outDimData[2] = {height, width};
    // We have to fake a tensor here, to satisfy ResizeBilinear().
    Shape outDimShape;
    outDimShape.dimensions = {1, 1, 1, 2};

#ifndef HIFI_BUILD
    tflite::reference_ops::ResizeBilinear(
            inputData, convertShapeToDims(inputShape),
            outDimData, convertShapeToDims(outDimShape),
            outputData, convertShapeToDims(outputShape));
#else
    tflite::reference_ops::ResizeBilinear(
            inputData, convertShapeToDims(inputShape),
            outDimData, convertShapeToDims(outDimShape),
            outputData, convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    return true;
}

bool depthToSpaceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         int32_t blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
       tflite::reference_ops::DepthToSpace(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#else
       tflite::reference_ops::DepthToSpace(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_BUILD
        tflite::reference_ops::DepthToSpace(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::DepthToSpace(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

bool spaceToDepthGeneric(const uint8_t* inputData, const Shape& inputShape,
                         int32_t blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
        tflite::reference_ops::SpaceToDepth(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::SpaceToDepth(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_BUILD
        tflite::reference_ops::SpaceToDepth(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::SpaceToDepth(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

bool padGeneric(const uint8_t* inputData, const Shape& inputShape,
                const int32_t* paddings,
                uint8_t* outputData, const Shape& outputShape) {
    int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(inputShape));

    std::vector<int> beforePadding;
    std::vector<int> afterPadding;
    // The lower level implementation expects the paddings in the reverse order.
    for (int32_t i = numInputDims - 1; i >= 0; --i) {
        beforePadding.push_back(paddings[i * 2]);
        afterPadding.push_back(paddings[i * 2 + 1]);
    }

    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
        tflite::reference_ops::Pad(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                beforePadding, afterPadding,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::Pad(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                beforePadding, afterPadding,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_BUILD
        tflite::reference_ops::Pad(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                beforePadding, afterPadding,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::Pad(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                beforePadding, afterPadding,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

bool batchToSpaceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    // Needed by low level implementation, but not really used.
    tflite::Dims<4> blockSizeDim;
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
       tflite::reference_ops::BatchToSpaceND(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#else
       tflite::reference_ops::BatchToSpaceND(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_BUILD
        tflite::reference_ops::BatchToSpaceND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::BatchToSpaceND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

bool spaceToBatchGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* blockSize,
                         const int32_t* padding, const Shape& paddingShape,
                         uint8_t* outputData, const Shape& outputShape) {
    // Needed by low level implementation, but not really used.
    tflite::Dims<4> blockSizeDim;
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
#ifndef HIFI_BUILD
        tflite::reference_ops::SpaceToBatchND(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::SpaceToBatchND(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
#ifndef HIFI_BUILD
        tflite::reference_ops::SpaceToBatchND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#else
        tflite::reference_ops::SpaceToBatchND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
#endif //HiFi_BUILD
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}

bool squeezeGeneric(const void* inputData, const Shape& inputShape,
                    void* outputData, const Shape& outputShape) {
    size_t count = sizeOfData(inputShape.type, inputShape.dimensions);
    memcpy(outputData, inputData, count);
    return true;
}

bool transposeGeneric(const uint8_t* inputData, const Shape& inputShape,
                      const int32_t* perm, const Shape& permShape,
                      uint8_t* outputData, const Shape& outputShape) {
    // Reverse the permuted axes and convert to 4D due to the way Dims are
    // constructed.
    const int32_t kOutputDimensionNum = 4;

    int32_t permSize = static_cast<int32_t>(getSizeOfDimension(permShape, 0));
    int32_t reversed_perm[kOutputDimensionNum];
    for (int32_t output_k = 0, input_k = permSize - 1; output_k < permSize;
             ++output_k, --input_k) {
        reversed_perm[output_k] = permSize - perm[input_k] - 1;
    }
    for (int32_t k = permSize; k < kOutputDimensionNum; ++k) {
        reversed_perm[k] = k;
    }
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::reference_ops::Transpose(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape),
                reversed_perm);
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::reference_ops::Transpose(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape),
                reversed_perm);
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Unsupported data type";
#endif //HIFI_BUILD
        return false;
    }
    return true;
}
} // namespace nn
} // namespace android
