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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_TYPES_H

#include <hidl/HidlSupport.h>
#ifndef HIFI_BUILD
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>
#endif //HIFI_BUILD
#if 0 //ppn
#else
#include <cstdint>
#include <cstddef>
#include <vector>
#endif 

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {

// Forward declaration for forward reference support:
enum class OperandType : int32_t;
enum class OperationType : int32_t;
enum class FusedActivationFunc : int32_t;
enum class OperandLifeTime : int32_t;
enum class DeviceStatus : int32_t;
struct PerformanceInfo;
struct Capabilities;
struct DataLocation;
struct Operand;
struct Operation;
struct Model;
struct RequestArgument;
struct Request;
enum class ErrorStatus : int32_t;

/**
 * Operand types.
 * 
 * The type of an operand in a model.
 * 
 * Types prefaced with TENSOR_* must be used for tensor data (i.e., tensors
 * with at least one dimension). Types not prefaced by TENSOR_* represent
 * scalar values and must have no dimensions.
 * 
 * Although many types are defined, most operators accept just a few
 * types. Most used are {@link OperandType::TENSOR_FLOAT32},
 * {@link OperandType::TENSOR_QUANT8_ASYMM},
 * and {@link OperandType::INT32}.
 */
enum class OperandType : int32_t {
    /**
     * A 32 bit floating point scalar value.  */
    FLOAT32 = 0,
    /**
     * A signed 32 bit integer scalar value.  */
    INT32 = 1,
    /**
     * An unsigned 32 bit integer scalar value.  */
    UINT32 = 2,
    /**
     * A tensor of 32 bit floating point values.  */
    TENSOR_FLOAT32 = 3,
    /**
     * A tensor of 32 bit integer values.  */
    TENSOR_INT32 = 4,
    /**
     * A tensor of 8 bit integers that represent real numbers.
     * 
     * Attached to this tensor are two numbers that can be used to convert the
     * 8 bit integer to the real value and vice versa. These two numbers are:
     * - scale: a 32 bit floating point value greater than zero.
     * - zeroPoint: a 32 bit integer, in range [0, 255].
     * 
     * The formula is:
     * real_value = (integer_value - zeroPoint) * scale.
     */
    TENSOR_QUANT8_ASYMM = 5,
    /**
     * OEM specific scalar value.  */
    OEM = 10000,
    /**
     * A tensor of OEM specific values.  */
    TENSOR_OEM_BYTE = 10001,
};

/**
 * Operation types.
 * 
 * The type of an operation in a model.
 */
enum class OperationType : int32_t {
    /**
     * Adds two tensors, element-wise.
     * 
     * Takes two input tensors of identical {@link OperandType} and compatible
     * dimensions. The output is the sum of both input tensors, optionally
     * modified by an activation function.
     * 
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     * 
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its
     * way forward.
     * 
     * Example:
     * 
     *     input1.dimension = {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandType}, and compatible dimensions
     *      as input0.
     * * 2: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The sum, a tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 27.
     */
    ADD = 0,
    /**
     * Performs a 2-D average pooling operation.
     * 
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[b, i, j, channel] =
     *         sum_{di, dj}(
     *             input[b, strides[1] * i + di, strides[2] * j + dj, channel]
     *         ) / sum(1)
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" (i.e., Num_samples, Height, Width,
     * and Channels) data layout.
     * 
     * Both explicit padding and implicit padding are supported.
     * 
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
     * * 2: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     * 
     * Available since API level 27.
     */
    AVERAGE_POOL_2D = 1,
    /**
     * Concatenates the input tensors along the given dimension.
     * 
     * The input tensors must have identical {@link OperandType} and the same
     * dimensions except the dimension along the concatenation axis.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0 ~ n-1: The list of n input tensors, of shape
     *            [D0, D1, ..., Daxis(i), ..., Dm]. For inputs of
     *            {@link OperandType::TENSOR_QUANT8_ASYMM}, all input tensors
     *            must have the same scale and zeroPoint.
     * * n: An {@link OperandType::INT32} scalar, specifying the
     *      concatenation axis.
     * 
     * Outputs:
     * * 0: The output, a tensor of the same {@link OperandType} as the input
     *      tensors. The output shape is [D0, D1, ..., sum(Daxis(i)), ..., Dm].
     * 
     * Available since API level 27.
     */
    CONCATENATION = 2,
    /**
     * Performs an 2-D convolution operation.
     * 
     * The CONV_2D op sweeps a 2-D filter that can mix channels together over a
     * batch of images, applying the filter to each window of each image of the
     * appropriate size.
     * 
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[b, i, j, channel] =
     *         sum_{di, dj, k} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
     *             filter[channel, di, dj, k]
     *         ) + bias[channel]
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Both explicit padding and implicit padding are supported.
     * 
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
     *      For input tensor of {@link OperandType::TENSOR_FLOAT32}, the bias
     *      should also be of {@link OperandType::TENSOR_FLOAT32}. For input
     *      tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias
     *      should be of {@link OperandType::TENSOR_INT32}, with zeroPoint of
     *      0 and bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape
     *      [depth_out, filter_height, filter_width, depth_in], specifying the
     *      filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
     *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
     *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
     * * 4: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link OperandType::TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale.
     * 
     * Available since API level 27.
     */
    CONV_2D = 3,
    /**
     * Performs a depthwise 2-D convolution operation.
     * 
     * Given an input tensor of shape [batches, height, width, depth_in] and a
     * filter tensor of shape [1, filter_height, filter_width, depth_out]
     * containing depth_out convolutional filters of depth 1, DEPTHWISE_CONV
     * applies a different filter to each input channel (expanding from 1
     * channel to channel_multiplier channels for each), then concatenates the
     * results together.
     * 
     * The output has depth_out = depth_in * depth_multiplier channels.
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[b, i, j, k * channel_multiplier + q] =
     *         sum_{di, dj} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
     *             filter[1, di, dj, k * channel_multiplier + q]
     *         ) + bias[k * channel_multiplier + q]
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Both explicit padding and implicit padding are supported.
     * 
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
     *      specifying the filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
     *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
     *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 7: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 8: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 9: An {@link OperandType::INT32} scalar, specifying the depthwise
     *      multiplier.
     * * 10: An {@link OperandType::INT32} scalar, and has to be one of the
     *       {@link FusedActivationFunc} values. Specifies the activation to
     *       invoke on the result.
     * 
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
     *      specifying the filter.
     * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
     *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
     *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
     *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
     * * 4: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the depthwise
     *      multiplier.
     * * 7: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth_out]. For output tensor of
     *      {@link OperandType::TENSOR_QUANT8_ASYMM}, the following condition
     *      must be satisfied: output_scale > input_scale * filter_scale.
     * 
     * Available since API level 27.
     */
    DEPTHWISE_CONV_2D = 4,
    /**
     * Rearranges data from depth into blocks of spatial data.
     * 
     * More specifically, this op outputs a copy of the input tensor where
     * values from the depth dimension are moved in spatial blocks to the height
     * and width dimensions. The value block_size indicates the input block size
     * and how the data is moved.
     * 
     * Chunks of data of size block_size * block_size from depth are rearranged
     * into non-overlapping blocks of size block_size x block_size.
     * 
     * The width of the output tensor is input_depth * block_size, whereas the
     * height is input_height * block_size. The depth of the input tensor must
     * be divisible by block_size * block_size
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the block_size.
     *      block_size must be >=1 and block_size * block_size must be a divisor
     *      of the input depth.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batch, height*block_size,
     *      width*block_size, depth/(block_size*block_size)].
     * 
     * Available since API level 27.
     */
    DEPTH_TO_SPACE = 5,
    /**
     * Dequantizes the input tensor.
     * 
     * The formula is:
     * 
     *     output = (input - zeroPoint) * scale.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: A tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0, but with
     *      {@link OperandType::TENSOR_FLOAT32}.
     * 
     * Available since API level 27.
     */
    DEQUANTIZE = 6,
    /**
     * Looks up sub-tensors in the input tensor.
     * 
     * This operator takes for input a tensor of values (Values) and
     * a one-dimensional tensor of selection indices (Lookups).
     * The output tensor is the concatenation of sub-tensors of Values as
     * selected by Lookups.
     * 
     * Think of Values as being sliced along its first dimension:
     * The entries in Lookups select which slices are concatenated together
     * to create the output tensor.
     * 
     * For example, if Values has shape of [40, 200, 300] and
     * Lookups has shape of [3], all three values found in Lookups are
     * expected to be between 0 and 39. The resulting tensor must
     * have shape of [3, 200, 300].
     * 
     * If a value in Lookups is out of bounds, the operation must fail
     * and an error must be reported.
     * 
     * Inputs:
     * * 0: Lookups. A 1-D tensor of {@link OperandType::TENSOR_INT32}.
     *      The values are indices into the first dimension of Values.
     * * 1: Values. An n-D tensor, where n >= 2, from which sub-tensors are
     *      extracted.
     * 
     * Output:
     * * 0: A n-D tensor with the same rank and shape as the Values
     *      tensor, except for the first dimension which has the same size
     *      as Lookups' only dimension.
     * 
     * Available since API level 27.
     */
    EMBEDDING_LOOKUP = 7,
    /**
     * Computes element-wise floor() on the input tensor.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: A tensor.
     * 
     * Outputs:
     * * 0: The output tensor, of the same {@link OperandType} and dimensions as
     *      the input tensor.
     * 
     * Available since API level 27.
     */
    FLOOR = 8,
    /**
     * Denotes a fully (densely) connected layer, which connects all elements
     * in the input tensor with each element in the output tensor.
     * 
     * This layer implements the operation:
     * 
     *     outputs = activation(inputs * weights’ + bias)
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor of at least rank 2, specifying the input. If rank is
     *      greater than 2, then it gets flattened to a 2-D Tensor. The
     *      (flattened) 2-D Tensor is reshaped (if necessary) to
     *      [batch_size, input_size], where "input_size" corresponds to the
     *      number of inputs to the layer, matching the second dimension of
     *      weights, and "batch_size" is calculated by dividing the number of
     *      elements by "input_size".
     * * 1: A 2-D tensor, specifying the weights, of shape
     *      [num_units, input_size], where "num_units" corresponds to the number
     *      of output nodes.
     * * 2: A 1-D tensor, of shape [num_units], specifying the bias. For input
     *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
     *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
     *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
     *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
     *      bias_scale == input_scale * filter_scale.
     * * 3: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output tensor, of shape [batch_size, num_units]. For output
     *      tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}, the following
     *      condition must be satisfied:
     *      output_scale > input_scale * filter_scale.
     * 
     * Available since API level 27.
     */
    FULLY_CONNECTED = 9,
    /**
     * Looks up sub-tensors in the input tensor using a key-value map.
     * 
     * This operator takes for input a tensor of values (Values),
     * a one-dimensional tensor of selection values (Lookups) and
     * a one-dimensional tensor that maps these values to Values
     * indexes. The output tensor is the concatenation of sub-tensors of
     * Values as selected by Lookups via Keys.
     * 
     * Think of Values as being sliced along its outer-most dimension.
     * The output is a concatenation of selected slices, with one slice
     * for each entry of Lookups. The slice selected is the one at the
     * same index as the Maps entry that matches the value in Lookups.
     * 
     * For a hit, the corresponding sub-tensor of Values is included
     * in the Output tensor. For a miss, the corresponding sub-tensor in
     * Output must have zero values.
     * 
     * For example, if Values has shape of [40, 200, 300],
     * Keys should have a shape of [40]. If Lookups tensor has shape
     * of [3], three slices are being concatenated, so the resulting tensor
     * must have the shape of [3, 200, 300]. If the first entry in Lookups
     * has the value 123456, that value must be located in Keys tensor.
     * If the sixth entry of Keys contains 123456, the sixth slice of Values
     * must be selected. If no entry in Keys has 123456, a slice of zeroes
     * must be concatenated.
     * 
     * Inputs:
     * * 0: Lookups. A 1-D {@link OperandType::TENSOR_INT32} tensor with
     *      shape [ k ].
     * * 1: Keys. A 1-D {@link OperandType::TENSOR_INT32} tensor with shape
     *      [ n ]; Keys and Values pair represent a map, i.e., the ith element
     *      in Keys (Keys[i]) is the key to select the ith sub-tensor in Values
     *      (Values[i]), where 0 <= i <= n-1. Keys tensor *MUST* be sorted in
     *      ascending order.
     * * 2: Values. A tensor with shape of [ n, … ]; i.e., the first dimension
     *      must be n.
     * 
     * Outputs:
     * * 0: Output. A tensor with shape [ k …].
     * * 1: Hits. A boolean tensor with shape [ k ] indicates whether the lookup
     *      hits (True) or not (False).
     *      Stored as {@link OperandType::TENSOR_QUANT8_ASYMM} with offset 0
     *      and scale 1.0f.
     *      A non-zero byte represents True, a hit. A zero indicates otherwise.
     * 
     * Available since API level 27.
     */
    HASHTABLE_LOOKUP = 10,
    /**
     * Applies L2 normalization along the depth dimension.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[batch, row, col, channel] =
     *         input[batch, row, col, channel] /
     *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
     * 
     * For input tensor with more dimensions, independently normalizes each 1-D
     * slice along dimension dim.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout (i.e., Num_samples,
     * Height, Width, and Channels).
     * 
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth].
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of the same shape as input
     *      [batches, height, width, depth].
     * 
     * Available since API level 27.
     */
    L2_NORMALIZATION = 11,
    /**
     * Performs an 2-D L2 pooling operation.
     * 
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[b, i, j, c] =
     *         sqrt(sum_{di, dj} pow(input[b, strides[1] * i + di, strides[2] * j + dj, c], 2) /
     *              sum(1))
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Both explicit padding and implicit padding are supported.
     * 
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
     * * 2: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     * 
     * Available since API level 27.
     */
    L2_POOL_2D = 12,
    /**
     * Applies Local Response Normalization along the depth dimension.
     * 
     * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the
     * last dimension), and each vector is normalized independently. Within a
     * given vector, each component is divided by the weighted, squared sum of
     * inputs within depth_radius.
     * 
     * The output is calculated using this formula:
     * 
     *     sqr_sum[a, b, c, d] = sum(
     *         pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2))
     *     output = input / pow((bias + alpha * sqr_sum), beta)
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the radius of
     *      the normalization window.
     * * 2: An {@link OperandType::FLOAT32} scalar, specifying the bias, must
     *      not be zero.
     * * 3: An {@link OperandType::FLOAT32} scalar, specifying the scale
     *      factor, alpha.
     * * 4: An {@link OperandType::FLOAT32} scalar, specifying the exponent,
     *      beta.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     * 
     * Available since API level 27.
     */
    LOCAL_RESPONSE_NORMALIZATION = 13,
    /**
     * Computes sigmoid activation on the input tensor element-wise.
     * 
     * The output is calculated using this formula:
     * 
     *     output = 1 / (1 + exp(-input))
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link OperandType::TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 256 and the zeroPoint must be 0.
     * 
     * Available since API level 27.
     */
    LOGISTIC = 14,
    /**
     * Projects an input to a bit vector via locality senstive hashing.
     * 
     * Inputs:
     * * 0: Hash functions. Dim.size == 2, DataType: Float.
     *            Tensor[0].Dim[0]: Number of hash functions.
     *            Tensor[0].Dim[1]: Number of seeds per hash functions.
     *            Tensor[0].Dim[1] <= 32 in sparse case.
     * 
     * * 1: Input. Dim.size >= 1, no restriction on DataType.
     * * 2: Weight. Optional. Dim.size == 1, DataType: Float.
     *     If not set, each input element is considered to have the same weight
     *     of 1.0.
     *     Tensor[1].Dim[0] == Tensor[2].Dim[0]
     * * 3: Type:
     *        Sparse: Value LSHProjectionType_SPARSE(=1).
     *          Computed bit vector is considered to be sparse.
     *          Each output element is an int32 made up of multiple bits
     *          computed from hash functions.
     * 
     *        Dense: Value LSHProjectionType_DENSE(=2).
     *          Computed bit vector is considered to be dense. Each output
     *          element represents a bit and can take the value of either
     *          0 or 1.
     * 
     * Outputs:
     * * 0: If the projection type is sparse:
     *        Output.Dim == { Tensor[0].Dim[0] }
     *        A tensor of int32 that represents hash signatures.
     *      If the projection type is Dense:
     *        Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
     *        A flattened tensor that represents projected bit vectors.
     * 
     * Available since API level 27.
     */
    LSH_PROJECTION = 15,
    /**
     * Performs a single time step in a Long Short-Term Memory (LSTM) layer
     * 
     * The LSTM operation is described by the following equations.
     * 
     * \f{eqnarray*}{
     * i_t =& \sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) & \\
     * f_t =& \sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) & \\
     * C_t =& clip(f_t \odot C_{t-1} + i_t \odot
     *        g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) & \\
     * o_t =& \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o) & \\
     *      & & \\
     *      & clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})
     *      & if\ there\ is\ a\ projection; \\
     * h_t =& & \\
     *      & o_t \odot g(C_t) & otherwise. \\
     * \f}
     * Where:
     * * \f$x_t\f$ is the input,
     * * \f$i_t\f$ is the input gate,
     * * \f$f_t\f$ is the forget gate,
     * * \f$C_t\f$ is the cell state,
     * * \f$o_t\f$ is the output,
     * * \f$h_t\f$ is the output state,
     * * \f$\sigma\f$ is the logistic sigmoid function,
     * * \f$g\f$ is the cell input and cell output activation function, usually
     *   \f$tahn\f$,
     * * \f$W_{xi}\f$ is the input-to-input weight matrix,
     * * \f$W_{hi}\f$ is the recurrent to input weight matrix,
     * * \f$W_{ci}\f$ is the cell-to-input weight matrix,
     * * \f$b_i\f$ is the input gate bias,
     * * \f$W_{xf}\f$ is the input-to-forget weight matrix,
     * * \f$W_{hf}\f$ is the recurrent-to-forget weight matrix,
     * * \f$W_{cf}\f$ is the cell-to-forget weight matrix,
     * * \f$b_f\f$ is the forget gate bias,
     * * \f$W_{xc}\f$ is the input-to-cell weight matrix,
     * * \f$W_{hc}\f$ is the recurrent-to-cell weight matrix,
     * * \f$b_c\f$ is the cell bias,
     * * \f$W_{xo}\f$ is the input-to-output weight matrix,
     * * \f$W_{ho}\f$ is the recurrent-to-output weight matrix,
     * * \f$W_{co}\f$ is the cell-to-output weight matrix,
     * * \f$b_o\f$ is the output gate bias,
     * * \f$W_{proj}\f$ is the projection weight matrix,
     * * \f$b_{proj}\f$ is the projection bias,
     * * \f$t_{cell}\f$ is the threshold for clipping the cell state, and
     * * \f$t_{proj}\f$ is the threshold for clipping the projected output.
     * * \f$\odot\f$ is the
     *   <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)">
     *   Hadamard product</a> that takes two matrices and produces another
     *   matrix, each element of which is the product of the corresponding
     *   elements of the input matrices.
     * 
     * The operation has the following independently optional inputs:
     * * The input-to-input weights (\f$W_{xi}\f$), recurrent-to-input weights
     *   (\f$W_{hi}\f$), cell-to-input (\f$W_{ci}\f$) weights, and input gate
     *   bias (\f$b_i\f$) either all have values, or none of them have values
     *   (i.e., all set to null). If they have no values, coupling of input and
     *   forget gates (CIFG) is used, in which case the input gate (\f$i_t\f$)
     *   is calculated using the following equation instead.
     *   \f{eqnarray*}{
     *   i_t = 1 - f_t
     *   \f}
     * * The cell-to-forget weights (\f$W_{cf}\f$) and cell-to-output weights
     *   (\f$W_{co}\f$) either both have values or neither of them have values.
     *   If they have values, the peephole optimization is used. Additionally,
     *   if CIFG is not used, cell-to-input weights (\f$W_{ci}\f$) is also
     *   required to have values for peephole optimization.
     * * The projection weights (\f$W_{proj}\f$) is required only for the
     *   recurrent projection layer, and should otherwise have no value.
     * * The projection bias (\f$b_{proj}\f$) may (but not required to) have a
     *   value if the recurrent projection layer exists, and should otherwise
     *   have no value.
     * 
     * References:
     * 
     * The default non-peephole non-CIFG implementation is based on:
     * http://www.bioinf.jku.at/publications/older/2604.pdf
     * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
     * Computation, 9(8):1735-1780, 1997.
     * 
     * The peephole implementation and projection layer is based on:
     * https://research.google.com/pubs/archive/43905.pdf
     * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
     * recurrent neural network architectures for large scale acoustic
     * modeling." INTERSPEECH, 2014.
     * (However, the concept of peephole optimization was introduced in work
     * prior to this paper.)
     * 
     * The coupling of input and forget gate (CIFG) is based on:
     * http://arxiv.org/pdf/1503.04069.pdf
     * Greff et al. "LSTM: A Search Space Odyssey"
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Inputs:
     * * 0: The input (\f$x_t\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, input_size], where “batch_size” corresponds to the
     *      batching dimension, and “input_size” is the size of the input.
     * * 1: The input-to-input weights (\f$W_{xi}\f$). Optional.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size], where “num_units” corresponds to the
     *      number of cell units.
     * * 2: The input-to-forget weights (\f$W_{xf}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size].
     * * 3: The input-to-cell weights (\f$W_{xc}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size].
     * * 4: The input-to-output weights (\f$W_{xo}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size].
     * * 5: The recurrent-to-input weights (\f$W_{hi}\f$). Optional.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, output_size], where “output_size” corresponds to either
     *      the number of cell units (i.e., “num_units”), or the second
     *      dimension of the “projection_weights”, if defined.
     * * 6: The recurrent-to-forget weights (\f$W_{hf}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, output_size].
     * * 7: The recurrent-to-cell weights (\f$W_{hc}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, output_size].
     * * 8: The recurrent-to-output weights (\f$W_{ho}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, output_size].
     * * 9: The cell-to-input weights (\f$W_{ci}\f$). Optional.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 10:The cell-to-forget weights (\f$W_{cf}\f$). Optional.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 11:The cell-to-output weights (\f$W_{co}\f$). Optional.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 12:The input gate bias (\f$b_i\f$). Optional.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 13:The forget gate bias (\f$b_f\f$).
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 14:The cell bias (\f$b_c\f$).
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 15:The output gate bias (\f$b_o\f$).
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 16:The projection weights (\f$W_{proj}\f$). Optional.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [output_size, num_units].
     * * 17:The projection bias (\f$b_{proj}\f$). Optional.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [output_size].
     * * 18:The output state (in) (\f$h_{t-1}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, output_size].
     * * 19:The cell state (in) (\f$C_{t-1}\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units].
     * * 20:The activation function (\f$g\f$).
     *      A value indicating the activation function:
     *      <ul>
     *      <li>0: None;
     *      <li>1: Relu;
     *      <li>3: Relu6;
     *      <li>4: Tanh;
     *      <li>6: Sigmoid.
     *      </ul>
     * * 21:The clipping threshold (\f$t_{cell}\f$) for the cell state, such
     *      that values are bound within [-cell_clip, cell_clip]. If set to 0.0
     *      then clipping is disabled.
     * * 22:The clipping threshold (\f$t_{proj}\f$) for the output from the
     *      projection layer, such that values are bound within
     *      [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     * 
     * Outputs:
     * * 0: The scratch buffer.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units * 4] with CIFG, or
     *      [batch_size, num_units * 3] without CIFG.
     * * 1: The output state (out) (\f$h_t\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, output_size].
     * * 2: The cell state (out) (\f$C_t\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units].
     * * 3: The output (\f$o_t\f$).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, output_size]. This is effectively the same as the
     *      current “output state (out)” value.
     * 
     * Available since API level 27.
     */
    LSTM = 16,
    /**
     * Performs an 2-D max pooling operation.
     * 
     * The output dimensions are functions of the filter dimensions, stride, and
     * padding.
     * 
     * The values in the output tensor are computed as:
     * 
     *     output[b, i, j, channel] =
     *         max_{di, dj} (
     *             input[b, strides[1] * i + di, strides[2] * j + dj, channel]
     *         )
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Both explicit padding and implicit padding are supported.
     * 
     * Inputs (explicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the left, in the ‘width’ dimension.
     * * 2: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the right, in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the top, in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
     *      the bottom, in the ‘height’ dimension.
     * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 6: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 7: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 8: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Inputs (implicit padding):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the implicit
     *      padding scheme, has to be one of the
     *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
     * * 2: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘width’ dimension.
     * * 3: An {@link OperandType::INT32} scalar, specifying the stride when
     *      walking through input in the ‘height’ dimension.
     * * 4: An {@link OperandType::INT32} scalar, specifying the filter
     *      width.
     * * 5: An {@link OperandType::INT32} scalar, specifying the filter
     *      height.
     * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, out_height, out_width, depth].
     * 
     * Available since API level 27.
     */
    MAX_POOL_2D = 17,
    /**
     * Multiplies two tensors, element-wise.
     * 
     * Takes two input tensors of identical {@link OperandType} and compatible
     * dimensions. The output is the product of both input tensors, optionally
     * modified by an activation function.
     * 
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     * 
     * The size of the resulting output is the maximum size along each dimension
     * of the input operands. It starts with the trailing dimensions, and works
     * its way forward.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: A tensor.
     * * 1: A tensor of the same {@link OperandType}, and compatible dimensions
     *      as input0.
     * * 2: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: The product, a tensor of the same {@link OperandType} as input0.
     *      For output tensor of {@link OperandType::TENSOR_QUANT8_ASYMM},
     *      the following condition must be satisfied:
     *      output_scale > input1_scale * input2_scale.
     * 
     * Available since API level 27.
     */
    MUL = 18,
    /**
     * Computes rectified linear activation on the input tensor element-wise.
     * 
     * The output is calculated using this formula:
     * 
     *     output = max(0, input)
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     * 
     * Available since API level 27.
     */
    RELU = 19,
    /**
     * Computes rectified linear 1 activation on the input tensor element-wise.
     * 
     * The output is calculated using this formula:
     * 
     *     output = min(1.f, max(-1.f, input))
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     * 
     * Available since API level 27.
     */
    RELU1 = 20,
    /**
     * Computes rectified linear 6 activation on the input tensor element-wise.
     * 
     * The output is calculated using this formula:
     * 
     *     output = min(6, max(0, input))
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     * 
     * Available since API level 27.
     */
    RELU6 = 21,
    /**
     * Reshapes a tensor.
     * 
     * Given tensor, this operation returns a tensor that has the same values as
     * tensor, but with a newly specified shape.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the tensor to be reshaped.
     * * 1: A 1-D tensor of {@link OperandType::TENSOR_INT32}, defining the
     *      shape of the output tensor. The number of elements implied by shape
     *      must be the same as the number of elements in the input tensor.
     * 
     * Outputs:
     * * 0: The output tensor, of shape specified by the input shape.
     * 
     * Available since API level 27.
     */
    RESHAPE = 22,
    /**
     * Resizes images to given size using the bilinear interpretation.
     * 
     * Resized images must be distorted if their output aspect ratio is not the
     * same as input aspect ratio. The corner pixels of output may not be the
     * same as corner pixels of input.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the output
     *      height of the output tensor.
     * * 2: An {@link OperandType::INT32} scalar, specifying the output
     *      width of the output tensor.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, new_height, new_width, depth].
     * 
     * Available since API level 27.
     */
    RESIZE_BILINEAR = 23,
    /**
     * A basic recurrent neural network layer.
     * 
     * This layer implements the operation:
     * outputs = state = activation(inputs * input_weights +
     *                              state * recurrent_weights + bias)
     * 
     * Where:
     * * “input_weights” is a weight matrix that multiplies the inputs;
     * * “recurrent_weights” is a weight matrix that multiplies the current
     *    “state” which itself is the output from the previous time step
     *    computation;
     * * “bias” is a bias vector (added to each output vector in the batch);
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Inputs:
     * * 0: input.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32} of shape
     *      [batch_size, input_size], where “batch_size” corresponds to the
     *      batching dimension, and “input_size” is the size of the input.
     * * 1: weights.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size], where “num_units” corresponds to the
     *      number of units.
     * * 2: recurrent_weights.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, num_units], with columns corresponding to the weights
     *      from each unit.
     * * 3: bias.
     *      A 1-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units].
     * * 4: hidden state (in).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units].
     * * 5: fused_activation_function.
     *      An optional {@link FusedActivationFunc} value indicating the
     *      activation function. If “NONE” is specified then it results in a
     *      linear activation.
     * 
     * Outputs:
     * * 0: hidden state (out).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units].
     * 
     * * 1: output.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units]. This is effectively the same as the
     *      current state value.
     * 
     * Available since API level 27.
     */
    RNN = 24,
    /**
     * Computes the softmax activation on the input tensor element-wise, per
     * batch, by normalizing the input vector so the maximum coefficient is
     * zero.
     * 
     * The output is calculated using this formula:
     * 
     *     output[batch, i] =
     *         exp((input[batch, i] - max(input[batch, :])) * beta) /
     *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 2 or 4.
     * 
     * Inputs:
     * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
     * * 1: An {@link OperandType::FLOAT32} scalar, specifying the positive
     *      scaling factor for the exponent, beta.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link OperandType::TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 256 and the zeroPoint must be 0.
     * 
     * Available since API level 27.
     */
    SOFTMAX = 25,
    /**
     * Rearranges blocks of spatial data, into depth.
     * 
     * More specifically, this op outputs a copy of the input tensor where
     * values from the height and width dimensions are moved to the depth
     * dimension. The value block_size indicates the input block size and how
     * the data is moved.
     * 
     * Chunks of data of size block_size * block_size from depth are rearranged
     * into non-overlapping blocks of size block_size x block_size.
     * 
     * The depth of the output tensor is input_depth * block_size * block_size.
     * The input tensor's height and width must be divisible by block_size.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4, with "NHWC" data layout.
     * 
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: An {@link OperandType::INT32} scalar, specifying the block_size.
     *      block_size must be >=1 and block_size must be a divisor of both the
     *      input height and width.
     * 
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, height/block_size,
     *      width/block_size, depth_in*block_size*block_size].
     * 
     * Available since API level 27.
     */
    SPACE_TO_DEPTH = 26,
    /**
     * SVDF op is a kind of stateful layer derived from the notion that a
     * densely connected layer that's processing a sequence of input frames can
     * be approximated by using a singular value decomposition of each of its
     * nodes. The implementation is based on:
     * 
     * https://research.google.com/pubs/archive/43813.pdf
     * 
     * P. Nakkiran, R. Alvarez, R. Prabhavalkar, C. Parada.
     * “Compressing Deep Neural Networks using a Rank-Constrained Topology”.
     * INTERSPEECH, 2015.
     * 
     * It processes the incoming input using a 2-stage filtering mechanism:
     * * stage 1 performs filtering on the "features" dimension, whose outputs
     *   get pushed into a memory of fixed-size memory_size.
     * * stage 2 performs filtering on the "time" dimension of the memory_size
     *   memoized outputs of stage 1.
     * 
     * Specifically, for rank 1, this layer implements the operation:
     * 
     *     memory = push(conv1d(inputs, weights_feature, feature_dim,
     *                          "PADDING_VALID"));
     *     outputs = activation(memory * weights_time + bias);
     * 
     * Where:
     * * “weights_feature” is a weights matrix that processes the inputs (by
     *   convolving the input with every “feature filter”), and whose outputs
     *   get pushed, stacked in order, into the fixed-size “memory” (the oldest
     *   entry gets dropped);
     * * “weights_time” is a weights matrix that processes the “memory” (by a
     *   batched matrix multiplication on the num_units);
     * * “bias” is an optional bias vector (added to each output vector in the
     *   batch); and
     * * “activation” is the function passed as the “fused_activation_function”
     *   argument (if not “NONE”).
     * 
     * Each rank adds a dimension to the weights matrices by means of stacking
     * the filters.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Inputs:
     * * 0: input.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, input_size], where “batch_size” corresponds to the
     *      batching dimension, and “input_size” is the size of the input.
     * * 1: weights_feature.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, input_size], where “num_units” corresponds to the
     *      number of units.
     * * 2: weights_time.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [num_units, memory_size], where “memory_size” corresponds to the
     *      fixed-size of the memory.
     * * 3: bias.
     *      An optional 1-D tensor of {@link OperandType::TENSOR_FLOAT32},
     *      of shape [num_units].
     * * 4: state (in).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, (memory_size - 1) * num_units * rank].
     * * 5: rank.
     *      The rank of the SVD approximation.
     * * 6: fused_activation_function.
     *      An optional {@link FusedActivationFunc} value indicating the
     *      activation function. If “NONE” is specified then it results in a
     *      linear activation.
     * 
     * Outputs:
     * * 0: state (out).
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, (memory_size - 1) * num_units * rank].
     * * 1: output.
     *      A 2-D tensor of {@link OperandType::TENSOR_FLOAT32}, of shape
     *      [batch_size, num_units].
     * 
     * Available since API level 27.
     */
    SVDF = 27,
    /**
     * Computes hyperbolic tangent of input tensor element-wise.
     * 
     * The output is calculated using this formula:
     * 
     *     output = tanh(input)
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: up to 4.
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * 
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     * 
     * Available since API level 27.
     */
    TANH = 28,
    /**
     * OEM specific operation.
     * 
     * This operation is OEM specific. It should only be used for OEM
     * applications.
     */
    OEM_OPERATION = 10000,
};

/**
 * Fused activation function types.
 */
enum class FusedActivationFunc : int32_t {
    NONE = 0,
    RELU = 1,
    RELU1 = 2,
    RELU6 = 3,
};

/**
 * How an operand is used.
 */
enum class OperandLifeTime : int32_t {
    /**
     * The operand is internal to the model. It's created by an operation and
     * consumed by other operations. It must be an output operand of
     * exactly one operation.
     */
    TEMPORARY_VARIABLE = 0,
    /**
     * The operand is an input of the model. It must not be an output
     * operand of any operation.
     * 
     * An operand can't be both input and output of a model.
     */
    MODEL_INPUT = 1 /* (::android::hardware::neuralnetworks::V1_0::OperandLifeTime.TEMPORARY_VARIABLE implicitly + 1) */,
    /**
     * The operand is an output of the model. It must be an output
     * operand of exactly one operation.
     * 
     * An operand can't be both input and output of a model.
     */
    MODEL_OUTPUT = 2 /* (::android::hardware::neuralnetworks::V1_0::OperandLifeTime.MODEL_INPUT implicitly + 1) */,
    /**
     * The operand is a constant found in Model.operandValues. It must
     * not be an output operand of any operation.
     */
    CONSTANT_COPY = 3 /* (::android::hardware::neuralnetworks::V1_0::OperandLifeTime.MODEL_OUTPUT implicitly + 1) */,
    /**
     * The operand is a constant that was specified via a Memory
     * object. It must not be an output operand of any operation.
     */
    CONSTANT_REFERENCE = 4 /* (::android::hardware::neuralnetworks::V1_0::OperandLifeTime.CONSTANT_COPY implicitly + 1) */,
    /**
     * The operand does not have a value. This is valid only for optional
     * arguments of operations.
     */
    NO_VALUE = 5 /* (::android::hardware::neuralnetworks::V1_0::OperandLifeTime.CONSTANT_REFERENCE implicitly + 1) */,
};

/**
 * Status of a device.
 */
enum class DeviceStatus : int32_t {
    AVAILABLE = 0,
    BUSY = 1 /* (::android::hardware::neuralnetworks::V1_0::DeviceStatus.AVAILABLE implicitly + 1) */,
    OFFLINE = 2 /* (::android::hardware::neuralnetworks::V1_0::DeviceStatus.BUSY implicitly + 1) */,
    UNKNOWN = 3 /* (::android::hardware::neuralnetworks::V1_0::DeviceStatus.OFFLINE implicitly + 1) */,
};

/**
 * Performance information for the reference workload.
 * 
 * Used by a driver to report its performance characteristics.
 */
struct PerformanceInfo final {
    float execTime __attribute__ ((aligned(4)));
    float powerUsage __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::PerformanceInfo, execTime) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::PerformanceInfo, powerUsage) == 4, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::PerformanceInfo) == 8, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::PerformanceInfo) == 4, "wrong alignment");

/**
 * The capabilities of a driver.
 */
struct Capabilities final {
    ::android::hardware::neuralnetworks::V1_0::PerformanceInfo float32Performance __attribute__ ((aligned(4)));
    ::android::hardware::neuralnetworks::V1_0::PerformanceInfo quantized8Performance __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Capabilities, float32Performance) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Capabilities, quantized8Performance) == 8, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::Capabilities) == 16, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::Capabilities) == 4, "wrong alignment");

/**
 * Describes the location of a data object.
 */
struct DataLocation final {
    uint32_t poolIndex __attribute__ ((aligned(4)));
    uint32_t offset __attribute__ ((aligned(4)));
    uint32_t length __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::DataLocation, poolIndex) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::DataLocation, offset) == 4, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::DataLocation, length) == 8, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::DataLocation) == 12, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::DataLocation) == 4, "wrong alignment");

/**
 * Describes one operand of the model's graph.
 */
struct Operand final {
    ::android::hardware::neuralnetworks::V1_0::OperandType type __attribute__ ((aligned(4)));
#ifndef HIFI_BUILD
    ::android::hardware::hidl_vec<uint32_t> dimensions __attribute__ ((aligned(8)));
#else
    std::vector<uint32_t> dimensions __attribute__ ((aligned(8)));
#endif //HIFI_BUILD
    uint32_t numberOfConsumers __attribute__ ((aligned(4)));
    float scale __attribute__ ((aligned(4)));
    int32_t zeroPoint __attribute__ ((aligned(4)));
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime lifetime __attribute__ ((aligned(4)));
    ::android::hardware::neuralnetworks::V1_0::DataLocation location __attribute__ ((aligned(4)));
};

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, type) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, dimensions) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, numberOfConsumers) == 24, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, scale) == 28, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, zeroPoint) == 32, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, lifetime) == 36, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operand, location) == 40, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::Operand) == 56, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::Operand) == 8, "wrong alignment");
#endif //ppn

/**
 * Describes one operation of the model's graph.
 */
struct Operation final {
    ::android::hardware::neuralnetworks::V1_0::OperationType type __attribute__ ((aligned(4)));
#ifndef HIFI_BUILD
    ::android::hardware::hidl_vec<uint32_t> inputs __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> outputs __attribute__ ((aligned(8)));
#else
    std::vector<uint32_t> inputs __attribute__ ((aligned(8)));
    std::vector<uint32_t> outputs __attribute__ ((aligned(8)));
#endif //HIFI_BUILD
};

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operation, type) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operation, inputs) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Operation, outputs) == 24, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::Operation) == 40, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::Operation) == 8, "wrong alignment");
#endif //ppn

/**
 * A Neural Network Model.
 * 
 * This includes not only the execution graph, but also constant data such as
 * weights or scalars added at construction time. The only information that
 * might not be known is the shape of the input tensors.
 */
#ifndef HIFI_BUILD
struct Model final {
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_0::Operand> operands __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_0::Operation> operations __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> inputIndexes __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> outputIndexes __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint8_t> operandValues __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::hidl_memory> pools __attribute__ ((aligned(8)));
};
#else
struct Model final {
    std::vector <::android::hardware::neuralnetworks::V1_0::Operand> operands __attribute__ ((aligned(8)));
    std::vector <::android::hardware::neuralnetworks::V1_0::Operation> operations __attribute__ ((aligned(8)));
    std::vector <uint32_t> inputIndexes __attribute__ ((aligned(8)));
    std::vector <uint32_t> outputIndexes __attribute__ ((aligned(8)));
    std::vector <uint8_t> operandValues __attribute__ ((aligned(8)));
    std::vector <hidl_memory> pools __attribute__ ((aligned(8)));
};
#endif //HIFI_BUILD

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, operands) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, operations) == 16, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, inputIndexes) == 32, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, outputIndexes) == 48, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, operandValues) == 64, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Model, pools) == 80, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::Model) == 96, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::Model) == 8, "wrong alignment");
#endif //ppn

/**
 * Metadata information specifying the location of the input or output data and
 * any updates to the input or output operand.
 */
struct RequestArgument final {
    bool hasNoValue __attribute__ ((aligned(1)));
    ::android::hardware::neuralnetworks::V1_0::DataLocation location __attribute__ ((aligned(4)));
#ifndef HIFI_BUILD
    ::android::hardware::hidl_vec<uint32_t> dimensions __attribute__ ((aligned(8)));
#else
    std::vector <uint32_t> dimensions __attribute__ ((aligned(8)));
#endif //HIFI_BUILD
};

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::RequestArgument, hasNoValue) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::RequestArgument, location) == 4, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::RequestArgument, dimensions) == 16, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::RequestArgument) == 32, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::RequestArgument) == 8, "wrong alignment");
#endif //ppn

/**
 * Inputs to be sent to and outputs to be retrieved from a prepared model.
 * 
 * A Request serves two primary tasks:
 * 1) Provides the input and output data to be used when executing the model.
 * 2) Specifies any updates to the input operand metadata that were left
 *    unspecified at model preparation time.
 * 
 * An output must not overlap with any other output, with an input, or
 * with an operand of lifetime CONSTANT_REFERENCE.
 */
#ifndef HIFI_BUILD
struct Request final {
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_0::RequestArgument> inputs __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_0::RequestArgument> outputs __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::hidl_memory> pools __attribute__ ((aligned(8)));
};
#else
struct Request final {
    std::vector <::android::hardware::neuralnetworks::V1_0::RequestArgument> inputs __attribute__ ((aligned(8)));
    std::vector <::android::hardware::neuralnetworks::V1_0::RequestArgument> outputs __attribute__ ((aligned(8)));
    std::vector <hidl_memory> pools __attribute__ ((aligned(8)));
};
#endif //HIFI_BUILD

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Request, inputs) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Request, outputs) == 16, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_0::Request, pools) == 32, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_0::Request) == 48, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_0::Request) == 8, "wrong alignment");
#endif //ppn

/**
 * Return status of a function.
 */
enum class ErrorStatus : int32_t {
    NONE = 0,
    DEVICE_UNAVAILABLE = 1 /* (::android::hardware::neuralnetworks::V1_0::ErrorStatus.NONE implicitly + 1) */,
    GENERAL_FAILURE = 2 /* (::android::hardware::neuralnetworks::V1_0::ErrorStatus.DEVICE_UNAVAILABLE implicitly + 1) */,
    OUTPUT_INSUFFICIENT_SIZE = 3 /* (::android::hardware::neuralnetworks::V1_0::ErrorStatus.GENERAL_FAILURE implicitly + 1) */,
    INVALID_ARGUMENT = 4 /* (::android::hardware::neuralnetworks::V1_0::ErrorStatus.OUTPUT_INSUFFICIENT_SIZE implicitly + 1) */,
};

//
// type declarations for package
//

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperandType o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperandType lhs, const ::android::hardware::neuralnetworks::V1_0::OperandType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperandType rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperandType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperandType lhs, const ::android::hardware::neuralnetworks::V1_0::OperandType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperandType rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperandType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperandType e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperandType e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperationType o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperationType lhs, const ::android::hardware::neuralnetworks::V1_0::OperationType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperationType rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperationType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperationType lhs, const ::android::hardware::neuralnetworks::V1_0::OperationType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperationType rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperationType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperationType e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperationType e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc lhs, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc lhs, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperandLifeTime o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime lhs, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime lhs, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::OperandLifeTime e) {
    v &= static_cast<int32_t>(e);
    return v;
}

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::DeviceStatus o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::DeviceStatus lhs, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::DeviceStatus lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::DeviceStatus lhs, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::DeviceStatus lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::DeviceStatus e) {
    v &= static_cast<int32_t>(e);
    return v;
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& lhs, const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& lhs, const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Capabilities& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_0::Capabilities& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_0::Capabilities& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::DataLocation& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::DataLocation& lhs, const ::android::hardware::neuralnetworks::V1_0::DataLocation& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::DataLocation& lhs, const ::android::hardware::neuralnetworks::V1_0::DataLocation& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Operand& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Operand& lhs, const ::android::hardware::neuralnetworks::V1_0::Operand& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Operand& lhs, const ::android::hardware::neuralnetworks::V1_0::Operand& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Operation& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Operation& lhs, const ::android::hardware::neuralnetworks::V1_0::Operation& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Operation& lhs, const ::android::hardware::neuralnetworks::V1_0::Operation& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Model& o);
// operator== and operator!= are not generated for Model

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& lhs, const ::android::hardware::neuralnetworks::V1_0::RequestArgument& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& lhs, const ::android::hardware::neuralnetworks::V1_0::RequestArgument& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Request& o);
// operator== and operator!= are not generated for Request

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_0::ErrorStatus o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::ErrorStatus lhs, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_0::ErrorStatus lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::ErrorStatus lhs, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_0::ErrorStatus lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_0::ErrorStatus e) {
    v &= static_cast<int32_t>(e);
    return v;
}

//
// type header definitions for package
//

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::OperandType>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::OperandType> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::FLOAT32) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::FLOAT32)) {
        os += (first ? "" : " | ");
        os += "FLOAT32";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::FLOAT32;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::INT32) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::INT32)) {
        os += (first ? "" : " | ");
        os += "INT32";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::INT32;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::UINT32) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::UINT32)) {
        os += (first ? "" : " | ");
        os += "UINT32";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::UINT32;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_FLOAT32) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_FLOAT32)) {
        os += (first ? "" : " | ");
        os += "TENSOR_FLOAT32";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_FLOAT32;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_INT32) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_INT32)) {
        os += (first ? "" : " | ");
        os += "TENSOR_INT32";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_INT32;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM)) {
        os += (first ? "" : " | ");
        os += "TENSOR_QUANT8_ASYMM";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::OEM) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::OEM)) {
        os += (first ? "" : " | ");
        os += "OEM";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::OEM;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_OEM_BYTE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_OEM_BYTE)) {
        os += (first ? "" : " | ");
        os += "TENSOR_OEM_BYTE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_OEM_BYTE;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperandType o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::FLOAT32) {
        return "FLOAT32";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::INT32) {
        return "INT32";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::UINT32) {
        return "UINT32";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_FLOAT32) {
        return "TENSOR_FLOAT32";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_INT32) {
        return "TENSOR_INT32";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM) {
        return "TENSOR_QUANT8_ASYMM";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::OEM) {
        return "OEM";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_OEM_BYTE) {
        return "TENSOR_OEM_BYTE";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::OperationType>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::OperationType> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::ADD) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::ADD)) {
        os += (first ? "" : " | ");
        os += "ADD";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::ADD;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "AVERAGE_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::CONCATENATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::CONCATENATION)) {
        os += (first ? "" : " | ");
        os += "CONCATENATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::CONCATENATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::CONV_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::CONV_2D)) {
        os += (first ? "" : " | ");
        os += "CONV_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::CONV_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D)) {
        os += (first ? "" : " | ");
        os += "DEPTHWISE_CONV_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE)) {
        os += (first ? "" : " | ");
        os += "DEPTH_TO_SPACE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::DEQUANTIZE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::DEQUANTIZE)) {
        os += (first ? "" : " | ");
        os += "DEQUANTIZE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::DEQUANTIZE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP)) {
        os += (first ? "" : " | ");
        os += "EMBEDDING_LOOKUP";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::FLOOR) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::FLOOR)) {
        os += (first ? "" : " | ");
        os += "FLOOR";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::FLOOR;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::FULLY_CONNECTED) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::FULLY_CONNECTED)) {
        os += (first ? "" : " | ");
        os += "FULLY_CONNECTED";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::FULLY_CONNECTED;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP)) {
        os += (first ? "" : " | ");
        os += "HASHTABLE_LOOKUP";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::L2_NORMALIZATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::L2_NORMALIZATION)) {
        os += (first ? "" : " | ");
        os += "L2_NORMALIZATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::L2_NORMALIZATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::L2_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::L2_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "L2_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::L2_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION)) {
        os += (first ? "" : " | ");
        os += "LOCAL_RESPONSE_NORMALIZATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::LOGISTIC) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::LOGISTIC)) {
        os += (first ? "" : " | ");
        os += "LOGISTIC";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::LOGISTIC;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::LSH_PROJECTION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::LSH_PROJECTION)) {
        os += (first ? "" : " | ");
        os += "LSH_PROJECTION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::LSH_PROJECTION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::LSTM) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::LSTM)) {
        os += (first ? "" : " | ");
        os += "LSTM";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::LSTM;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::MAX_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::MAX_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "MAX_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::MAX_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::MUL) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::MUL)) {
        os += (first ? "" : " | ");
        os += "MUL";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::MUL;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RELU) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RELU)) {
        os += (first ? "" : " | ");
        os += "RELU";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RELU;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RELU1) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RELU1)) {
        os += (first ? "" : " | ");
        os += "RELU1";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RELU1;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RELU6) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RELU6)) {
        os += (first ? "" : " | ");
        os += "RELU6";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RELU6;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RESHAPE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RESHAPE)) {
        os += (first ? "" : " | ");
        os += "RESHAPE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RESHAPE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR)) {
        os += (first ? "" : " | ");
        os += "RESIZE_BILINEAR";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::RNN) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::RNN)) {
        os += (first ? "" : " | ");
        os += "RNN";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::RNN;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::SOFTMAX) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::SOFTMAX)) {
        os += (first ? "" : " | ");
        os += "SOFTMAX";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::SOFTMAX;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::SPACE_TO_DEPTH) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::SPACE_TO_DEPTH)) {
        os += (first ? "" : " | ");
        os += "SPACE_TO_DEPTH";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::SPACE_TO_DEPTH;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::SVDF) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::SVDF)) {
        os += (first ? "" : " | ");
        os += "SVDF";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::SVDF;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::TANH) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::TANH)) {
        os += (first ? "" : " | ");
        os += "TANH";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::TANH;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperationType::OEM_OPERATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperationType::OEM_OPERATION)) {
        os += (first ? "" : " | ");
        os += "OEM_OPERATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperationType::OEM_OPERATION;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperationType o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::ADD) {
        return "ADD";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D) {
        return "AVERAGE_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::CONCATENATION) {
        return "CONCATENATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::CONV_2D) {
        return "CONV_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D) {
        return "DEPTHWISE_CONV_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE) {
        return "DEPTH_TO_SPACE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::DEQUANTIZE) {
        return "DEQUANTIZE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP) {
        return "EMBEDDING_LOOKUP";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::FLOOR) {
        return "FLOOR";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::FULLY_CONNECTED) {
        return "FULLY_CONNECTED";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP) {
        return "HASHTABLE_LOOKUP";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::L2_NORMALIZATION) {
        return "L2_NORMALIZATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::L2_POOL_2D) {
        return "L2_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION) {
        return "LOCAL_RESPONSE_NORMALIZATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::LOGISTIC) {
        return "LOGISTIC";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::LSH_PROJECTION) {
        return "LSH_PROJECTION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::LSTM) {
        return "LSTM";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::MAX_POOL_2D) {
        return "MAX_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::MUL) {
        return "MUL";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RELU) {
        return "RELU";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RELU1) {
        return "RELU1";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RELU6) {
        return "RELU6";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RESHAPE) {
        return "RESHAPE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR) {
        return "RESIZE_BILINEAR";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::RNN) {
        return "RNN";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::SOFTMAX) {
        return "SOFTMAX";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::SPACE_TO_DEPTH) {
        return "SPACE_TO_DEPTH";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::SVDF) {
        return "SVDF";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::TANH) {
        return "TANH";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperationType::OEM_OPERATION) {
        return "OEM_OPERATION";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::FusedActivationFunc>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::FusedActivationFunc> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::NONE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::NONE)) {
        os += (first ? "" : " | ");
        os += "NONE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::NONE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU)) {
        os += (first ? "" : " | ");
        os += "RELU";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU1) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU1)) {
        os += (first ? "" : " | ");
        os += "RELU1";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU1;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU6) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU6)) {
        os += (first ? "" : " | ");
        os += "RELU6";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU6;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::FusedActivationFunc o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::NONE) {
        return "NONE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU) {
        return "RELU";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU1) {
        return "RELU1";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU6) {
        return "RELU6";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::OperandLifeTime>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::OperandLifeTime> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::TEMPORARY_VARIABLE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::TEMPORARY_VARIABLE)) {
        os += (first ? "" : " | ");
        os += "TEMPORARY_VARIABLE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::TEMPORARY_VARIABLE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_INPUT) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_INPUT)) {
        os += (first ? "" : " | ");
        os += "MODEL_INPUT";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_INPUT;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_OUTPUT) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_OUTPUT)) {
        os += (first ? "" : " | ");
        os += "MODEL_OUTPUT";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_OUTPUT;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_COPY) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_COPY)) {
        os += (first ? "" : " | ");
        os += "CONSTANT_COPY";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_COPY;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_REFERENCE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_REFERENCE)) {
        os += (first ? "" : " | ");
        os += "CONSTANT_REFERENCE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_REFERENCE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::NO_VALUE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::OperandLifeTime::NO_VALUE)) {
        os += (first ? "" : " | ");
        os += "NO_VALUE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::NO_VALUE;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::OperandLifeTime o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::TEMPORARY_VARIABLE) {
        return "TEMPORARY_VARIABLE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_INPUT) {
        return "MODEL_INPUT";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_OUTPUT) {
        return "MODEL_OUTPUT";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_COPY) {
        return "CONSTANT_COPY";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_REFERENCE) {
        return "CONSTANT_REFERENCE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::NO_VALUE) {
        return "NO_VALUE";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::DeviceStatus>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::DeviceStatus> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::DeviceStatus::AVAILABLE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::DeviceStatus::AVAILABLE)) {
        os += (first ? "" : " | ");
        os += "AVAILABLE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::DeviceStatus::AVAILABLE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::DeviceStatus::BUSY) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::DeviceStatus::BUSY)) {
        os += (first ? "" : " | ");
        os += "BUSY";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::DeviceStatus::BUSY;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::DeviceStatus::OFFLINE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::DeviceStatus::OFFLINE)) {
        os += (first ? "" : " | ");
        os += "OFFLINE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::DeviceStatus::OFFLINE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::DeviceStatus::UNKNOWN) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::DeviceStatus::UNKNOWN)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::DeviceStatus::UNKNOWN;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::DeviceStatus o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::DeviceStatus::AVAILABLE) {
        return "AVAILABLE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::DeviceStatus::BUSY) {
        return "BUSY";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::DeviceStatus::OFFLINE) {
        return "OFFLINE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::DeviceStatus::UNKNOWN) {
        return "UNKNOWN";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".execTime = ";
    os += ::android::hardware::toString(o.execTime);
    os += ", .powerUsage = ";
    os += ::android::hardware::toString(o.powerUsage);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& lhs, const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& rhs) {
    if (lhs.execTime != rhs.execTime) {
        return false;
    }
    if (lhs.powerUsage != rhs.powerUsage) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& lhs, const ::android::hardware::neuralnetworks::V1_0::PerformanceInfo& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Capabilities& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".float32Performance = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.float32Performance);
    os += ", .quantized8Performance = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.quantized8Performance);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_0::Capabilities& rhs) {
    if (lhs.float32Performance != rhs.float32Performance) {
        return false;
    }
    if (lhs.quantized8Performance != rhs.quantized8Performance) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_0::Capabilities& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::DataLocation& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".poolIndex = ";
    os += ::android::hardware::toString(o.poolIndex);
    os += ", .offset = ";
    os += ::android::hardware::toString(o.offset);
    os += ", .length = ";
    os += ::android::hardware::toString(o.length);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::DataLocation& lhs, const ::android::hardware::neuralnetworks::V1_0::DataLocation& rhs) {
    if (lhs.poolIndex != rhs.poolIndex) {
        return false;
    }
    if (lhs.offset != rhs.offset) {
        return false;
    }
    if (lhs.length != rhs.length) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::DataLocation& lhs, const ::android::hardware::neuralnetworks::V1_0::DataLocation& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Operand& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".type = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.type);
    os += ", .dimensions = ";
    os += ::android::hardware::toString(o.dimensions);
    os += ", .numberOfConsumers = ";
    os += ::android::hardware::toString(o.numberOfConsumers);
    os += ", .scale = ";
    os += ::android::hardware::toString(o.scale);
    os += ", .zeroPoint = ";
    os += ::android::hardware::toString(o.zeroPoint);
    os += ", .lifetime = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.lifetime);
    os += ", .location = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.location);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Operand& lhs, const ::android::hardware::neuralnetworks::V1_0::Operand& rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }
    if (lhs.dimensions != rhs.dimensions) {
        return false;
    }
    if (lhs.numberOfConsumers != rhs.numberOfConsumers) {
        return false;
    }
    if (lhs.scale != rhs.scale) {
        return false;
    }
    if (lhs.zeroPoint != rhs.zeroPoint) {
        return false;
    }
    if (lhs.lifetime != rhs.lifetime) {
        return false;
    }
    if (lhs.location != rhs.location) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Operand& lhs, const ::android::hardware::neuralnetworks::V1_0::Operand& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Operation& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".type = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.type);
    os += ", .inputs = ";
    os += ::android::hardware::toString(o.inputs);
    os += ", .outputs = ";
    os += ::android::hardware::toString(o.outputs);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::Operation& lhs, const ::android::hardware::neuralnetworks::V1_0::Operation& rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }
    if (lhs.inputs != rhs.inputs) {
        return false;
    }
    if (lhs.outputs != rhs.outputs) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::Operation& lhs, const ::android::hardware::neuralnetworks::V1_0::Operation& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Model& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".operands = ";
    os += ::android::hardware::toString(o.operands);
    os += ", .operations = ";
    os += ::android::hardware::toString(o.operations);
    os += ", .inputIndexes = ";
    os += ::android::hardware::toString(o.inputIndexes);
    os += ", .outputIndexes = ";
    os += ::android::hardware::toString(o.outputIndexes);
    os += ", .operandValues = ";
    os += ::android::hardware::toString(o.operandValues);
    os += ", .pools = ";
    os += ::android::hardware::toString(o.pools);
    os += "}"; return os;
}

// operator== and operator!= are not generated for Model

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".hasNoValue = ";
    os += ::android::hardware::toString(o.hasNoValue);
    os += ", .location = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.location);
    os += ", .dimensions = ";
    os += ::android::hardware::toString(o.dimensions);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& lhs, const ::android::hardware::neuralnetworks::V1_0::RequestArgument& rhs) {
    if (lhs.hasNoValue != rhs.hasNoValue) {
        return false;
    }
    if (lhs.location != rhs.location) {
        return false;
    }
    if (lhs.dimensions != rhs.dimensions) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_0::RequestArgument& lhs, const ::android::hardware::neuralnetworks::V1_0::RequestArgument& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_0::Request& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".inputs = ";
    os += ::android::hardware::toString(o.inputs);
    os += ", .outputs = ";
    os += ::android::hardware::toString(o.outputs);
    os += ", .pools = ";
    os += ::android::hardware::toString(o.pools);
    os += "}"; return os;
}

// operator== and operator!= are not generated for Request

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_0::ErrorStatus>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_0::ErrorStatus> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_0::ErrorStatus::NONE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::ErrorStatus::NONE)) {
        os += (first ? "" : " | ");
        os += "NONE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::ErrorStatus::NONE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::ErrorStatus::DEVICE_UNAVAILABLE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::ErrorStatus::DEVICE_UNAVAILABLE)) {
        os += (first ? "" : " | ");
        os += "DEVICE_UNAVAILABLE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::ErrorStatus::DEVICE_UNAVAILABLE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::ErrorStatus::GENERAL_FAILURE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::ErrorStatus::GENERAL_FAILURE)) {
        os += (first ? "" : " | ");
        os += "GENERAL_FAILURE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::ErrorStatus::GENERAL_FAILURE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE)) {
        os += (first ? "" : " | ");
        os += "OUTPUT_INSUFFICIENT_SIZE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_0::ErrorStatus::INVALID_ARGUMENT) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_0::ErrorStatus::INVALID_ARGUMENT)) {
        os += (first ? "" : " | ");
        os += "INVALID_ARGUMENT";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_0::ErrorStatus::INVALID_ARGUMENT;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_0::ErrorStatus o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_0::ErrorStatus::NONE) {
        return "NONE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::ErrorStatus::DEVICE_UNAVAILABLE) {
        return "DEVICE_UNAVAILABLE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::ErrorStatus::GENERAL_FAILURE) {
        return "GENERAL_FAILURE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE) {
        return "OUTPUT_INSUFFICIENT_SIZE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_0::ErrorStatus::INVALID_ARGUMENT) {
        return "INVALID_ARGUMENT";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android


//
// global type declarations for package
//

#if 0 //ppn
namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::OperandType, 8> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::OperandType> = {
    ::android::hardware::neuralnetworks::V1_0::OperandType::FLOAT32,
    ::android::hardware::neuralnetworks::V1_0::OperandType::INT32,
    ::android::hardware::neuralnetworks::V1_0::OperandType::UINT32,
    ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_FLOAT32,
    ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_INT32,
    ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_QUANT8_ASYMM,
    ::android::hardware::neuralnetworks::V1_0::OperandType::OEM,
    ::android::hardware::neuralnetworks::V1_0::OperandType::TENSOR_OEM_BYTE,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::OperationType, 30> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::OperationType> = {
    ::android::hardware::neuralnetworks::V1_0::OperationType::ADD,
    ::android::hardware::neuralnetworks::V1_0::OperationType::AVERAGE_POOL_2D,
    ::android::hardware::neuralnetworks::V1_0::OperationType::CONCATENATION,
    ::android::hardware::neuralnetworks::V1_0::OperationType::CONV_2D,
    ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTHWISE_CONV_2D,
    ::android::hardware::neuralnetworks::V1_0::OperationType::DEPTH_TO_SPACE,
    ::android::hardware::neuralnetworks::V1_0::OperationType::DEQUANTIZE,
    ::android::hardware::neuralnetworks::V1_0::OperationType::EMBEDDING_LOOKUP,
    ::android::hardware::neuralnetworks::V1_0::OperationType::FLOOR,
    ::android::hardware::neuralnetworks::V1_0::OperationType::FULLY_CONNECTED,
    ::android::hardware::neuralnetworks::V1_0::OperationType::HASHTABLE_LOOKUP,
    ::android::hardware::neuralnetworks::V1_0::OperationType::L2_NORMALIZATION,
    ::android::hardware::neuralnetworks::V1_0::OperationType::L2_POOL_2D,
    ::android::hardware::neuralnetworks::V1_0::OperationType::LOCAL_RESPONSE_NORMALIZATION,
    ::android::hardware::neuralnetworks::V1_0::OperationType::LOGISTIC,
    ::android::hardware::neuralnetworks::V1_0::OperationType::LSH_PROJECTION,
    ::android::hardware::neuralnetworks::V1_0::OperationType::LSTM,
    ::android::hardware::neuralnetworks::V1_0::OperationType::MAX_POOL_2D,
    ::android::hardware::neuralnetworks::V1_0::OperationType::MUL,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RELU,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RELU1,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RELU6,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RESHAPE,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RESIZE_BILINEAR,
    ::android::hardware::neuralnetworks::V1_0::OperationType::RNN,
    ::android::hardware::neuralnetworks::V1_0::OperationType::SOFTMAX,
    ::android::hardware::neuralnetworks::V1_0::OperationType::SPACE_TO_DEPTH,
    ::android::hardware::neuralnetworks::V1_0::OperationType::SVDF,
    ::android::hardware::neuralnetworks::V1_0::OperationType::TANH,
    ::android::hardware::neuralnetworks::V1_0::OperationType::OEM_OPERATION,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::FusedActivationFunc, 4> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::FusedActivationFunc> = {
    ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::NONE,
    ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU,
    ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU1,
    ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc::RELU6,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::OperandLifeTime, 6> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::OperandLifeTime> = {
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::TEMPORARY_VARIABLE,
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_INPUT,
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::MODEL_OUTPUT,
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_COPY,
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::CONSTANT_REFERENCE,
    ::android::hardware::neuralnetworks::V1_0::OperandLifeTime::NO_VALUE,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::DeviceStatus, 4> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::DeviceStatus> = {
    ::android::hardware::neuralnetworks::V1_0::DeviceStatus::AVAILABLE,
    ::android::hardware::neuralnetworks::V1_0::DeviceStatus::BUSY,
    ::android::hardware::neuralnetworks::V1_0::DeviceStatus::OFFLINE,
    ::android::hardware::neuralnetworks::V1_0::DeviceStatus::UNKNOWN,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_0::ErrorStatus, 5> hidl_enum_values<::android::hardware::neuralnetworks::V1_0::ErrorStatus> = {
    ::android::hardware::neuralnetworks::V1_0::ErrorStatus::NONE,
    ::android::hardware::neuralnetworks::V1_0::ErrorStatus::DEVICE_UNAVAILABLE,
    ::android::hardware::neuralnetworks::V1_0::ErrorStatus::GENERAL_FAILURE,
    ::android::hardware::neuralnetworks::V1_0::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE,
    ::android::hardware::neuralnetworks::V1_0::ErrorStatus::INVALID_ARGUMENT,
};
}  // namespace details
}  // namespace hardware
}  // namespace android
#endif //ppn


#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_TYPES_H
