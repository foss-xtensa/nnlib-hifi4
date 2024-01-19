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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_TYPES_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_TYPES_H

#include <android/hardware/neuralnetworks/1.0/types.h>

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
namespace V1_1 {

// Forward declaration for forward reference support:
enum class OperationType : int32_t;
struct Capabilities;
struct Operation;
struct Model;
enum class ExecutionPreference : int32_t;

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
    /**
     * BatchToSpace for N-dimensional tensors.
     * 
     * This operation reshapes the batch dimension (dimension 0) into M + 1
     * dimensions of shape block_shape + [batch], interleaves these blocks back
     * into the grid defined by the spatial dimensions [1, ..., M], to obtain a
     * result with the same rank as the input.
     * 
     * This is the reverse of SpaceToBatch.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be reshaped
     * * 1: A 1-D Tensor of {@link OperandType::TENSOR_INT32}, the block
     *      sizes for each spatial dimension of the input tensor. All values
     *      must be >= 1.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    BATCH_TO_SPACE_ND = 29,
    /**
     * Element-wise division of two tensors.
     * 
     * Takes two input tensors of identical {@link OperandType} and compatible
     * dimensions. The output is the result of dividing the first input tensor
     * by the second, optionally modified by an activation function.
     * 
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     * 
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way
     * forward.
     * 
     * Example:
     *     input1.dimension =    {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the first input.
     * * 1: A tensor of the same {@link OperandType}, and compatible dimensions
     *      as input0.
     * * 2: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    DIV = 30,
    /**
     * Computes the mean of elements across dimensions of a tensor.
     * 
     * Reduces the input tensor along the given dimensions to reduce. Unless
     * keep_dims is true, the rank of the tensor is reduced by 1 for each entry
     * in axis. If keep_dims is true, the reduced dimensions are retained with
     * length 1.
     * 
     * If dimensions to reduce have no entries, all dimensions are reduced, and
     * a tensor with a single element is returned.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: A tensor, specifying the input.
     * * 1: A 1-D Tensor of {@link OperandType::TENSOR_INT32}. The dimensions
     *      to reduce. If None (the default), reduces all dimensions. Must be in
     *      the range [-rank(input_tensor), rank(input_tensor)).
     * * 2: An {@link OperandType::INT32} scalar, keep_dims. If positive,
     *      retains reduced dimensions with length 1.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    MEAN = 31,
    /**
     * Pads a tensor.
     * 
     * This operation pads a tensor according to the specified paddings.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be padded.
     * * 1: A 2-D Tensor of {@link OperandType::TENSOR_INT32}, the paddings
     *      for each spatial dimension of the input tensor. The shape of the
     *      tensor must be {rank(input0), 2}.
     *      padding[i, 0] specifies the number of elements to be padded in the
     *      front of dimension i.
     *      padding[i, 1] specifies the number of elements to be padded after the
     *      end of dimension i.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0. The
     *      output tensor has the same rank as input0, and each
     *      dimension of the output tensor has the same size as the
     *      corresponding dimension of the input tensor plus the size
     *      of the padding:
     *          output0.dimension[i] =
     *              padding[i, 0] + input0.dimension[i] + padding[i, 1]
     * 
     * Available since API level 28.
     */
    PAD = 32,
    /**
     * SpaceToBatch for N-Dimensional tensors.
     * 
     * This operation divides "spatial" dimensions [1, ..., M] of the input into
     * a grid of blocks of shape block_shape, and interleaves these blocks with
     * the "batch" dimension (0) such that in the output, the spatial dimensions
     * [1, ..., M] correspond to the position within the grid, and the batch
     * dimension combines both the position within a spatial block and the
     * original batch position. Prior to division into blocks, the spatial
     * dimensions of the input are optionally zero padded according to paddings.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the input.
     * * 1: A 1-D Tensor of {@link OperandType::TENSOR_INT32}, the block
     *      sizes for each spatial dimension of the input tensor. All values
     *      must be >= 1.
     * * 2: A 2-D Tensor of {@link OperandType::TENSOR_INT32}, the paddings
     *      for each spatial dimension of the input tensor. All values must be
     *      >= 0. The shape of the tensor must be {rank(input0), 2}.
     *      padding[i, 0] specifies the number of element to be padded in the
     *      front of dimension i.
     *      padding[i, 1] specifies the number of element to be padded after the
     *      end of dimension i.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    SPACE_TO_BATCH_ND = 33,
    /**
     * Removes dimensions of size 1 from the shape of a tensor.
     * 
     * Given a tensor input, this operation returns a tensor of the same
     * {@link OperandType} with all dimensions of size 1 removed. If you don't
     * want to remove all size 1 dimensions, you can remove specific size 1
     * dimensions by specifying the axes (input1).
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, the tensor to be squeezed.
     * * 1: An optional 1-D tensor of {@link OperandType::TENSOR_INT32}. The
     *      dimensions to squeeze. If specified only squeezes the dimensions
     *      listed. Otherwise, squeezes all dimensions. The dimension index
     *      starts at 0. An error must be reported if squeezing a dimension that
     *      is not 1.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0. Contains the
     *      same data as input, but has one or more dimensions of size 1
     *      removed.
     * 
     * Available since API level 28.
     */
    SQUEEZE = 34,
    /**
     * Extracts a strided slice of a tensor.
     * 
     * Roughly speaking, this op extracts a slice of size (end - begin) / stride
     * from the given input tensor. Starting at the location specified by begin
     * the slice continues by adding stride to the index until all dimensions
     * are not less than end. Note that a stride can be negative, which causes a
     * reverse slice.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be sliced.
     * * 1: begin, a 1-D tensor of {@link OperandType::TENSOR_INT32}. The
     *      starts of the dimensions of the input tensor to be sliced. The
     *      length must be of rank(input0).
     * * 2: end, a 1-D tensor of {@link OperandType::TENSOR_INT32}. The
     *      ends of the dimensions of the input tensor to be sliced. The length
     *      must be of rank(input0).
     * * 3: strides, a 1-D tensor of {@link OperandType::TENSOR_INT32}. The
     *      strides of the dimensions of the input tensor to be sliced. The
     *      length must be of rank(input0). The entries must be non-zero.
     * * 4: begin_mask, an {@link OperandType::INT32} scalar. If the ith bit
     *      of begin_mask is set, begin[i] is ignored and the fullest possible
     *      range in that dimension is used instead.
     * * 5: end_mask, an {@link OperandType::INT32} scalar. If the ith bit of
     *      end_mask is set, end[i] is ignored and the fullest possible range in
     *      that dimension is used instead.
     * * 6: shrink_axis_mask, an {@link OperandType::INT32} scalar. If the
     *      ith bit of shrink_axis_mask is set, the ith dimension specification
     *      shrinks the dimensionality by 1, taking on the value at index
     *      begin[i]. In this case, the ith specification must define a
     *      slice of size 1, e.g. begin[i] = x, end[i] = x + 1.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0 and rank (n - k),
     *      where k is the number of bits set in shrink_axis_mask.
     * 
     * Available since API level 28.
     */
    STRIDED_SLICE = 35,
    /**
     * Element-wise subtraction of two tensors.
     * 
     * Takes two input tensors of identical {@link OperandType} and compatible
     * dimensions. The output is the result of subtracting the second input
     * tensor from the first one, optionally modified by an activation function.
     * 
     * Two dimensions are compatible when:
     *     1. they are equal, or
     *     2. one of them is 1
     * 
     * The size of the output is the maximum size along each dimension of the
     * input operands. It starts with the trailing dimensions, and works its way
     * forward.
     * 
     * Example:
     *     input1.dimension =    {4, 1, 2}
     *     input2.dimension = {5, 4, 3, 1}
     *     output.dimension = {5, 4, 3, 2}
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the first input.
     * * 1: A tensor of the same {@link OperandType}, and compatible dimensions
     *      as input0.
     * * 2: An {@link OperandType::INT32} scalar, and has to be one of the
     *      {@link FusedActivationFunc} values. Specifies the activation to
     *      invoke on the result.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    SUB = 36,
    /**
     * Transposes the input tensor, permuting the dimensions according to the
     * perm tensor.
     * 
     * The returned tensor's dimension i corresponds to the input dimension
     * perm[i]. If perm is not given, it is set to (n-1...0), where n is the
     * rank of the input tensor. Hence by default, this operation performs a
     * regular matrix transpose on 2-D input Tensors.
     * 
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     * 
     * Supported tensor rank: up to 4
     * 
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be transposed.
     * * 1: An optional 1-D Tensor of {@link OperandType::TENSOR_INT32},
     *      the permutation of the dimensions of the input tensor.
     * 
     * Outputs:
     * * 0: A tensor of the same {@link OperandType} as input0.
     * 
     * Available since API level 28.
     */
    TRANSPOSE = 37,
};

/**
 * The capabilities of a driver.
 */
struct Capabilities final {
    ::android::hardware::neuralnetworks::V1_0::PerformanceInfo float32Performance __attribute__ ((aligned(4)));
    ::android::hardware::neuralnetworks::V1_0::PerformanceInfo quantized8Performance __attribute__ ((aligned(4)));
    ::android::hardware::neuralnetworks::V1_0::PerformanceInfo relaxedFloat32toFloat16Performance __attribute__ ((aligned(4)));
};

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Capabilities, float32Performance) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Capabilities, quantized8Performance) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Capabilities, relaxedFloat32toFloat16Performance) == 16, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_1::Capabilities) == 24, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_1::Capabilities) == 4, "wrong alignment");
#endif //ppn

/**
 * Describes one operation of the model's graph.
 */
struct Operation final {
    ::android::hardware::neuralnetworks::V1_1::OperationType type __attribute__ ((aligned(4)));
#ifndef HIFI_BUILD
    ::android::hardware::hidl_vec<uint32_t> inputs __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> outputs __attribute__ ((aligned(8)));
#else
    std::vector <uint32_t> inputs __attribute__ ((aligned(8)));
    std::vector <uint32_t> outputs __attribute__ ((aligned(8)));
#endif //HIFI_BUILD
};

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Operation, type) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Operation, inputs) == 8, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Operation, outputs) == 24, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_1::Operation) == 40, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_1::Operation) == 8, "wrong alignment");
#endif //ppn

/**
 * A Neural Network Model.
 * 
 * This includes not only the execution graph, but also constant data such as
 * weights or scalars added at construction time. The only information that
 * may not be known is the shape of the input tensors.
 */
#ifndef HIFI_BUILD
struct Model final {
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_0::Operand> operands __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::neuralnetworks::V1_1::Operation> operations __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> inputIndexes __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint32_t> outputIndexes __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<uint8_t> operandValues __attribute__ ((aligned(8)));
    ::android::hardware::hidl_vec<::android::hardware::hidl_memory> pools __attribute__ ((aligned(8)));
    bool relaxComputationFloat32toFloat16 __attribute__ ((aligned(1)));
};
#else
struct Model final {
    std::vector <::android::hardware::neuralnetworks::V1_0::Operand> operands __attribute__ ((aligned(8)));
    std::vector <::android::hardware::neuralnetworks::V1_1::Operation> operations __attribute__ ((aligned(8)));
    std::vector <uint32_t> inputIndexes __attribute__ ((aligned(8)));
    std::vector <uint32_t> outputIndexes __attribute__ ((aligned(8)));
    std::vector <uint8_t> operandValues __attribute__ ((aligned(8)));
    std::vector <hidl_memory> pools __attribute__ ((aligned(8)));
    bool relaxComputationFloat32toFloat16 __attribute__ ((aligned(1)));
};
#endif //HIFI_BUILD

#if 0 //ppn
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, operands) == 0, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, operations) == 16, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, inputIndexes) == 32, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, outputIndexes) == 48, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, operandValues) == 64, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, pools) == 80, "wrong offset");
static_assert(offsetof(::android::hardware::neuralnetworks::V1_1::Model, relaxComputationFloat32toFloat16) == 96, "wrong offset");
static_assert(sizeof(::android::hardware::neuralnetworks::V1_1::Model) == 104, "wrong size");
static_assert(__alignof(::android::hardware::neuralnetworks::V1_1::Model) == 8, "wrong alignment");
#endif //ppn

/**
 * Execution preferences.
 */
enum class ExecutionPreference : int32_t {
    /**
     * Prefer executing in a way that minimizes battery drain.
     * This is desirable for compilations that will be executed often.
     */
    LOW_POWER = 0,
    /**
     * Prefer returning a single answer as fast as possible, even if this causes
     * more power consumption.
     */
    FAST_SINGLE_ANSWER = 1,
    /**
     * Prefer maximizing the throughput of successive frames, for example when
     * processing successive frames coming from the camera.
     */
    SUSTAINED_SPEED = 2,
};

//
// type declarations for package
//

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_1::OperationType o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_1::OperationType lhs, const ::android::hardware::neuralnetworks::V1_1::OperationType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_1::OperationType rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_1::OperationType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_1::OperationType lhs, const ::android::hardware::neuralnetworks::V1_1::OperationType rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_1::OperationType rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_1::OperationType lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_1::OperationType e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_1::OperationType e) {
    v &= static_cast<int32_t>(e);
    return v;
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Capabilities& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_1::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_1::Capabilities& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_1::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_1::Capabilities& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Operation& o);
static inline bool operator==(const ::android::hardware::neuralnetworks::V1_1::Operation& lhs, const ::android::hardware::neuralnetworks::V1_1::Operation& rhs);
static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_1::Operation& lhs, const ::android::hardware::neuralnetworks::V1_1::Operation& rhs);

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Model& o);
// operator== and operator!= are not generated for Model

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hardware::neuralnetworks::V1_1::ExecutionPreference o);

constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference lhs, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference lhs, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static inline int32_t &operator|=(int32_t& v, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static inline int32_t &operator&=(int32_t& v, const ::android::hardware::neuralnetworks::V1_1::ExecutionPreference e) {
    v &= static_cast<int32_t>(e);
    return v;
}

//
// type header definitions for package
//

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_1::OperationType>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_1::OperationType> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::ADD) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::ADD)) {
        os += (first ? "" : " | ");
        os += "ADD";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::ADD;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::AVERAGE_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::AVERAGE_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "AVERAGE_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::AVERAGE_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::CONCATENATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::CONCATENATION)) {
        os += (first ? "" : " | ");
        os += "CONCATENATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::CONCATENATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::CONV_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::CONV_2D)) {
        os += (first ? "" : " | ");
        os += "CONV_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::CONV_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTHWISE_CONV_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::DEPTHWISE_CONV_2D)) {
        os += (first ? "" : " | ");
        os += "DEPTHWISE_CONV_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTHWISE_CONV_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTH_TO_SPACE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::DEPTH_TO_SPACE)) {
        os += (first ? "" : " | ");
        os += "DEPTH_TO_SPACE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTH_TO_SPACE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::DEQUANTIZE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::DEQUANTIZE)) {
        os += (first ? "" : " | ");
        os += "DEQUANTIZE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::DEQUANTIZE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::EMBEDDING_LOOKUP) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::EMBEDDING_LOOKUP)) {
        os += (first ? "" : " | ");
        os += "EMBEDDING_LOOKUP";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::EMBEDDING_LOOKUP;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::FLOOR) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::FLOOR)) {
        os += (first ? "" : " | ");
        os += "FLOOR";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::FLOOR;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::FULLY_CONNECTED) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::FULLY_CONNECTED)) {
        os += (first ? "" : " | ");
        os += "FULLY_CONNECTED";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::FULLY_CONNECTED;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::HASHTABLE_LOOKUP) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::HASHTABLE_LOOKUP)) {
        os += (first ? "" : " | ");
        os += "HASHTABLE_LOOKUP";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::HASHTABLE_LOOKUP;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::L2_NORMALIZATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::L2_NORMALIZATION)) {
        os += (first ? "" : " | ");
        os += "L2_NORMALIZATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::L2_NORMALIZATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::L2_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::L2_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "L2_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::L2_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION)) {
        os += (first ? "" : " | ");
        os += "LOCAL_RESPONSE_NORMALIZATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::LOGISTIC) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::LOGISTIC)) {
        os += (first ? "" : " | ");
        os += "LOGISTIC";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::LOGISTIC;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::LSH_PROJECTION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::LSH_PROJECTION)) {
        os += (first ? "" : " | ");
        os += "LSH_PROJECTION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::LSH_PROJECTION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::LSTM) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::LSTM)) {
        os += (first ? "" : " | ");
        os += "LSTM";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::LSTM;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::MAX_POOL_2D) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::MAX_POOL_2D)) {
        os += (first ? "" : " | ");
        os += "MAX_POOL_2D";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::MAX_POOL_2D;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::MUL) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::MUL)) {
        os += (first ? "" : " | ");
        os += "MUL";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::MUL;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RELU) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RELU)) {
        os += (first ? "" : " | ");
        os += "RELU";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RELU;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RELU1) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RELU1)) {
        os += (first ? "" : " | ");
        os += "RELU1";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RELU1;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RELU6) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RELU6)) {
        os += (first ? "" : " | ");
        os += "RELU6";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RELU6;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RESHAPE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RESHAPE)) {
        os += (first ? "" : " | ");
        os += "RESHAPE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RESHAPE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RESIZE_BILINEAR) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RESIZE_BILINEAR)) {
        os += (first ? "" : " | ");
        os += "RESIZE_BILINEAR";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RESIZE_BILINEAR;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::RNN) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::RNN)) {
        os += (first ? "" : " | ");
        os += "RNN";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::RNN;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SOFTMAX) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SOFTMAX)) {
        os += (first ? "" : " | ");
        os += "SOFTMAX";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SOFTMAX;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_DEPTH) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_DEPTH)) {
        os += (first ? "" : " | ");
        os += "SPACE_TO_DEPTH";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_DEPTH;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SVDF) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SVDF)) {
        os += (first ? "" : " | ");
        os += "SVDF";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SVDF;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::TANH) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::TANH)) {
        os += (first ? "" : " | ");
        os += "TANH";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::TANH;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::OEM_OPERATION) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::OEM_OPERATION)) {
        os += (first ? "" : " | ");
        os += "OEM_OPERATION";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::OEM_OPERATION;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::BATCH_TO_SPACE_ND) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::BATCH_TO_SPACE_ND)) {
        os += (first ? "" : " | ");
        os += "BATCH_TO_SPACE_ND";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::BATCH_TO_SPACE_ND;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::DIV) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::DIV)) {
        os += (first ? "" : " | ");
        os += "DIV";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::DIV;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::MEAN) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::MEAN)) {
        os += (first ? "" : " | ");
        os += "MEAN";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::MEAN;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::PAD) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::PAD)) {
        os += (first ? "" : " | ");
        os += "PAD";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::PAD;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_BATCH_ND) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_BATCH_ND)) {
        os += (first ? "" : " | ");
        os += "SPACE_TO_BATCH_ND";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_BATCH_ND;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SQUEEZE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SQUEEZE)) {
        os += (first ? "" : " | ");
        os += "SQUEEZE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SQUEEZE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::STRIDED_SLICE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::STRIDED_SLICE)) {
        os += (first ? "" : " | ");
        os += "STRIDED_SLICE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::STRIDED_SLICE;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::SUB) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::SUB)) {
        os += (first ? "" : " | ");
        os += "SUB";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::SUB;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::OperationType::TRANSPOSE) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::OperationType::TRANSPOSE)) {
        os += (first ? "" : " | ");
        os += "TRANSPOSE";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::OperationType::TRANSPOSE;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_1::OperationType o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::ADD) {
        return "ADD";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::AVERAGE_POOL_2D) {
        return "AVERAGE_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::CONCATENATION) {
        return "CONCATENATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::CONV_2D) {
        return "CONV_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTHWISE_CONV_2D) {
        return "DEPTHWISE_CONV_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTH_TO_SPACE) {
        return "DEPTH_TO_SPACE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::DEQUANTIZE) {
        return "DEQUANTIZE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::EMBEDDING_LOOKUP) {
        return "EMBEDDING_LOOKUP";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::FLOOR) {
        return "FLOOR";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::FULLY_CONNECTED) {
        return "FULLY_CONNECTED";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::HASHTABLE_LOOKUP) {
        return "HASHTABLE_LOOKUP";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::L2_NORMALIZATION) {
        return "L2_NORMALIZATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::L2_POOL_2D) {
        return "L2_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION) {
        return "LOCAL_RESPONSE_NORMALIZATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::LOGISTIC) {
        return "LOGISTIC";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::LSH_PROJECTION) {
        return "LSH_PROJECTION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::LSTM) {
        return "LSTM";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::MAX_POOL_2D) {
        return "MAX_POOL_2D";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::MUL) {
        return "MUL";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RELU) {
        return "RELU";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RELU1) {
        return "RELU1";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RELU6) {
        return "RELU6";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RESHAPE) {
        return "RESHAPE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RESIZE_BILINEAR) {
        return "RESIZE_BILINEAR";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::RNN) {
        return "RNN";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SOFTMAX) {
        return "SOFTMAX";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_DEPTH) {
        return "SPACE_TO_DEPTH";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SVDF) {
        return "SVDF";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::TANH) {
        return "TANH";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::OEM_OPERATION) {
        return "OEM_OPERATION";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::BATCH_TO_SPACE_ND) {
        return "BATCH_TO_SPACE_ND";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::DIV) {
        return "DIV";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::MEAN) {
        return "MEAN";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::PAD) {
        return "PAD";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_BATCH_ND) {
        return "SPACE_TO_BATCH_ND";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SQUEEZE) {
        return "SQUEEZE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::STRIDED_SLICE) {
        return "STRIDED_SLICE";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::SUB) {
        return "SUB";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::OperationType::TRANSPOSE) {
        return "TRANSPOSE";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Capabilities& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".float32Performance = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.float32Performance);
    os += ", .quantized8Performance = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.quantized8Performance);
    os += ", .relaxedFloat32toFloat16Performance = ";
    os += ::android::hardware::neuralnetworks::V1_0::toString(o.relaxedFloat32toFloat16Performance);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_1::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_1::Capabilities& rhs) {
    if (lhs.float32Performance != rhs.float32Performance) {
        return false;
    }
    if (lhs.quantized8Performance != rhs.quantized8Performance) {
        return false;
    }
    if (lhs.relaxedFloat32toFloat16Performance != rhs.relaxedFloat32toFloat16Performance) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_1::Capabilities& lhs, const ::android::hardware::neuralnetworks::V1_1::Capabilities& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Operation& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".type = ";
    os += ::android::hardware::neuralnetworks::V1_1::toString(o.type);
    os += ", .inputs = ";
    os += ::android::hardware::toString(o.inputs);
    os += ", .outputs = ";
    os += ::android::hardware::toString(o.outputs);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hardware::neuralnetworks::V1_1::Operation& lhs, const ::android::hardware::neuralnetworks::V1_1::Operation& rhs) {
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

static inline bool operator!=(const ::android::hardware::neuralnetworks::V1_1::Operation& lhs, const ::android::hardware::neuralnetworks::V1_1::Operation& rhs){
    return !(lhs == rhs);
}

static inline std::string toString(const ::android::hardware::neuralnetworks::V1_1::Model& o) {
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
    os += ", .relaxComputationFloat32toFloat16 = ";
    os += ::android::hardware::toString(o.relaxComputationFloat32toFloat16);
    os += "}"; return os;
}

// operator== and operator!= are not generated for Model

template<>
inline std::string toString<::android::hardware::neuralnetworks::V1_1::ExecutionPreference>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hardware::neuralnetworks::V1_1::ExecutionPreference> flipped = 0;
    bool first = true;
    if ((o & ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::LOW_POWER) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::ExecutionPreference::LOW_POWER)) {
        os += (first ? "" : " | ");
        os += "LOW_POWER";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::LOW_POWER;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::FAST_SINGLE_ANSWER) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::ExecutionPreference::FAST_SINGLE_ANSWER)) {
        os += (first ? "" : " | ");
        os += "FAST_SINGLE_ANSWER";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::FAST_SINGLE_ANSWER;
    }
    if ((o & ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::SUSTAINED_SPEED) == static_cast<int32_t>(::android::hardware::neuralnetworks::V1_1::ExecutionPreference::SUSTAINED_SPEED)) {
        os += (first ? "" : " | ");
        os += "SUSTAINED_SPEED";
        first = false;
        flipped |= ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::SUSTAINED_SPEED;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hardware::neuralnetworks::V1_1::ExecutionPreference o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::LOW_POWER) {
        return "LOW_POWER";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::FAST_SINGLE_ANSWER) {
        return "FAST_SINGLE_ANSWER";
    }
    if (o == ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::SUSTAINED_SPEED) {
        return "SUSTAINED_SPEED";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}


}  // namespace V1_1
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
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_1::OperationType, 39> hidl_enum_values<::android::hardware::neuralnetworks::V1_1::OperationType> = {
    ::android::hardware::neuralnetworks::V1_1::OperationType::ADD,
    ::android::hardware::neuralnetworks::V1_1::OperationType::AVERAGE_POOL_2D,
    ::android::hardware::neuralnetworks::V1_1::OperationType::CONCATENATION,
    ::android::hardware::neuralnetworks::V1_1::OperationType::CONV_2D,
    ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTHWISE_CONV_2D,
    ::android::hardware::neuralnetworks::V1_1::OperationType::DEPTH_TO_SPACE,
    ::android::hardware::neuralnetworks::V1_1::OperationType::DEQUANTIZE,
    ::android::hardware::neuralnetworks::V1_1::OperationType::EMBEDDING_LOOKUP,
    ::android::hardware::neuralnetworks::V1_1::OperationType::FLOOR,
    ::android::hardware::neuralnetworks::V1_1::OperationType::FULLY_CONNECTED,
    ::android::hardware::neuralnetworks::V1_1::OperationType::HASHTABLE_LOOKUP,
    ::android::hardware::neuralnetworks::V1_1::OperationType::L2_NORMALIZATION,
    ::android::hardware::neuralnetworks::V1_1::OperationType::L2_POOL_2D,
    ::android::hardware::neuralnetworks::V1_1::OperationType::LOCAL_RESPONSE_NORMALIZATION,
    ::android::hardware::neuralnetworks::V1_1::OperationType::LOGISTIC,
    ::android::hardware::neuralnetworks::V1_1::OperationType::LSH_PROJECTION,
    ::android::hardware::neuralnetworks::V1_1::OperationType::LSTM,
    ::android::hardware::neuralnetworks::V1_1::OperationType::MAX_POOL_2D,
    ::android::hardware::neuralnetworks::V1_1::OperationType::MUL,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RELU,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RELU1,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RELU6,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RESHAPE,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RESIZE_BILINEAR,
    ::android::hardware::neuralnetworks::V1_1::OperationType::RNN,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SOFTMAX,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_DEPTH,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SVDF,
    ::android::hardware::neuralnetworks::V1_1::OperationType::TANH,
    ::android::hardware::neuralnetworks::V1_1::OperationType::OEM_OPERATION,
    ::android::hardware::neuralnetworks::V1_1::OperationType::BATCH_TO_SPACE_ND,
    ::android::hardware::neuralnetworks::V1_1::OperationType::DIV,
    ::android::hardware::neuralnetworks::V1_1::OperationType::MEAN,
    ::android::hardware::neuralnetworks::V1_1::OperationType::PAD,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SPACE_TO_BATCH_ND,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SQUEEZE,
    ::android::hardware::neuralnetworks::V1_1::OperationType::STRIDED_SLICE,
    ::android::hardware::neuralnetworks::V1_1::OperationType::SUB,
    ::android::hardware::neuralnetworks::V1_1::OperationType::TRANSPOSE,
};
}  // namespace details
}  // namespace hardware
}  // namespace android

namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hardware::neuralnetworks::V1_1::ExecutionPreference, 3> hidl_enum_values<::android::hardware::neuralnetworks::V1_1::ExecutionPreference> = {
    ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::LOW_POWER,
    ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::FAST_SINGLE_ANSWER,
    ::android::hardware::neuralnetworks::V1_1::ExecutionPreference::SUSTAINED_SPEED,
};
}  // namespace details
}  // namespace hardware
}  // namespace android
#endif //ppn


#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_TYPES_H
