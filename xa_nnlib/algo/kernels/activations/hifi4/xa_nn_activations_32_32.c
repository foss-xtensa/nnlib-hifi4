/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common.h"
#include "NatureDSP_Signal_math.h"

WORD32 xa_nn_vec_sigmoid_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_sigmoid32x32(p_out, p_vec, vec_length);
  return 0;
}

WORD32 xa_nn_vec_tanh_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_tanh32x32(p_out, p_vec, vec_length);
  return 0;
}

#define MAX_WORD32 (int)(0x7FFFFFFF)
WORD32 xa_nn_vec_relu_std_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
    vec_relu32x32(p_out, p_vec, (MAX_WORD32), vec_length);
    return 0;
}
WORD32 xa_nn_vec_relu_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       threshold,                    /* threshold, Q16.15 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_relu32x32(p_out, p_vec, threshold, vec_length);
  return 0;
}

WORD32 xa_nn_vec_relu1_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_relu32x32(p_out, p_vec, (1<<15), vec_length); // threshold=1, Q16.15
  return 0;
}

WORD32 xa_nn_vec_relu6_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_relu32x32(p_out, p_vec, (6<<15), vec_length); // threshold=6, Q16.15
  return 0;
}

WORD32 xa_nn_vec_softmax_32_32(
    WORD32       * __restrict__ p_out,         /* result, Q16.15 */
    const WORD32 * __restrict__ p_vec,         /* input data, Q6.25 */
    WORD32       vec_length)                   /* length of vectors */
{
  vec_softmax32x32(p_out, p_vec, vec_length);
  return 0;
}

