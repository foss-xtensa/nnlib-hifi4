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
#include "xa_nnlib_common.h"

void xa_nn_elm_mul_16x16_16(WORD16 * __restrict__ p_out,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD16 * __restrict__ p_inp2,
                      WORD32 num_elm)
{
    #pragma aligned(p_inp1, 8)
    #pragma aligned(p_inp2, 8)
    int i;
    ae_f16x4 *inp1 = (ae_f16x4 *)p_inp1;
    ae_f16x4 *inp2 = (ae_f16x4 *)p_inp2;
    ae_f16x4 *out = (ae_f16x4 *)p_out;

    for(i=0;i < num_elm>>2;i++)
    {
        out[i]  = AE_MULFP16X4S(inp1[i], inp2[i]);
    }
}
