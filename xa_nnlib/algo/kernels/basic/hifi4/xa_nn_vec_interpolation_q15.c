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

WORD32 xa_nn_vec_interpolation_q15(WORD16 * __restrict__ p_out,
         const WORD16 * __restrict__ p_ifact,
         const WORD16 * __restrict__ p_inp1,
         const WORD16 * __restrict__ p_inp2,
         WORD32 num_elements)
{
    XA_NNLIB_ARG_CHK_PTR(p_out,    -1);
    XA_NNLIB_ARG_CHK_PTR(p_ifact,  -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1,   -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2,   -1);
    XA_NNLIB_ARG_CHK_COND(((num_elements&3) != 0), -1);

    int i;
    ae_f16x4 *p_fi = (ae_f16x4 *)p_ifact;
    ae_f16x4 *p_si = (ae_f16x4 *)p_inp1;
    ae_f16x4 *p_ti = (ae_f16x4 *)p_inp2;
    ae_f16x4 *p_r  = (ae_f16x4 *)p_out, one;
    one = AE_MOVDA16(0x7fff);

    for(i=0; i<num_elements>>2; i++)
    {
        p_si[i] = p_r[i]  = AE_ADD16S(AE_MULFP16X4S(p_fi[i], p_si[i]), AE_MULFP16X4S(one-p_fi[i], p_ti[i]));
    }

    return 0;
}

