/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "xa_nn_basic_state.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp, inp1, multiplier, l_shift, right_shift, out_off) \
    AE_MUL2P32X4S(inp, inp1, inp, inp1, l_shift, l_shift); \
    AE_MULF2P32X4RAS(inp, inp1, inp, inp1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);\
    inp1 = AE_SRAA32SYMS(inp1, right_shift);\
    out = AE_SAT16X4(inp, inp1); \
    out = AE_ADD16S(AE_MOVDA16(out_off), out); \
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); 

#define PACK_32X2(dst1, src1, src2) \
dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e0c0a08, 0x06040200)));

WORD32 xa_nn_elm_quantize_asym16s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
    return -1;
}

