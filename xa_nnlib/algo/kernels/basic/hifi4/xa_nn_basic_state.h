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
#ifndef __XA_NN_BASIC_STATE_H__
#define __XA_NN_BASIC_STATE_H__


#define ALIGNMENT   8   /* 8 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define LIMIT(input, min, max) \
    input = XT_MAX(min, XT_MIN(max, input));

#define CLAMP_VAL(out, val, min, max){\
    ae_f32x2 temp_max;\
    temp_max = AE_MAX32(min, val);\
    out = AE_MIN32(temp_max, max);\
}

#define STORE_8X4_FROM_32X4(out_ptr, val12, val34){\
    int o1, o2, o3, o4;\
    o1 = AE_MOVAD32_H(val12);\
    o2 = AE_MOVAD32_L(val12);\
    o3 = AE_MOVAD32_H(val34);\
    o4 = AE_MOVAD32_L(val34);\
    *out_ptr++ = (UWORD8)o1;\
    *out_ptr++ = (UWORD8)o2;\
    *out_ptr++ = (UWORD8)o3;\
    *out_ptr++ = (UWORD8)o4;\
}

#endif /* #ifndef __XA_NN_BASIC_STATE_H__ */

