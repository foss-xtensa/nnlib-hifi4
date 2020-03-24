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
#ifndef __XA_NN_BASIC_STATE_H__
#define __XA_NN_BASIC_STATE_H__


#define ALIGNMENT   8   /* 8 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define LIMIT(input, min, max) \
    input = XT_MAX(min, XT_MIN(max, input));

#define MultiplyByQuantizedMultiplierSmallerThanOneExp(prod, val, multiplier, lsh) {\
    ae_int64 temp64_h, temp64_l;\
    prod = AE_MULFP32X2RAS(val, multiplier);\
    temp64_h = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH((ae_int32x2)prod, ZERO));\
    temp64_l = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL((ae_int32x2)prod, ZERO));\
    temp64_h = AE_SLAA64S(temp64_h, lsh);\
    temp64_l = AE_SLAA64S(temp64_l, lsh);\
    prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}

// This is the another implementation (32x32--> 64 --> shift --> symm rounding
#define MultiplyByQuantizedMultiplierSmallerThanOneExp_NEW(prod, val, multiplier, lsh) {\
    ae_int64 temp64_h, temp64_l;\
    temp64_h = AE_MULF32S_HH(val, multiplier);\
    temp64_l = AE_MULF32S_LL(val, multiplier);\
    temp64_h = AE_SLAA64S(temp64_h, lsh);\
    temp64_l = AE_SLAA64S(temp64_l, lsh);\
    prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}

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

#define PRIME_8X4U(p_char, tmp) \
    int offset_##p_char = 0, ls_##p_char, rs_##p_char; \
    rs_##p_char = 0; \
    ls_##p_char = 64; \
    tmp = AE_ZERO16(); \
    while(((unsigned int)p_char + offset_##p_char) & 3) {\
        ae_int16x4 tmp2 = AE_MOVDA16(*(((const UWORD8 *)p_char)+offset_##p_char)); \
        tmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(tmp2), 48)); \
        tmp = AE_MOVINT16X4_FROMINT64(AE_SLAI64(AE_MOVINT64_FROMINT16X4(tmp), 16)); \
        tmp = AE_OR16(tmp, tmp2); \
        rs_##p_char += 16;  \
        ls_##p_char -= 16; \
        offset_##p_char++; \
    }\
    tmp = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(tmp), ls_##p_char)); \

#define AE_LA8X4U_IP(d, a, p) { \
    ae_int16x4 d_tmp, d_tmp2; \
    d_tmp = AE_L8X4F_I(p+offset_##p, 0); \
    p += 4; \
    d_tmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_tmp), rs_##p+8)); \
    d = AE_OR16(a, d_tmp2); \
    a = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_tmp), ls_##p-8)); \
}

#endif /* #ifndef __XA_NN_BASIC_STATE_H__ */

