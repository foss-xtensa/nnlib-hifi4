/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
#ifndef __XA_NNLIB_COMMON_H__
#define __XA_NNLIB_COMMON_H__

/* FOR HIFI4 NN LIB CROSS-COMPILATION ON HIFI5 */
#if XCHAL_HAVE_HIFI5
#define ALIGN_REGISTER_TYPE ae_valign

#define PRIME_8X4U(p_char, tmp) \
    tmp = AE_LA64_PP(p_char) \

#else
#define ALIGN_REGISTER_TYPE ae_int16x4

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
#endif

#define PRIME_8X4F(p_char, tmp) \
    int offset_##p_char = 0, ls_##p_char, rs_##p_char; \
    rs_##p_char = 0; \
    ls_##p_char = 64; \
    tmp = AE_ZERO16(); \
    while(((unsigned int)p_char + offset_##p_char) & 3) {\
        ae_int16x4 tmp2 = AE_MOVDA16(((short)*(p_char+offset_##p_char)) << 8); \
        tmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(tmp2), 48)); \
        tmp = AE_MOVINT16X4_FROMINT64(AE_SLAI64(AE_MOVINT64_FROMINT16X4(tmp), 16)); \
        tmp = AE_OR16(tmp, tmp2); \
        rs_##p_char += 16;  \
        ls_##p_char -= 16; \
        offset_##p_char++; \
    }\
    tmp = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(tmp), ls_##p_char)); \

#define AE_LA8X4F_IP(d, a, p) { \
    ae_int16x4 d_tmp, d_tmp2; \
    d_tmp = AE_L8X4F_I(p+offset_##p, 0); \
    p += 4; \
    d_tmp2 = AE_MOVINT16X4_FROMINT64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(d_tmp), rs_##p)); \
    d = AE_OR16(a, d_tmp2); \
    a = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_MOVINT64_FROMINT16X4(d_tmp), ls_##p)); \
}

/* FOR HIFI4 NN LIB CROSS-COMPILATION ON HIFI3Z */
#ifndef AE_ADD32S_HL_LH                                                                                                                                                                                                                       
#define AE_ADD32S_HL_LH(sum_exp,sum_exp_) AE_ADD32S(sum_exp, AE_SEL32_LH(sum_exp_, sum_exp_)); 
#endif

#ifndef AE_ADDCIRC16X4_XC
/* Alignment requirement of 2 bytes */
#define AE_ADDCIRC16X4_XC(ptr, inc) \
{ \
  ae_int16x4 dummy; \
  AE_L16_XC(dummy, (ae_int16*)ptr, inc); \
}
#else
#define HW_AE_ADDCIRC16X4_XC
#endif

#ifndef AE_MULA16_00
#define AE_MULA16_00(q0, d0, d1) \
{ \
  ae_int32x2 d2,d3; \
  AE_MUL16X4(d2,d3,d0,d1); \
  ae_int64 o; \
  o = AE_MOVINT64_FROMINT32X2(d3); \
  o = AE_SLAI64(o, 32); \
  o = AE_SRAI64(o, 32); \
  q0 = AE_ADD64(q0, o); \
}
#endif

/* FOR HIFI4 NN LIB CROSS-COMPILATION ON FUSION F1 */
#ifndef AE_MULAAAAQ16
#define AE_MULAAAAQ16(q0, d0, d1) \
{ \
  ae_int32x2 d2,d3; \
  ae_int16x4 d = 1; \
  AE_MUL16X4(d3,d2,d0,d1); \
  d3 = AE_ADD32S(d3, d2); \
  AE_MULAAD32X16_H0_L1(q0,d3,d); \
}
#endif

#ifndef AE_MOVBA
#define AE_MOVBA(a) (xtbool)((unsigned int)a&1)
#endif

#ifndef  AE_MULFP32X16X2RS_L_S2
#define  AE_MULFP32X16X2RS_L_S2 AE_MULFP32X16X2RS_L
#endif

/* FOR HIFI4 NN LIB CROSS-COMPILATION ON HIFI3 */
#ifndef AE_L8X4F_IP
#define AE_L8X4F_IP(d, p, inc) \
{ \
  ae_int32 *p32; \
  ae_int32x2 d32x2; \
  ae_int16x4 d16x4_0, d16x4_1; \
  p32 = (ae_int32 *)p; \
  AE_L32_IP(d32x2, p32, inc); \
  p = (WORD8 *)p32; \
  d16x4_0 = AE_MOVINT16X4_FROMINT32X2(d32x2); \
  d16x4_1 = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(d32x2, 8)); \
  d = AE_SEL16_7362(d16x4_0, d16x4_1); \
  d = AE_SHORTSWAP(d); \
  d = AE_AND16(d, AE_MOVDA16(0xff00)); \
}
#endif

#ifndef AE_L8X4F_I
static inline ae_int16x4 AE_L8X4F_I(const WORD8 *p, int inc)
{
  ae_int16x4 d;
  ae_int32 d32x2;
  ae_int16x4 d16x4_0, d16x4_1;
  d32x2 = AE_L32_X((const ae_int32 *)p, inc);
  d16x4_0 = AE_MOVINT16X4_FROMINT32X2(d32x2);
  d16x4_1 = AE_MOVINT16X4_FROMINT32X2(AE_SLAI32(d32x2, 8));
  d = AE_SEL16_7362(d16x4_0, d16x4_1);
  d = AE_SHORTSWAP(d);
  d = AE_AND16(d, AE_MOVDA16(0xff00));
  return d;
}
#endif

#ifndef AE_SEL16_7531
#define AE_SEL16_7531(d0, d1) \
  AE_TRUNC16X4F32(AE_MOVINT32X2_FROMINT16X4(d0), AE_MOVINT32X2_FROMINT16X4(d1))
#endif

/* FOR HIFI5-RI6 NN LIB CROSS-COMPILATION ON HIFI5-RI5 */
#ifndef AE_S8X4U_XP
#define STORE_16x4x2_8x4x2(outbuf_0, outbuf_1, pOut, out_offset) \
{ \
  ae_int8x8 out32; \
  out32 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(outbuf_0), AE_MOVINT8X8_FROMINT16X4(outbuf_1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406))); \
  AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32), (ae_int32 *)pOut, out_offset); \
  AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32), (ae_int32 *)pOut, out_offset); \
}
#define STORE_16x4_8x4(outbuf, pOut, out_offset) \
{ \
  ae_int8x8 out32; \
  out32 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(outbuf), AE_MOVINT8X8_FROMINT16X4(outbuf), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406))); \
  AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32), (ae_int32 *)pOut, out_offset); \
}
#else
#define STORE_16x4x2_8x4x2(outbuf_0, outbuf_1, pOut, out_offset) \
{ \
  AE_S8X4U_XP(outbuf_0, (ae_int32 *)pOut, out_offset); \
  AE_S8X4U_XP(outbuf_1, (ae_int32 *)pOut, out_offset); \
}
#define STORE_16x4_8x4(outbuf, pOut, out_offset) \
{ \
  AE_S8X4U_XP(outbuf, (ae_int32 *)pOut, out_offset); \
}
#endif

#endif /* __XA_NNLIB_COMMON_H__ */
