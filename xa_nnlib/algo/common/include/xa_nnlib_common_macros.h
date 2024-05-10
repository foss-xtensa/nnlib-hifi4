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

#ifndef __XA_NNLIB_COMMON_MACROS_H__
#define __XA_NNLIB_COMMON_MACROS_H__

#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
#include <xtensa/config/core-isa.h>
#endif
#include <stddef.h>
#include "xa_nnlib_quant_macros.h"
#include "xa_nnlib_common_internal.h"

#ifndef NULL
#define NULL (void *)0
#endif /* NULL */

#define MAX(a, b)   (((a) > (b)) ? (a) : (b))

#if XCHAL_HAVE_HIFI1

#if (XCHAL_HW_VERSION >= 281090)

#define MEMCPY_8b(out, inp, N) \
{ \
    unsigned int itr; \
    ae_int8x8 di0; \
    ae_int8x8 *pae_i = (ae_int8x8 *) inp ;\
    ae_int8x8 *pae_o = (ae_int8x8 *) out ;\
    ae_valign o_a = AE_ZALIGN64();        \
    ae_valign i_a = AE_LA64_PP(pae_i);    \
                                          \
    for(itr = 0; itr < (unsigned int)(N>>3); itr++) { \
        AE_LA8X8_IP(di0, i_a, pae_i);     \
        AE_SA8X8_IP(di0, o_a, pae_o);     \
    }                                     \
    AE_LAV8X8_XP(di0, i_a, pae_i, (N&7)); \
    AE_SAV8X8_XP(di0, o_a, pae_o, (N&7)); \
    AE_SA64POS_FP(o_a, pae_o);            \
}
#define MEMCPY_8bx2(out0, out1, inp0, inp1, N) \
{ \
    unsigned int itr; \
    ae_int8x8 di0_0,di0_1; \
    ae_int8x8 *pae0_i = (ae_int8x8 *) inp0 ;\
    ae_int8x8 *pae0_o = (ae_int8x8 *) out0 ;\
    ae_int8x8 *pae1_i = (ae_int8x8 *) inp1 ;\
    ae_int8x8 *pae1_o = (ae_int8x8 *) out1 ;\
    ae_valign o0_a = AE_ZALIGN64();      \
    ae_valign i0_a = AE_LA64_PP(pae0_i); \
    ae_valign o1_a = AE_ZALIGN64();      \
    ae_valign i1_a = AE_LA64_PP(pae1_i); \
                                         \
    for(itr = 0; itr < (unsigned int)(N>>3); itr++) { \
        AE_LA8X8_IP(di0_0, i0_a, pae0_i);     \
        AE_SA8X8_IP(di0_0, o0_a, pae0_o);     \
        AE_LA8X8_IP(di0_1, i1_a, pae1_i);     \
        AE_SA8X8_IP(di0_1, o1_a, pae1_o);     \
    }                                         \
    AE_LAV8X8_XP(di0_0, i0_a, pae0_i, (N&7)); \
    AE_SAV8X8_XP(di0_0, o0_a, pae0_o, (N&7)); \
    AE_SA64POS_FP(o0_a, pae0_o);              \
    AE_LAV8X8_XP(di0_1, i1_a, pae1_i, (N&7)); \
    AE_SAV8X8_XP(di0_1, o1_a, pae1_o, (N&7)); \
    AE_SA64POS_FP(o1_a, pae1_o);              \
}

#define MEMCPY_2D_8b_CONT_OUT(dst, src, rows, cols, inp_row_stride){    \
                                                                        \
    void *p_src = src;                                                  \
    void *p_dst = dst;                                                  \
                                                                        \
    for(int row = 0; row < rows; row++){                                \
        MEMCPY_8b(p_dst, p_src, cols);                                  \
        p_src += inp_row_stride;                                        \
        p_dst += cols;                                                  \
    }                                                                   \
}

#define DUAL_MEMCPY_2D_8b_CONT_OUT(dst0, dst1, src0, src1, rows, cols, inp_row_stride){ \
                                                                                        \
    void *p_src0 = src0;                                                                \
    void *p_src1 = src1;                                                                \
                                                                                        \
    void *p_dst0 = dst0;                                                                \
    void *p_dst1 = dst1;                                                                \
                                                                                        \
    for(int row = 0; row < rows; row++){                                                \
        MEMCPY_8bx2(p_dst0, p_dst1, p_src0, p_src1, cols);                              \
                                                                                        \
        p_src0 += inp_row_stride;                                                       \
        p_src1 += inp_row_stride;                                                       \
                                                                                        \
        p_dst0 += cols;                                                                 \
        p_dst1 += cols;                                                                 \
    }                                                                                   \
}

#define MEMCPY_2D_8b_CONT_INP(dst, src, rows, cols, out_row_stride){    \
                                                                        \
    void *p_src = src;                                                  \
    void *p_dst = dst;                                                  \
                                                                        \
    for(int row = 0; row < rows; row++){                                \
        MEMCPY_8b(p_dst, p_src, cols);                                  \
        p_src += cols;                                                  \
        p_dst += out_row_stride;                                        \
    }                                                                   \
}

#define DUAL_MEMCPY_2D_8b_CONT_INP(dst0, dst1, src0, src1, rows, cols, out_row_stride){ \
                                                                                        \
    void *p_src0 = src0;                                                                \
    void *p_src1 = src1;                                                                \
                                                                                        \
    void *p_dst0 = dst0;                                                                \
    void *p_dst1 = dst1;                                                                \
                                                                                        \
    for(int row = 0; row < rows; row++){                                                \
        MEMCPY_8bx2(p_dst0, p_dst1, p_src0, p_src1, cols);                              \
                                                                                        \
        p_src0 += cols;                                                                 \
        p_src1 += cols;                                                                 \
                                                                                        \
        p_dst0 += out_row_stride;                                                       \
        p_dst1 += out_row_stride;                                                       \
    }                                                                                   \
}

#else //(XCHAL_HW_VERSION >= 281090)
/* Macros for memcpy */
#define MEMCPY_8b(out, inp, N) \
{ \
  unsigned int itr; \
  ae_int16x4 di0; \
  ae_valign i_a, o_a; \
  WORD8 *inp_ptr = (WORD8 *)inp ;\
  WORD8 *out_ptr = (WORD8 *)out ;\
  if( (((uintptr_t)inp_ptr & 0x1) == 0 ) && (((uintptr_t)out_ptr & 0x1) == 0 ))\
  {\
    ae_int16x4 *pae_i = (ae_int16x4 *) inp_ptr ;\
    ae_int16x4 *pae_o = (ae_int16x4 *) out_ptr ;\
    i_a = AE_LA64_PP((void*)pae_i);\
    o_a = AE_ZALIGN64();\
    for( itr = 0; itr < (unsigned int)(N>>3); itr++ )\
    {\
      AE_LA16X4_IP(di0, i_a, pae_i);\
      AE_SA16X4_IP(di0, o_a, pae_o);\
    }\
    AE_SA64POS_FP(o_a, pae_o); \
    unsigned int rem = (N&0x7);\
    for( itr = 0; itr < (unsigned int)(rem>>1); itr++)\
    {\
      AE_L16_IP(di0, (ae_int16 *)pae_i, 2);\
      AE_S16_0_IP(di0, (ae_int16 *)pae_o, 2);\
    }\
    if(rem&0x1)\
    {\
      AE_L8S_IP(di0, (WORD8 *)pae_i, 1);\
      AE_S8_0_IP_HIFI1(di0, (WORD8 *)pae_o, 1);\
    }\
  }\
  else\
  {\
    WORD8 *pae_i, *pae_o;\
    pae_i = (WORD8 *)(inp_ptr); \
    pae_o = (WORD8 *)(out_ptr); \
  i_a = AE_LA64_PP((void*)pae_i); \
  o_a = AE_ZALIGN64(); \
  for(itr = 0; itr < (unsigned int)((N)>>2); itr++) \
  { \
    AE_LA8X4S_IP(di0, i_a, pae_i); \
    AE_SA8X4U_IP(di0, o_a, (ae_int32*)pae_o); \
  } \
  AE_SA64POS_FP(o_a, pae_o); \
  for(itr = 0; itr < (unsigned int)(N&3); itr++) \
  { \
    AE_L8S_IP(di0, pae_i, 1); \
    AE_S8_0_IP_HIFI1(di0, pae_o, 1); \
  } \
  }\
}

#define MEMCPY_2D_8b_CONT_OUT(out0, inp0, rows, cols, inp_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int16x4 di0_0; \
  ae_valign in0_a, out0_a; \
  WORD8 *pae_out0 = (WORD8 *)(out0); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    WORD8 *pae_in0 = (WORD8 *)(&(inp0)[itr_r * inp_row_offset]); \
    in0_a = AE_LA64_PP(pae_in0);\
    out0_a = AE_ZALIGN64();\
    if( (((uintptr_t)pae_in0 & 0x1) == 0 ) && (((uintptr_t)pae_out0 & 0x1) == 0 ))\
    {\
      for( itr_c = 0; itr_c < (cols>>3); itr_c++ )\
      {\
        AE_LA16X4_IP(di0_0, in0_a, (ae_int16x4 *)pae_in0);\
        AE_SA16X4_IP(di0_0, out0_a, (ae_int16x4 *)pae_out0);\
      }\
      AE_SA64POS_FP(out0_a, pae_out0); \
      int rem = (cols&0x7);\
      for( itr_c = 0; itr_c < (rem>>1); itr_c++)\
      {\
        AE_L16_IP(di0_0, (ae_int16 *)pae_in0, 2);\
        AE_S16_0_IP(di0_0, (ae_int16 *)pae_out0, 2);\
      }\
      if(rem&0x1)\
      {\
        AE_L8S_IP(di0_0, (WORD8 *)pae_in0, 1);\
        AE_S8_0_IP_HIFI1(di0_0, (WORD8 *)pae_out0, 1);\
      }\
    }\
    else\
    {\
    for(itr_c = 0; itr_c < ((cols)>>2); itr_c++) \
    { \
      AE_LA8X4S_IP(di0_0, in0_a, pae_in0); \
      AE_SA8X4U_IP(di0_0, out0_a, (ae_int32*)pae_out0); \
    } \
    AE_SA64POS_FP(out0_a, pae_out0); \
    for(itr_c = 0; itr_c < (cols&3); itr_c++) \
    { \
      AE_L8S_IP(di0_0, pae_in0, 1); \
      AE_S8_0_IP_HIFI1(di0_0, pae_out0, 1); \
    } \
  } \
  } \
}

#define DUAL_MEMCPY_2D_8b_CONT_OUT(out0, out1, inp0, inp1, rows, cols, inp_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int16x4 di0_0; \
  ae_int16x4 di1_0; \
  WORD8 *pae_in0, *pae_out0; \
  ae_valign in0_a, out0_a; \
  WORD8 *pae_in1, *pae_out1; \
  ae_valign in1_a, out1_a; \
  pae_out0 = (WORD8 *)(out0); \
  out0_a = AE_ZALIGN64(); \
  pae_out1 = (WORD8 *)(out1); \
  out1_a = AE_ZALIGN64(); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_in0 = (WORD8 *)(&(inp0)[itr_r * inp_row_offset]); \
    in0_a = AE_LA64_PP(pae_in0); \
    pae_in1 = (WORD8 *)(&(inp1)[itr_r * inp_row_offset]); \
    in1_a = AE_LA64_PP(pae_in1); \
__Pragma("no_unroll") \
    for(itr_c = 0; itr_c < ((cols)>>2); itr_c++) \
    { \
      AE_LA8X4S_IP(di0_0, in0_a, pae_in0); \
      AE_SA8X4U_IP(di0_0, out0_a, (ae_int32*)pae_out0); \
      AE_LA8X4S_IP(di1_0, in1_a, pae_in1); \
      AE_SA8X4U_IP(di1_0, out1_a, (ae_int32*)pae_out1); \
    } \
    AE_SA64POS_FP(out0_a, pae_out0); \
    AE_SA64POS_FP(out1_a, pae_out1); \
    for(itr_c = 0; itr_c < (cols&3); itr_c++) \
    { \
      AE_L8S_IP(di0_0, pae_in0, 1); \
      AE_S8_0_IP_HIFI1(di0_0, pae_out0, 1); \
      AE_L8S_IP(di1_0, pae_in1, 1); \
      AE_S8_0_IP_HIFI1(di1_0, pae_out1, 1); \
    } \
  } \
}

#define MEMCPY_2D_8b_CONT_INP(out0, inp0, rows, cols, out_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int16x4 di0_0; \
  WORD8 *pae_in0, *pae_out0; \
  ae_valign in0_a, out0_a; \
  pae_in0 = (WORD8 *)(inp0); \
  in0_a = AE_LA64_PP(pae_in0); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_out0 = (WORD8 *)(&(out0)[itr_r * out_row_offset]); \
    out0_a = AE_ZALIGN64(); \
    in0_a = AE_LA64_PP(pae_in0); \
    for(itr_c = 0; itr_c < ((cols)>>2); itr_c++) \
    { \
      AE_LA8X4S_IP(di0_0, in0_a, pae_in0); \
      AE_SA8X4U_IP(di0_0, out0_a, (ae_int32*)pae_out0); \
    } \
    AE_SA64POS_FP(out0_a, pae_out0); \
    for(itr_c = 0; itr_c < (cols&3); itr_c++) \
    { \
      AE_L8S_IP(di0_0, pae_in0, 1); \
      AE_S8_0_IP_HIFI1(di0_0, pae_out0, 1); \
    } \
  } \
}

#define DUAL_MEMCPY_2D_8b_CONT_INP(out0, out1, inp0, inp1, rows, cols, out_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int16x4 di0_0; \
  ae_int16x4 di1_0; \
  WORD8 *pae_in0, *pae_out0; \
  ae_valign in0_a, out0_a; \
  WORD8 *pae_in1, *pae_out1; \
  ae_valign in1_a, out1_a; \
  pae_in0 = (WORD8 *)(inp0); \
  in0_a = AE_LA64_PP(pae_in0); \
  pae_in1 = (WORD8 *)(inp1); \
  in1_a = AE_LA64_PP(pae_in1); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_out0 = (WORD8 *)(&(out0)[itr_r * out_row_offset]); \
    out0_a = AE_ZALIGN64(); \
    pae_out1 = (WORD8 *)(&(out1)[itr_r * out_row_offset]); \
    out1_a = AE_ZALIGN64(); \
    in0_a = AE_LA64_PP(pae_in0); \
    in1_a = AE_LA64_PP(pae_in1); \
__Pragma("no_unroll") \
    for(itr_c = 0; itr_c < ((cols)>>2); itr_c++) \
    { \
      AE_LA8X4S_IP(di0_0, in0_a, pae_in0); \
      AE_SA8X4U_IP(di0_0, out0_a, (ae_int32*)pae_out0); \
      AE_LA8X4S_IP(di1_0, in1_a, pae_in1); \
      AE_SA8X4U_IP(di1_0, out1_a, (ae_int32*)pae_out1); \
    } \
    AE_SA64POS_FP(out0_a, pae_out0); \
    AE_SA64POS_FP(out1_a, pae_out1); \
    for(itr_c = 0; itr_c < (cols&3); itr_c++) \
    { \
      AE_L8S_IP(di0_0, pae_in0, 1); \
      AE_S8_0_IP_HIFI1(di0_0, pae_out0, 1); \
      AE_L8S_IP(di1_0, pae_in1, 1); \
      AE_S8_0_IP_HIFI1(di1_0, pae_out1, 1); \
    } \
  } \
}
#endif //(XCHAL_HW_VERSION >= 281090)

#else // XCHAL_HAVE_HIFI1

#ifdef XCHAL_HAVE_HIFI4
  #define MEMCPY_8b(MEMCPY_8b_cp_dst, MEMCPY_8b_cp_src, MEMCPY_8b_num_elements)                           \
  {                                                                                                       \
    unsigned int MEMCPY_8b_i = 0;                                                                         \
    const unsigned int MEMCPY_8b_num_elem = MEMCPY_8b_num_elements;                                       \
                                                                                                          \
    WORD8 *MEMCPY_8b_dst = (WORD8 *)MEMCPY_8b_cp_dst;                                                     \
    WORD8 *MEMCPY_8b_src = (WORD8 *)MEMCPY_8b_cp_src;                                                     \
                                                                                                          \
    /* Both pointers are 64-bit aligned */                                                                \
    if ( (uintptr_t)MEMCPY_8b_dst%8 == 0 && (uintptr_t)MEMCPY_8b_src%8 == 0 )                             \
    {                                                                                                     \
      ae_int16x4 * __restrict__ MEMCPY_8b_s_addr = (ae_int16x4 *) MEMCPY_8b_src;                          \
      ae_int16x4 * __restrict__ MEMCPY_8b_d_addr = (ae_int16x4 *) MEMCPY_8b_dst;                          \
                                                                                                          \
      for (MEMCPY_8b_i=0; MEMCPY_8b_i<MEMCPY_8b_num_elem>>3; MEMCPY_8b_i++) {                             \
        MEMCPY_8b_d_addr[MEMCPY_8b_i] = MEMCPY_8b_s_addr[MEMCPY_8b_i];                                    \
      }                                                                                                   \
                                                                                                          \
      for (MEMCPY_8b_i=MEMCPY_8b_num_elem&(~7); MEMCPY_8b_i<MEMCPY_8b_num_elem; MEMCPY_8b_i++) {          \
        MEMCPY_8b_dst[MEMCPY_8b_i] = MEMCPY_8b_src[MEMCPY_8b_i];                                          \
      }                                                                                                   \
    }                                                                                                     \
    /* Both pointers are 16-bit aligned */                                                                \
    else if( ((uintptr_t)MEMCPY_8b_dst%2 == 0) && ((uintptr_t)MEMCPY_8b_src%2 == 0) )                     \
    {                                                                                                     \
      ae_int16x4 * __restrict__ MEMCPY_8b_s_addr = (ae_int16x4 *) MEMCPY_8b_src;                          \
      ae_int16x4 * __restrict__ MEMCPY_8b_d_addr = (ae_int16x4 *) MEMCPY_8b_dst;                          \
      ae_valign MEMCPY_8b_s_align = AE_LA64_PP(MEMCPY_8b_s_addr);                                         \
      ae_valign MEMCPY_8b_d_align = AE_ZALIGN64();                                                        \
                                                                                                          \
      ae_int16x4 MEMCPY_8b_data0, MEMCPY_8b_data1;                                                        \
                                                                                                          \
      const unsigned int MEMCPY_8b_n_by_2 = MEMCPY_8b_num_elem/2;                                         \
      for (MEMCPY_8b_i=0; MEMCPY_8b_i < MEMCPY_8b_n_by_2/8; MEMCPY_8b_i++) {                              \
          AE_LA16X4_IP(MEMCPY_8b_data0, MEMCPY_8b_s_align, MEMCPY_8b_s_addr);                             \
          AE_LA16X4_IP(MEMCPY_8b_data1, MEMCPY_8b_s_align, MEMCPY_8b_s_addr);                             \
          AE_SA16X4_IP(MEMCPY_8b_data0, MEMCPY_8b_d_align, MEMCPY_8b_d_addr);                             \
          AE_SA16X4_IP(MEMCPY_8b_data1, MEMCPY_8b_d_align, MEMCPY_8b_d_addr);                             \
      }                                                                                                   \
      AE_SA64POS_FP(MEMCPY_8b_d_align, MEMCPY_8b_d_addr);                                                 \
                                                                                                          \
      ae_int16 *s_src = (ae_int16 *) MEMCPY_8b_src;                                                       \
      ae_int16 *s_dst = (ae_int16 *) MEMCPY_8b_dst;                                                       \
      for (MEMCPY_8b_i=MEMCPY_8b_n_by_2&(~7); MEMCPY_8b_i<MEMCPY_8b_n_by_2; MEMCPY_8b_i++) {              \
          s_dst[MEMCPY_8b_i] = s_src[MEMCPY_8b_i];                                                        \
      }                                                                                                   \
                                                                                                          \
      if (MEMCPY_8b_num_elem%2) {                                                                         \
          MEMCPY_8b_dst[MEMCPY_8b_num_elem-1] = MEMCPY_8b_src[MEMCPY_8b_num_elem-1];                      \
      }                                                                                                   \
    }                                                                                                     \
    /* generic alignment, use AE_LA24X2_IP/AE_SA24X2_IP */                                                \
    else                                                                                                  \
    {                                                                                                     \
      ae_int24x2 * __restrict__ MEMCPY_8b_s_addr = (ae_int24x2 *)MEMCPY_8b_src;                           \
      ae_int24x2 * __restrict__ MEMCPY_8b_d_addr = (ae_int24x2 *)MEMCPY_8b_dst;                           \
                                                                                                          \
      ae_valign MEMCPY_8b_s_align = AE_LA64_PP(MEMCPY_8b_s_addr);                                         \
      ae_valign MEMCPY_8b_d_align = AE_ZALIGN64();                                                        \
                                                                                                          \
      ae_int24x2 MEMCPY_8b_data;                                                                          \
                                                                                                          \
      unsigned int MEMCPY_8b_Nby6 =                                                                       \
          AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(MEMCPY_8b_num_elem, 0x2AAAAAAB)));             \
      unsigned int MEMCPY_8b_remainder_start = 6*MEMCPY_8b_Nby6;                                          \
                                                                                                          \
      /* copy-paste 6 octets at a time */                                                                 \
      for(MEMCPY_8b_i=0; MEMCPY_8b_i < MEMCPY_8b_Nby6; MEMCPY_8b_i++){                                    \
          AE_LA24X2_IP(MEMCPY_8b_data, MEMCPY_8b_s_align, MEMCPY_8b_s_addr);                              \
          AE_SA24X2_IP(MEMCPY_8b_data, MEMCPY_8b_d_align, MEMCPY_8b_d_addr);                              \
      }                                                                                                   \
      AE_SA64POS_FP(MEMCPY_8b_d_align, MEMCPY_8b_d_addr);                                                 \
                                                                                                          \
      /* remaining MEMCPY_8b_num_elem-mod-6 octets */                                                     \
      for(MEMCPY_8b_i=MEMCPY_8b_remainder_start; MEMCPY_8b_i < MEMCPY_8b_num_elem; MEMCPY_8b_i++){        \
          MEMCPY_8b_dst[MEMCPY_8b_i] = MEMCPY_8b_src[MEMCPY_8b_i];                                        \
      }                                                                                                   \
    }                                                                                                     \
  }
#else
    #define MEMCPY_8b memcpy
#endif

#define MEMCPY_2D_8b_CONT_OUT(dst, src, rows, cols, inp_row_stride){    \
                                                                        \
    void *p_src = src;                                                  \
    void *p_dst = dst;                                                  \
                                                                        \
    for(int row = 0; row < rows; row++){                                \
        MEMCPY_8b(p_dst, p_src, cols);                                  \
        p_src += inp_row_stride;                                        \
        p_dst += cols;                                                  \
    }                                                                   \
}

#define DUAL_MEMCPY_2D_8b_CONT_OUT(dst0, dst1, src0, src1, rows, cols, inp_row_stride){ \
                                                                                        \
    void *p_src0 = src0;                                                                \
    void *p_src1 = src1;                                                                \
                                                                                        \
    void *p_dst0 = dst0;                                                                \
    void *p_dst1 = dst1;                                                                \
                                                                                        \
    for(int row = 0; row < rows; row++){                                                \
        MEMCPY_8b(p_dst0, p_src0, cols);                                                \
        MEMCPY_8b(p_dst1, p_src1, cols);                                                \
                                                                                        \
        p_src0 += inp_row_stride;                                                       \
        p_src1 += inp_row_stride;                                                       \
                                                                                        \
        p_dst0 += cols;                                                                 \
        p_dst1 += cols;                                                                 \
    }                                                                                   \
}

#define MEMCPY_2D_8b_CONT_INP(dst, src, rows, cols, out_row_stride){    \
                                                                        \
    void *p_src = src;                                                  \
    void *p_dst = dst;                                                  \
                                                                        \
    for(int row = 0; row < rows; row++){                                \
        MEMCPY_8b(p_dst, p_src, cols);                                  \
        p_src += cols;                                                  \
        p_dst += out_row_stride;                                        \
    }                                                                   \
}

#define DUAL_MEMCPY_2D_8b_CONT_INP(dst0, dst1, src0, src1, rows, cols, out_row_stride){ \
                                                                                        \
    void *p_src0 = src0;                                                                \
    void *p_src1 = src1;                                                                \
                                                                                        \
    void *p_dst0 = dst0;                                                                \
    void *p_dst1 = dst1;                                                                \
                                                                                        \
    for(int row = 0; row < rows; row++){                                                \
        MEMCPY_8b(p_dst0, p_src0, cols);                                                \
        MEMCPY_8b(p_dst1, p_src1, cols);                                                \
                                                                                        \
        p_src0 += cols;                                                                 \
        p_src1 += cols;                                                                 \
                                                                                        \
        p_dst0 += out_row_stride;                                                       \
        p_dst1 += out_row_stride;                                                       \
    }                                                                                   \
}

#endif // XCHAL_HAVE_HIFI1

#define ALIGNMENT   8
/*Macro checking matmul kernels alignment */
#define CHK_MATMUL_ALIGN(chk_align, mat, algn_m, vec, algn_v, cols, row_str, vec_off, unr) \
  chk_align = 0; \
  if(!((unsigned int)(mat) & (algn_m-1)) && !((unsigned int)(vec) & (algn_v - 1)) && (cols%unr==0) && (row_str%unr==0) && (vec_off%unr==0)) \
  { \
    chk_align = 1; \
  }

/* Macro for zero value */
#define ZERO64   AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0))
#define ZERO16X4 AE_MOVDA16(0)
#define ZERO16   (0)
#define ZERO32   (0)

/* Macro for 1 */
#define ONE16X4 AE_MOVDA16(1)

/* Value of ROW_UNROLL currently supported are 1,2,4,8 only */
#ifndef ROW_UNROLL
#define ROW_UNROLL 8
#endif
#define VEC_UNROLL 2

#define ACC_LSH_AFTER_FIRST_MATXVEC 0

/* Increment in bytes required for particular load
 * instructions. */
#define INCREMENT_IN_BYTES_FOR_WORD8     1
#define INCREMENT_IN_BYTES_FOR_INT16     2
#define INCREMENT_IN_BYTES_FOR_INT32     (INCREMENT_IN_BYTES_FOR_INT16   * 2)
#define INCREMENT_IN_BYTES_FOR_WORD8X4   (INCREMENT_IN_BYTES_FOR_WORD8   * 4)
#define INCREMENT_IN_BYTES_FOR_INT16X4   (INCREMENT_IN_BYTES_FOR_INT16   * 4)
#define INCREMENT_IN_BYTES_FOR_INT64     INCREMENT_IN_BYTES_FOR_INT16X4
#define INCREMENT_IN_BYTES_FOR_FLOAT32   4
#define INCREMENT_IN_BYTES_FOR_FLOAT32x2 (INCREMENT_IN_BYTES_FOR_FLOAT32 * 2)

/* Limit effective bias_shift and acc_shift to [-63 ... 63] */
#define LIMIT_VARIABLE(_var, _left_limit, _right_limit) \
  _var = _var > _right_limit ? _right_limit : _var < _left_limit ? _left_limit : _var;

#define LIMIT_ACC_LSH \
  LIMIT_VARIABLE(acc_shift, -63, 63); \

#define LIMIT_BIAS_LSH \
  LIMIT_VARIABLE(bias_shift, -63, 63); \

#define BW(_datatype) sizeof(_datatype)

#define ADJUST_VAR_AxB(A, B) \
  (((8 * (4 - (BW(A) + BW(B))))))

#define ADJUST_VAR_C(C) \
  (((64 - (8 * BW(C)))))

#define ADJUST_ACC_LSH_AxB_C(A, B, C) \
  acc_shift = acc_shift + 32; \
  LIMIT_ACC_LSH;

#define ADJUST_BIAS_LSH_AxB(A, B) \
  LIMIT_BIAS_LSH;

#define ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(A, B, C) \
  ADJUST_ACC_LSH_AxB_C(A, B, C); \
  ADJUST_BIAS_LSH_AxB(A, B); \

/* ==================================================================================================== */
#define SETUP_BIAS_f32 \
  xtfloat _xtfloat_bias = (xtfloat)0.0f; \
  xtfloat *_xtfloat_p_bias = (xtfloat *) p_bias; \

#if XCHAL_HAVE_HIFI1
#define SETUP_BIAS_ASYM8b \
  ae_int32x2 _ae_int32x2_bias; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD32 *_WORD32_p_bias = (WORD32 *) p_bias; \

#else
#define SETUP_BIAS_ASYM8b \
  WORD32 _WORD32_bias; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD32 *_WORD32_p_bias = (WORD32 *) p_bias; \

#endif

#if XCHAL_HAVE_HIFI1
#define SETUP_BIAS_8b \
  ae_int16x4 _ae_int16x4_bias; \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD8 *_WORD8_p_bias = (WORD8 *) p_bias; \

#else
#define SETUP_BIAS_8b \
  WORD8 _WORD8_bias; \
  UWORD32 _UWORD32_bias; \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD8 *_WORD8_p_bias = (WORD8 *) p_bias; \

#endif

#if XCHAL_HAVE_HIFI1
#define SETUP_BIAS_8b_BATCH \
  ae_int16x4 _ae_int16x4_bias; \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD8 *_WORD8_p_bias = (WORD8 *) p_bias; \

#else
#define SETUP_BIAS_8b_BATCH \
  WORD8 _WORD8_bias; \
  UWORD32 _UWORD32_bias; \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD8 *_WORD8_p_bias = (WORD8 *) p_bias; \

#endif

#define SETUP_BIAS_32b \
  ae_int32 _ae_int32_bias = ZERO32; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int32 *_ae_int32_p_bias = (ae_int32 *) p_bias; \

#define SETUP_BIAS_16b \
  ae_int16 _ae_int16_bias = ZERO16; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int16 *_ae_int16_p_bias = (ae_int16 *) p_bias; \

#define SETUP_BIAS_64b \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int64 *_ae_int64_p_bias = (ae_int64 *) p_bias; \

#define SETUP_ACC_FOR_8bx8b(idx)   SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_8bx16b(idx)  SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_16bx8b(idx)  SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_16bx16b(idx) SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_ASYM8bxASYM8b(idx) SETUP_ACC_64b(idx)

/*------------------ time batching macros ----------------- */

#define SETUP_ACC_BATCH_ROW_FOR_16bx8b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx16b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx8b  SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_ROW_FOR_16bx16b

#define SETUP_ACC_BATCH_FOR_16bx8b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx16b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx8b  SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_FOR_16bx16b

#define SETUP_ACC_BATCH_ROW_FOR_16bx16b(idx_row)\
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);\

#define SETUP_ACC_BATCH_FOR_16bx16b(idx_row,idx_vec) \
  ae_int64 _ae_int64_acc_ ##idx_row ##_ ##idx_vec = ZERO64; \

#define SETUP_ACC_BATCH_ROW_FOR_f32(idx_row)\
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);\

#define SETUP_ACC_BATCH_FOR_f32(idx_row,idx_vec) \
  xtfloatx2 _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
  xtfloat _xtfloat_acc_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;\

#define SETUP_ACC_BATCH_FOR_f32_MATMUL(idx_row,idx_vec) \
  xtfloatx2 _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
  xtfloat _xtfloat_acc_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;\
  xtfloat _xtfloat_acc1_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;\
/*---------------------------------------------------------*/

#define SETUP_ACC_64b(idx) \
  ae_int64 _ae_int64_acc_ ## idx = ZERO64; \

#define SETUP_VEC1_8b \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  WORD8 *_WORD8_p_vec1 = (WORD8 *) p_vec1; \

#define SETUP_VEC1_8b_x2 \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  WORD8 *_WORD8_p_vec1 = (WORD8 *) p_vec1; \
  ae_int16x4 _ae_int16x4_vec1_I = ZERO16X4; \

#define SETUP_VEC2_8b_x2 \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  WORD8 *_WORD8_p_vec2 = (WORD8 *) p_vec2; \
  ae_int16x4 _ae_int16x4_vec2_I = ZERO16X4; \

#define SETUP_VEC1_16b_x2 \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec1 = (ae_int16x4 *) p_vec1; \
  ae_int16x4 _ae_int16x4_vec1_I = ZERO16X4; \

#define SETUP_VEC2_16b_x2 \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec2 = (ae_int16x4 *) p_vec2; \
  ae_int16x4 _ae_int16x4_vec2_I = ZERO16X4; \

#define SETUP_VEC2_8b \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  WORD8 *_WORD8_p_vec2 = (WORD8 *) p_vec2; \

#define SETUP_VEC1_16b \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec1 = (ae_int16x4 *) p_vec1; \

#define SETUP_VEC2_16b \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec2 = (ae_int16x4 *) p_vec2; \

#define SETUP_VEC1_ASYM8b SETUP_VEC1_8b
#define SETUP_VEC2_ASYM8b SETUP_VEC2_8b
/*------------------ time batching macros ----------------- */

#define SETUP_VEC_BATCH_8b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_BATCH_8b_x2(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec ##_I  = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_OFFSET_BATCH_8b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \

#define SETUP_VEC_OFFSET_BATCH_8b_UNALIGNED(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \
  ALIGN_REGISTER_TYPE _align_WORD8_p_vec_batch_ ##idx_vec;\
  PRIME_8X4F(_WORD8_p_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec)

#define SETUP_VEC_BATCH_16b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec_batch_ ##idx_vec  = (ae_int16x4 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_OFFSET_BATCH_16b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec_batch_ ##idx_vec  = (ae_int16x4 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \

#define SETUP_VEC_OFFSET_BATCH_16b_UNALIGNED(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec_batch_ ##idx_vec  = (ae_int16x4 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \
  ae_valign _align_ae_int16x4_p_vec_batch_ ##idx_vec = AE_LA64_PP(_ae_int16x4_p_vec_batch_ ##idx_vec);

#define SETUP_VEC_BATCH_f32(idx_vec)\
  xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
  xtfloatx2 *_xtfloatx2_p_vec_batch_ ##idx_vec  = (xtfloatx2 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_OFFSET_BATCH_f32(idx_vec)\
  xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
  xtfloatx2 *_xtfloatx2_p_vec_batch_ ##idx_vec  = (xtfloatx2 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \

#define SETUP_VEC_OFFSET_BATCH_f32_UNALIGNED(idx_vec)\
  xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
  xtfloat _xtfloat_vec_batch_ ##idx_vec  = (xtfloat)0.0f ; \
  xtfloatx2 *_xtfloatx2_p_vec_batch_ ##idx_vec  = (xtfloatx2 *)(p_vec1 + (vec_itr + idx_vec)*vec_offset); \
  xtfloat *_xtfloat_p_vec_batch_ ##idx_vec; \
  ae_valign _align_xtfloatx2_p_vec_batch_ ##idx_vec = AE_LA64_PP(_xtfloatx2_p_vec_batch_ ##idx_vec); 

#define SETUP_VEC_BATCH_ASYM8b SETUP_VEC_BATCH_8b
#define SETUP_VEC_OFFSET_BATCH_ASYM8b SETUP_VEC_OFFSET_BATCH_8b
#define SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED SETUP_VEC_OFFSET_BATCH_8b_UNALIGNED
/*---------------------------------------------------------*/

#define SETUP_MAT1_8b(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat1_ ## idx = (WORD8 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT1_8b_x2(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat1_ ## idx = (WORD8 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_int16x4 _ae_int16x4_mat1_ ## idx ## _I = ZERO16X4; \

#define SETUP_MAT2_8b_x2(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat2_ ## idx = (WORD8 *) &p_mat2[(m_itr+idx)*row_stride2]; \
  ae_int16x4 _ae_int16x4_mat2_ ## idx ## _I = ZERO16X4; \

#define SETUP_MAT1_16b_x2(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat1_ ## idx = (ae_int16x4 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_int16x4 _ae_int16x4_mat1_ ## idx ## _I = ZERO16X4; \

#define SETUP_MAT2_16b_x2(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat2_ ## idx = (ae_int16x4 *) &p_mat2[(m_itr+idx)*row_stride2]; \
  ae_int16x4 _ae_int16x4_mat2_ ## idx ## _I = ZERO16X4; \

#define SETUP_MAT1_8b_UNALIGNED(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat1_ ## idx = (WORD8 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ALIGN_REGISTER_TYPE _align_WORD8_p_mat1_ ## idx;\
  PRIME_8X4F(_WORD8_p_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx);

#define SETUP_MAT2_8b(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat2_ ## idx = (WORD8 *) &p_mat2[(m_itr+idx)*row_stride2]; \

#define SETUP_MAT1_16b(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat1_ ## idx = (ae_int16x4 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT1_16b_UNALIGNED(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat1_ ## idx = (ae_int16x4 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_valign _align_ae_int16x4_p_mat1_ ## idx = AE_LA64_PP(_ae_int16x4_p_mat1_ ##idx);

#define SETUP_MAT2_16b(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat2_ ## idx = (ae_int16x4 *) &p_mat2[(m_itr+idx)*row_stride2]; \

#define SETUP_MAT1_f32(idx) \
  xtfloatx2 _xtfloatx2_mat1_ ## idx = (xtfloatx2)0.0f; \
  xtfloatx2 *_xtfloatx2_p_mat1_ ## idx = (xtfloatx2 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT1_f32_UNALIGNED(idx) \
  xtfloatx2 _xtfloatx2_mat1_ ## idx = (xtfloatx2)0.0f; \
  xtfloat _xtfloat_mat1_ ## idx = (xtfloat)0.0f; \
  xtfloatx2 *_xtfloatx2_p_mat1_ ## idx = (xtfloatx2 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  xtfloat *_xtfloat_p_mat1_ ## idx; \
  ae_valign _align_xtfloatx2_p_mat1_ ## idx = AE_LA64_PP(_xtfloatx2_p_mat1_ ## idx);

#define SETUP_MAT1_ASYM8b SETUP_MAT1_8b
#define SETUP_MAT1_ASYM8b_UNALIGNED SETUP_MAT1_8b_UNALIGNED
#define SETUP_MAT2_ASYM8b SETUP_MAT2_8b
/* ====================================================================== */

#define LOAD_VEC1_8b \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC1_8b_x2 \
  _ae_int16x4_vec1_I = AE_L8X4F_I(_WORD8_p_vec1,INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1, 2 * INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC2_8b_x2 \
  _ae_int16x4_vec2_I = AE_L8X4F_I(_WORD8_p_vec2,INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2, 2 * INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC1_16b_x2 \
  _ae_int16x4_vec1_I = AE_L16X4_I(_ae_int16x4_p_vec1,INCREMENT_IN_BYTES_FOR_INT16X4); \
  AE_L16X4_IP(_ae_int16x4_vec1, _ae_int16x4_p_vec1, 2   * INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_16b_x2 \
  _ae_int16x4_vec2_I = AE_L16X4_I(_ae_int16x4_p_vec2,INCREMENT_IN_BYTES_FOR_INT16X4); \
  AE_L16X4_IP(_ae_int16x4_vec2, _ae_int16x4_p_vec2, 2   * INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_8b \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC1_16b \
  AE_L16X4_IP(_ae_int16x4_vec1, _ae_int16x4_p_vec1, INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_16b \
  AE_L16X4_IP(_ae_int16x4_vec2, _ae_int16x4_p_vec2, INCREMENT_IN_BYTES_FOR_INT16X4); \

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC1_ASYM8b \
  AE_L8X4U_IP(_ae_int16x4_vec1, _WORD8_p_vec1, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec1 = AE_ADD16(_ae_int16x4_vec1, AE_MOVDA16(vec1_zero_bias)); \

#else
#define LOAD_VEC1_ASYM8b \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec1), 8)); \
  _ae_int16x4_vec1 = AE_ADD16(_ae_int16x4_vec1, AE_MOVDA16(vec1_zero_bias)); \

#endif

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC2_ASYM8b \
  AE_L8X4U_IP(_ae_int16x4_vec2, _WORD8_p_vec2, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec2 = AE_ADD16(_ae_int16x4_vec2, AE_MOVDA16(vec2_zero_bias)); \

#else
#define LOAD_VEC2_ASYM8b \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec2), 8)); \
  _ae_int16x4_vec2 = AE_ADD16(_ae_int16x4_vec2, AE_MOVDA16(vec2_zero_bias)); \

#endif
/*------------------ time batching macros ----------------- */
#define LOAD_VEC_BATCH_f32(idx_vec) \
  XT_LSX2IP(_xtfloatx2_vec_batch_ ##idx_vec, _xtfloatx2_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_FLOAT32x2); \

#define LOAD_VEC_BATCH_f32_UNALIGNED(idx_vec) \
  XT_LASX2IP(_xtfloatx2_vec_batch_ ##idx_vec, _align_xtfloatx2_p_vec_batch_ ##idx_vec, _xtfloatx2_p_vec_batch_ ##idx_vec); \

#define LOAD_VEC_BATCH_f32_SINGLE_UNALIGNED(idx_vec) \
  _xtfloat_p_vec_batch_ ##idx_vec = (xtfloat *)_xtfloatx2_p_vec_batch_ ##idx_vec; \
  XT_LSIP(_xtfloat_vec_batch_ ##idx_vec, _xtfloat_p_vec_batch_ ##idx_vec, 4); \

#define LOAD_VEC_BATCH_8b(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC_BATCH_8b_x2(idx_vec) \
  _ae_int16x4_vec_batch_ ##idx_vec ##_I = AE_L8X4F_I(_WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, 2 * INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC_BATCH_8b_UNALIGNED(idx_vec) \
  AE_LA8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec); \

#define LOAD_VEC_BATCH_8b_SINGLE_UNALIGNED(idx_vec) \
 _ae_int16x4_vec_batch_ ##idx_vec = AE_MOVDA16(((short)*(_WORD8_p_vec_batch_ ##idx_vec)) << 8); \
 _WORD8_p_vec_batch_ ##idx_vec++;\

#define LOAD_VEC_BATCH_16b(idx_vec) \
  AE_L16X4_IP(_ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC_BATCH_16b_UNALIGNED(idx_vec) \
  AE_LA16X4_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_ae_int16x4_p_vec_batch_ ##idx_vec, _ae_int16x4_p_vec_batch_ ##idx_vec); \

#define LOAD_VEC_BATCH_16b_SINGLE_UNALIGNED(idx_vec) \
  AE_L16_IP(_ae_int16x4_vec_batch_ ##idx_vec, (ae_int16 *)_ae_int16x4_p_vec_batch_ ##idx_vec, 2); \

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC_BATCH_ASYM8b(idx_vec) \
  AE_L8X4U_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs(idx_vec) \
  AE_L8X4S_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#else
#define LOAD_VEC_BATCH_ASYM8b(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_ ##idx_vec), 8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_SRAI16(_ae_int16x4_vec_batch_ ##idx_vec, 8); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#endif

#if XCHAL_HAVE_HIFI1 //Unaligned loads not available
#define LOAD_VEC_BATCH_ASYM8b_UNALIGNED(idx_vec) \
  AE_LA8X4U_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(idx_vec) \
  AE_LA8X4S_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#else
#define LOAD_VEC_BATCH_ASYM8b_UNALIGNED(idx_vec) \
  AE_LA8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_ ##idx_vec), 8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs_UNALIGNED(idx_vec) \
  AE_LA8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _align_WORD8_p_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_SRAI16(_ae_int16x4_vec_batch_ ##idx_vec, 8); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \


#endif

#if XCHAL_HAVE_HIFI1
#define LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(idx_vec) \
  AE_L8U_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8 ); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(idx_vec) \
  AE_L8S_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8 ); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \


#else
#define LOAD_VEC_BATCH_ASYM8b_SINGLE_UNALIGNED(idx_vec) \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_MOVDA16(((short)*(_WORD8_p_vec_batch_ ##idx_vec)) << 8); \
  _WORD8_p_vec_batch_ ##idx_vec++;\
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_ ##idx_vec), 8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC_BATCH_ASYM8bs_SINGLE_UNALIGNED(idx_vec) \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_MOVDA16(((short)*(_WORD8_p_vec_batch_ ##idx_vec)) << 8); \
  _WORD8_p_vec_batch_ ##idx_vec++;\
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_SRAI16(_ae_int16x4_vec_batch_ ##idx_vec, 8); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#endif

#if XCHAL_HAVE_HIFI1
#define LOAD_BIAS_8b_FOR_8bx8b \
  AE_L8S_IP(_ae_int16x4_bias, _WORD8_p_bias, INCREMENT_IN_BYTES_FOR_WORD8); \
  _ae_int64_bias = AE_MOVINT64_FROMINT16X4(_ae_int16x4_bias); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 48); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \

#else
#define LOAD_BIAS_8b_FOR_8bx8b \
  _WORD8_bias = *_WORD8_p_bias++; \
  _UWORD32_bias = _WORD8_bias; \
  _ae_int64_bias = AE_MOVINT64_FROMINT32X2((AE_MOVDA32(_UWORD32_bias))); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 32); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \

#endif

#if XCHAL_HAVE_HIFI1
#define LOAD_BIAS_8b_FOR_8bx8b_MATMUL \
  if(p_bias!=NULL)\
  {\
   AE_L8S_IP(_ae_int16x4_bias, _WORD8_p_bias, INCREMENT_IN_BYTES_FOR_WORD8); \
  _ae_int64_bias = AE_MOVINT64_FROMINT16X4(_ae_int16x4_bias); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 48); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \
  }
#else
#define LOAD_BIAS_8b_FOR_8bx8b_MATMUL \
  if(p_bias!=NULL)\
  {\
  _WORD8_bias = *_WORD8_p_bias++; \
  _UWORD32_bias = _WORD8_bias; \
  _ae_int64_bias = AE_MOVINT64_FROMINT32X2((AE_MOVDA32(_UWORD32_bias))); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 32); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \
  }
#endif

#define LOAD_BIAS_16b_FOR_8bx16b \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \

#define LOAD_BIAS_16b_FOR_8bx16b_MATMUL \
  if(p_bias!=NULL)\
  {\
    ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
    _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  }

#define LOAD_BIAS_16b_FOR_16bx8b LOAD_BIAS_16b_FOR_8bx16b

#define LOAD_BIAS_16b_FOR_16bx16b \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \

#define LOAD_BIAS_16b_FOR_16bx16b_MATMUL \
  if(p_bias!=NULL)\
  {\
    ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
    _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  }

#define LOAD_BIAS_f32 \
  XT_LSIP(_xtfloat_bias, _xtfloat_p_bias, INCREMENT_IN_BYTES_FOR_FLOAT32); \

#define LOAD_BIAS_f32_MATMUL \
  if(p_bias!=NULL)\
  {\
      XT_LSIP(_xtfloat_bias, _xtfloat_p_bias, INCREMENT_IN_BYTES_FOR_FLOAT32); \
  }

#if XCHAL_HAVE_HIFI1
#define LOAD_BIAS_ASYM8b \
  AE_L32_IP(_ae_int32x2_bias, (ae_int32*)_WORD32_p_bias, 4); \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_bias), 32); \

#define LOAD_BIAS_ASYM8b_MATMUL \
  if(p_bias!=NULL)\
  {\
    AE_L32_IP(_ae_int32x2_bias, (ae_int32*)_WORD32_p_bias, 4); \
    _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_bias), 32); \
  }
#else
#define LOAD_BIAS_ASYM8b \
  _WORD32_bias = *_WORD32_p_bias++; \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \

#define LOAD_BIAS_ASYM8b_MATMUL \
  if(p_bias!=NULL)\
  {\
    _WORD32_bias = *_WORD32_p_bias++; \
    _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
  }
#endif
/*---------------------------------------------------------*/
#define LOAD_ROW_MAT1_8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ## idx, _WORD8_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_ROW_MAT1_8b_x2(idx) \
  _ae_int16x4_mat1_ ## idx ##_I = AE_L8X4F_I(_WORD8_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ## idx, _WORD8_p_mat1_ ## idx, 2 * INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_ROW_MAT1_8b_UNALIGNED(idx) \
  AE_LA8X4F_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \

#define LOAD_ROW_MAT1_8b_SINGLE_UNALIGNED(idx) \
 _ae_int16x4_mat1_ ## idx = AE_MOVDA16(((short)*(_WORD8_p_mat1_ ## idx)) << 8); \
 _WORD8_p_mat1_ ## idx++;\

#define LOAD_ROW_MAT2_8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_ROW_MAT1_16b(idx) \
  AE_L16X4_IP(_ae_int16x4_mat1_ ## idx, _ae_int16x4_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_ROW_MAT1_16b_UNALIGNED(idx) \
  AE_LA16X4_IP(_ae_int16x4_mat1_ ## idx, _align_ae_int16x4_p_mat1_ ## idx, _ae_int16x4_p_mat1_ ## idx); \

#define LOAD_ROW_MAT1_16b_SINGLE_UNALIGNED(idx) \
  AE_L16_IP(_ae_int16x4_mat1_ ## idx, (ae_int16 *)_ae_int16x4_p_mat1_ ## idx, 2); \

#define LOAD_ROW_MAT2_16b(idx) \
  AE_L16X4_IP(_ae_int16x4_mat2_ ## idx, _ae_int16x4_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_ROW_MAT1_f32(idx) \
  XT_LSX2IP(_xtfloatx2_mat1_ ## idx, _xtfloatx2_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_FLOAT32x2); \

#define LOAD_ROW_MAT1_f32_UNALIGNED(idx) \
  XT_LASX2IP(_xtfloatx2_mat1_ ## idx, _align_xtfloatx2_p_mat1_ ## idx, _xtfloatx2_p_mat1_ ## idx); \

#define LOAD_ROW_MAT1_f32_SINGLE_UNALIGNED(idx) \
  _xtfloat_p_mat1_ ## idx = (xtfloat *)_xtfloatx2_p_mat1_ ## idx;\
  XT_LSIP(_xtfloat_mat1_ ## idx, _xtfloat_p_mat1_ ## idx, 4); \

#if XCHAL_HAVE_HIFI1
#define LOAD_ROW_MAT1_ASYM8b(idx) \
  AE_L8X4U_IP(_ae_int16x4_mat1_ ##idx, _WORD8_p_mat1_ ##idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs(idx) \
  AE_L8X4S_IP(_ae_int16x4_mat1_ ##idx, _WORD8_p_mat1_ ##idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#else
#define LOAD_ROW_MAT1_ASYM8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ##idx, _WORD8_p_mat1_ ##idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ##idx, _WORD8_p_mat1_ ##idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat1_ ##idx = AE_SRAI16(_ae_int16x4_mat1_ ##idx, 8); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#endif

#if XCHAL_HAVE_HIFI1//Unaligned loads not available
#define LOAD_ROW_MAT1_ASYM8b_UNALIGNED(idx) \
  AE_LA8X4U_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(idx) \
  AE_LA8X4S_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#else //HIFI_LE
#define LOAD_ROW_MAT1_ASYM8b_UNALIGNED(idx) \
  AE_LA8X4F_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs_UNALIGNED(idx) \
  AE_LA8X4F_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \
  _ae_int16x4_mat1_ ##idx = AE_SRAI16(_ae_int16x4_mat1_ ##idx, 8); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#endif //HIFI_LE

#if XCHAL_HAVE_HIFI1
#define LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(idx) \
  AE_L8U_IP(_ae_int16x4_mat1_ ##idx , (_WORD8_p_mat1_ ## idx), INCREMENT_IN_BYTES_FOR_WORD8);\
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(idx) \
  AE_L8S_IP(_ae_int16x4_mat1_ ##idx , (_WORD8_p_mat1_ ## idx), INCREMENT_IN_BYTES_FOR_WORD8);\
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#else //HIFI_LE
#define LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(idx) \
  _ae_int16x4_mat1_ ## idx = AE_MOVDA16(((short)*(_WORD8_p_mat1_ ## idx)) << 8); \
  _WORD8_p_mat1_ ## idx++;\
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8bs_SINGLE_UNALIGNED(idx) \
  _ae_int16x4_mat1_ ## idx = AE_MOVDA16(((short)*(_WORD8_p_mat1_ ## idx)) << 8); \
  _WORD8_p_mat1_ ## idx++;\
  _ae_int16x4_mat1_ ##idx = AE_SRAI16(_ae_int16x4_mat1_ ##idx, 8); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#endif //HIFI_LE

#if XCHAL_HAVE_HIFI1
#define LOAD_ROW_MAT2_ASYM8b(idx) \
  AE_L8X4U_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat2_ ##idx = AE_ADD16(_ae_int16x4_mat2_ ##idx, AE_MOVDA16(mat2_zero_bias)); \

#else //HIFI_LE
#define LOAD_ROW_MAT2_ASYM8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat2_ ## idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat2_ ## idx), 8)); \
  _ae_int16x4_mat2_ ##idx = AE_ADD16(_ae_int16x4_mat2_ ##idx, AE_MOVDA16(mat2_zero_bias)); \

#endif //HIFI_LE

#define KERNEL_MAT1_VEC1_8b_8b(idx) \
  LOAD_ROW_MAT1_8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT1_VEC1_8b_8b_x2(idx) \
  _ae_int16x4_mat1_ ## idx ## _I = AE_L8X4F_I(_WORD8_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ## idx, _WORD8_p_mat1_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_WORD8X4)); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1_I, _ae_int16x4_mat1_ ## idx ## _I);

#define KERNEL_MAT2_VEC2_8b_8b_x2(idx) \
  _ae_int16x4_mat2_ ## idx ## _I = AE_L8X4F_I(_WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_WORD8X4));\
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2_I, _ae_int16x4_mat2_ ## idx ## _I); 

#define KERNEL_MAT1_VEC1_8b_16b_x2(idx) \
  _ae_int16x4_mat1_ ## idx ## _I = AE_L8X4F_I(_WORD8_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ## idx, _WORD8_p_mat1_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_WORD8X4)); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1_I, _ae_int16x4_mat1_ ## idx ## _I);

#define KERNEL_MAT2_VEC2_8b_16b_x2(idx) \
  _ae_int16x4_mat2_ ## idx ## _I = AE_L8X4F_I(_WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  AE_L8X4F_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_WORD8X4));\
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2_I, _ae_int16x4_mat2_ ## idx ## _I);

#define KERNEL_MAT1_VEC1_16b_16b_x2(idx) \
  _ae_int16x4_mat1_ ## idx ## _I = AE_L16X4_I(_ae_int16x4_p_mat1_ ## idx, INCREMENT_IN_BYTES_FOR_INT16X4); \
  AE_L16X4_IP(_ae_int16x4_mat1_ ## idx, _ae_int16x4_p_mat1_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_INT16X4)); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1_I, _ae_int16x4_mat1_ ## idx ## _I);

#define KERNEL_MAT2_VEC2_16b_16b_x2(idx) \
  _ae_int16x4_mat2_ ## idx ## _I = AE_L16X4_I(_ae_int16x4_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_INT16X4); \
  AE_L16X4_IP(_ae_int16x4_mat2_ ## idx, _ae_int16x4_p_mat2_ ## idx, (2 * INCREMENT_IN_BYTES_FOR_INT16X4));\
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2_I, _ae_int16x4_mat2_ ## idx ## _I);

#define KERNEL_MAT2_VEC2_8b_8b(idx) \
  LOAD_ROW_MAT2_8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_8b(idx) \
  LOAD_ROW_MAT1_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_16b_8b(idx) \
  LOAD_ROW_MAT2_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_8b_16b(idx) \
  LOAD_ROW_MAT1_8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_8b_16b(idx) \
  LOAD_ROW_MAT2_8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_16b(idx) \
  LOAD_ROW_MAT1_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_16b_16b(idx) \
  LOAD_ROW_MAT2_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT1_ASYM8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT2_ASYM8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

/*------------------ time batching macros ----------------- */

#define KERNEL_MAT1_VEC_BATCH_ROW_8b_8b             KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_8b_8b_X2          KERNEL_MAT1_VEC_BATCH_ROW_16b_16b_X2
#define KERNEL_MAT1_VEC_BATCH_ROW_16b_8b            KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_8b_16b            KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b     KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_8b                 KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_8b_x2              KERNEL_MAT1_VEC_BATCH_16b_16b_x2
#define KERNEL_MAT1_VEC_BATCH_16b_8b                KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_16b                KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b         KERNEL_MAT1_VEC_BATCH_16b_16b

#define KERNEL_MAT1_VEC_BATCH_ROW_16b_16b(idx_row)\
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT1_VEC_BATCH_ROW_16b_16b_X2(idx_row)\
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL_X2(idx_row);\

#define KERNEL_MAT1_VEC_BATCH_16b_16b(idx_row,idx_vec) \
  AE_MULAAAAQ16(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_mat1_ ##idx_row); \

#define KERNEL_MAT1_VEC_BATCH_16b_16b_x2(idx_row,idx_vec) \
  AE_MULAAAAQ16(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_mat1_ ##idx_row); \
  AE_MULAAAAQ16(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec ##_I, _ae_int16x4_mat1_ ##idx_row ##_I); \

#define KERNEL_MAT1_VEC_BATCH_8b_16b_SINGLE_UNALIGNED                KERNEL_MAT1_VEC_BATCH_16b_16b_SINGLE_UNALIGNED
#define KERNEL_MAT1_VEC_BATCH_8b_8b_SINGLE_UNALIGNED                 KERNEL_MAT1_VEC_BATCH_16b_16b_SINGLE_UNALIGNED
#define KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b_SINGLE_UNALIGNED         KERNEL_MAT1_VEC_BATCH_16b_16b_SINGLE_UNALIGNED

#define KERNEL_MAT1_VEC_BATCH_16b_16b_SINGLE_UNALIGNED(idx_row,idx_vec) \
  AE_MULA16_00(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_mat1_ ##idx_row); \

#define KERNEL_MAT1_VEC_BATCH_ROW_f32(idx_row)\
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT1_VEC_BATCH_f32(idx_row,idx_vec) \
  XT_MADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec, _xtfloatx2_vec_batch_ ##idx_vec, _xtfloatx2_mat1_ ##idx_row); \

#define KERNEL_MAT1_VEC_BATCH_f32_SINGLE_UNALIGNED(idx_row,idx_vec) \
  XT_MADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_vec_batch_ ##idx_vec, _xtfloat_mat1_ ##idx_row); \

/*---------------------------------------------------------*/

#if XCHAL_HAVE_HIFI1
#define ADD_BIAS_8b_ACC_FOR_8bx8b(idx) \
  AE_L8S_IP(_ae_int16x4_bias, _WORD8_p_bias, INCREMENT_IN_BYTES_FOR_WORD8); \
 _ae_int64_bias = AE_MOVINT64_FROMINT16X4(_ae_int16x4_bias); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 48); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \
  _ae_int64_acc_ ## idx = AE_SRAI64(_ae_int64_acc_ ## idx, 16); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#else
#define ADD_BIAS_8b_ACC_FOR_8bx8b(idx) \
  /* Load 8b bias */ \
  _WORD8_bias = *_WORD8_p_bias++; \
  /* Copy 8-bits to unsigned 32-bits */ \
  _UWORD32_bias = _WORD8_bias; \
  /*Move unsigned 32 bit value to DR register*/ \
  _ae_int64_bias = AE_MOVINT64_FROMINT32X2((AE_MOVDA32(_UWORD32_bias))); \
  _ae_int64_bias = AE_SRAI64(_ae_int64_bias, 32); \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift); \
  _ae_int64_acc_ ## idx = AE_SRAI64(_ae_int64_acc_ ## idx, 16); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#endif

#define ADD_BIAS_32b_ACC_FOR_8bx8b(idx) \
  ae_int32_loadip(_ae_int32_bias, _ae_int32_p_bias, INCREMENT_IN_BYTES_FOR_INT32); \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int32_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_SRAI64(_ae_int64_acc_ ## idx, 16); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_8bx16b(idx) \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_SRAI64(_ae_int64_acc_ ## idx, 8); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_16bx8b ADD_BIAS_16b_ACC_FOR_8bx16b

#define ADD_BIAS_64b_ACC_FOR_8bx16b(idx) \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias, INCREMENT_IN_BYTES_FOR_INT64); \
  /* Saturate 64b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int64_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_SRAI64(_ae_int64_acc_ ## idx, 8); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_16bx16b(idx) \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_64b_ACC_FOR_16bx16b(idx) \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias, INCREMENT_IN_BYTES_FOR_INT64); \
  /* Saturate 64b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int64_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#if XCHAL_HAVE_HIFI1
#define ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx) \
  AE_L32_IP(_ae_int32x2_bias, (ae_int32*)_WORD32_p_bias, 4); \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_bias), 32); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#else
#define ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx) \
    /* Load 32b bias */ \
  _WORD32_bias = *_WORD32_p_bias++; \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#endif

/*------------------ time batching macros ----------------- */
#define ADD_BIAS_BATCH_ROW_8b_ACC_FOR_8bx8b(idx_row)\
  LOAD_BIAS_8b_FOR_8bx8b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_8b_ACC_FOR_8bx8b_MATMUL(idx_row)\
  LOAD_BIAS_8b_FOR_8bx8b_MATMUL; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_8bx16b(idx_row)\
  LOAD_BIAS_16b_FOR_8bx16b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_8bx16b_MATMUL(idx_row)\
  LOAD_BIAS_16b_FOR_8bx16b_MATMUL; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx8b(idx_row)\
  LOAD_BIAS_16b_FOR_16bx8b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx16b(idx_row)\
  LOAD_BIAS_16b_FOR_16bx16b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx16b_MATMUL(idx_row)\
  LOAD_BIAS_16b_FOR_16bx16b_MATMUL; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  LOAD_BIAS_ASYM8b \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row); \

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(idx_row) \
  LOAD_BIAS_ASYM8b_MATMUL \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row); \

#define ADD_BIAS_BATCH_8b_ACC_FOR_8bx8b(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SRAI64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, 16); \
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \

#define ADD_BIAS_BATCH_8b_ACC_FOR_8bx8b_MATMUL(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SRAI64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, 16); \
  if(p_bias!=NULL)\
  {\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \
  }

#define ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SRAI64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, 8); \
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \

#define ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b_MATMUL(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SRAI64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, 8); \
  if(p_bias!=NULL)\
  {\
    _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \
  }

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b_MATMUL(idx_row,idx_vec)\
  if(p_bias!=NULL)\
  {\
    _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \
  }

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx8b               ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b
#define ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b     ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b

#define ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL     ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b_MATMUL

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32(idx_row)\
  LOAD_BIAS_f32; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32_MATMUL(idx_row)\
  LOAD_BIAS_f32_MATMUL; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ACC_FOR_f32(idx_row,idx_vec)\
  _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_RADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec);\
  _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_ADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_bias); \

#define ADD_BIAS_BATCH_ACC_FOR_f32_MATMUL(idx_row,idx_vec)\
  _xtfloat_acc1_ ##idx_row ##_ ##idx_vec = XT_RADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec);\
  _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_ADD_S(_xtfloat_acc1_ ##idx_row ##_ ##idx_vec, _xtfloat_acc_ ##idx_row ##_ ##idx_vec); \
  if(p_bias!=NULL)\
  {\
    _xtfloat_acc_ ##idx_row ##_ ##idx_vec = XT_ADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_bias); \
  }

#define STORE_ACC_8bx8b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \


#if XCHAL_HAVE_HIFI1
#define STORE_ACC_8bx8b_AT_OUT_8b(idx) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
	AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx = AE_SAT16X4( _ae_f32x2_tmp_var_ ## idx, _ae_f32x2_tmp_var_ ## idx ); \
  _ae_int16x4_tmp_var_ ## idx = AE_SAT8S( _ae_int16x4_tmp_var_ ## idx ); \
  AE_S8_0_I_HIFI1( _ae_int16x4_tmp_var_ ## idx, (WORD8 *)(p_out + m_itr + idx), 0 ); \

#else
#define STORE_ACC_8bx8b_AT_OUT_8b(idx) \
  ae_int32 _ae_int32_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)), 24); \
  _ae_int32_tmp_var_ ## idx = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx, 24); \
  (*((WORD8 *) p_out + m_itr + idx)) =(WORD8)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx)); \

#endif

#if XCHAL_HAVE_HIFI1
#define STORE_ACC_8bx8b_AT_OUT_16b(idx) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx = AE_SAT16X4(_ae_f32x2_tmp_var_ ## idx, _ae_f32x2_tmp_var_ ## idx); \
  AE_S16_0_I(_ae_int16x4_tmp_var_ ## idx, (ae_int16 *)(p_out + m_itr + idx), 0); \

#else
#define STORE_ACC_8bx8b_AT_OUT_16b(idx) \
      ae_int32 _ae_int32_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)), 16); \
  _ae_int32_tmp_var_ ## idx = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx, 16); \
  (*((WORD16 *) p_out + m_itr + idx)) =(WORD16)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx)); \

#endif

#define STORE_ACC_8bx8b_AT_OUT_32b(idx) \
  (*((ae_int32 *) p_out + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#if XCHAL_HAVE_HIFI1
#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx) \
  _ae_int32x2_acc_ ## idx = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ## idx, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ## idx), (WORD8 *)p_out+m_itr+idx, 0); \

#else
#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx) \
  _ae_int32x2_acc_ ## idx = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ## idx, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) p_out + m_itr + idx)) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ## idx); \

#endif
/* ==================================================================================================== */
#define STORE_ACC_8bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#if XCHAL_HAVE_HIFI1
#define STORE_ACC_8bx16b_AT_OUT_16b(idx) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx = AE_SAT16X4(_ae_f32x2_tmp_var_ ## idx, _ae_f32x2_tmp_var_ ## idx); \
  AE_S16_0_I(_ae_int16x4_tmp_var_ ## idx, (ae_int16 *)(p_out + m_itr + idx), 0); \

#else
#define STORE_ACC_8bx16b_AT_OUT_16b(idx) \
  ae_int32 _ae_int32_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)), 16); \
  _ae_int32_tmp_var_ ## idx = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx, 16); \
  (*((WORD16 *) p_out + m_itr + idx)) =(WORD16)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx)); \

#endif

#define STORE_ACC_16bx8b_AT_OUT_16b STORE_ACC_8bx16b_AT_OUT_16b

#define STORE_ACC_8bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *) p_out + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_8bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *) p_out + m_itr + idx)) = \
  AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift); \

/* ==================================================================================================== */
#define STORE_ACC_16bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#if XCHAL_HAVE_HIFI1
#define STORE_ACC_16bx16b_AT_OUT_16b(idx) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
   AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx = AE_SAT16X4(_ae_f32x2_tmp_var_ ## idx, _ae_f32x2_tmp_var_ ## idx); \
  AE_S16_0_I(_ae_int16x4_tmp_var_ ## idx, (ae_int16 *)(p_out + m_itr + idx), 0); \

#else
#define STORE_ACC_16bx16b_AT_OUT_16b(idx) \
      ae_int32 _ae_int32_tmp_var_ ## idx; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)), 16); \
  _ae_int32_tmp_var_ ## idx = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx, 16); \
  (*((WORD16 *) p_out + m_itr + idx)) =(WORD16)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx)); \

#endif

#define STORE_ACC_16bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *) p_out + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_16bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *) p_out + m_itr + idx)) = \
  AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift); \

/*------------------ time batching macros ----------------- */
#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_8b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_8bx8b_AT_OUT_32b(idx_row,idx_vec) \
  (*((ae_int32 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)); \

#if XCHAL_HAVE_HIFI1 && (XCHAL_HW_VERSION >= 281090)
#define STORE_ACC_BATCH_8bx8b_AT_OUT_32bx2(idx_row) \
  _ae_int64_acc_ ## idx_row ##_0 = AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_0, acc_shift); \
  _ae_int64_acc_ ## idx_row ##_1 = AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_1, acc_shift); \
  ae_f32x2 _ae_f32x2_tmp_var_ ##idx_row = AE_ROUND32X2F64SSYM(_ae_int64_acc_ ## idx_row ##_0, _ae_int64_acc_ ## idx_row ##_1); \
  AE_S32_H_I(_ae_f32x2_tmp_var_ ##idx_row, ((ae_int32 *) p_out[vec_itr + 0] + m_itr + idx_row), 0); \
  AE_S32_L_I(_ae_f32x2_tmp_var_ ##idx_row, ((ae_int32 *) p_out[vec_itr + 1] + m_itr + idx_row), 0); \

#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32bx2(idx_row)\
  STORE_ACC_BATCH_8bx8b_AT_OUT_32bx2(idx_row);\

#endif

#define STORE_ACC_BATCH_8bx8b_AT_OUT_8b(idx_row,idx_vec) \
  ae_int32 _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)), 24); \
  _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, 24); \
  (*((WORD8 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = (*((UWORD32 *)&_ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec)); \

#if XCHAL_HAVE_HIFI1
#define STORE_STRIDE_ACC_BATCH_8bx8b_AT_OUT_8b(idx_row,idx_vec) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SAT16X4(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec); \
  _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SAT8S(_ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec); \
  AE_S8_0_X_HIFI1( _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec, (WORD8 *)p_out, ((vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride) ); \

#else
#define STORE_STRIDE_ACC_BATCH_8bx8b_AT_OUT_8b(idx_row,idx_vec) \
  ae_int32 _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)), 24); \
  _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, 24); \
  (*((WORD8 *) p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)) =(WORD8)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec)); \

#endif

#define STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_16bx8b_AT_OUT_16b STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_16b STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_8bx16b_AT_OUT_64b(idx_row,idx_vec) \
  (*((ae_int64 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_SLAA64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift); \

#define STORE_ACC_BATCH_8bx16b_AT_OUT_16b(idx_row,idx_vec) \
  STORE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec); \

#define STORE_STRIDE_ACC_BATCH_8bx16b_AT_OUT_16b(idx_row,idx_vec) \
  STORE_STRIDE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec); \

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_16b STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_16bx16b_AT_OUT_64b(idx_row,idx_vec) \
  (*((ae_int64 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_SLAA64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift); \

#define STORE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec) \
      ae_int32 _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)), 16); \
  _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, 16); \
  (*((WORD16 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = (*((UWORD32 *)&_ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec)); \

#if XCHAL_HAVE_HIFI1
#define STORE_STRIDE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec) \
  ae_int16x4 _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)); \
  _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SAT16X4(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec); \
  AE_S16_0_I( _ae_int16x4_tmp_var_ ## idx_row ##_ ##idx_vec, (ae_int16 *)(p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride), 0 ); \

#else
#define STORE_STRIDE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec) \
  ae_int32 _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f32x2 _ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)), 16); \
  _ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SRAI32(_ae_f32x2_tmp_var_ ## idx_row ##_ ##idx_vec, 16); \
  (*((WORD16 *) p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)) =(WORD16)(*((UWORD32 *)&_ae_int32_tmp_var_ ## idx_row ##_ ##idx_vec)); \

#endif

#define STORE_ACC_BATCH_ROW_AT_OUT_f32(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_AT_OUT_f32(idx_row,idx_vec) \
  /*p_out value stored in a tmp pointer to make it inout for ISA */\
  p_out_tmp = (p_out[vec_itr + idx_vec] + m_itr + idx_row);\
  XT_SSIP(_xtfloat_acc_ ##idx_row ##_ ##idx_vec,p_out_tmp,0); \

#define STORE_STRIDE_ACC_BATCH_AT_OUT_f32(idx_row,idx_vec) \
  /*p_out value stored in a tmp pointer to make it inout for ISA */\
  p_out_tmp = p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride;\
  XT_SSIP(_xtfloat_acc_ ##idx_row ##_ ##idx_vec,p_out_tmp,0); \

#define STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row); \

#if XCHAL_HAVE_HIFI1
#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ## idx_row ##_ ##idx_vec), ((WORD8 *) (p_out[vec_itr + idx_vec] + m_itr + idx_row)) , 0); \

#else //HIFI_LE
#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) (p_out[vec_itr + idx_vec] + m_itr + idx_row))) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \

#endif //HIFI_LE

#if XCHAL_HAVE_HIFI1
#define STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec), ((WORD8 *) (p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)), 0); \

#if (XCHAL_HW_VERSION >= 281090)
#define STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(idx_row,idx_vec) \
  ae_int8x8  _ae_int8x8_acc_ ##idx_row ##_ ##idx_vec = AE_SAT8X4X32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \
  AE_S8_0_I(_ae_int8x8_acc_ ##idx_row ##_ ##idx_vec, ((ae_int8  *) (p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)), 0); \

#else //XCHAL_HW_VERSION
#define STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(-128)), AE_MOVDA32(127)); \
  AE_S8_0_I_HIFI1(AE_MOVINT16X4_FROMINT32X2(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec), ((WORD8 *) (p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)), 0); \

#endif //XCHAL_HW_VERSION

#else //XCHAL_HAVE_HIFI1
#define STORE_STRIDE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \

#define STORE_STRIDE_ACC_BATCH_ASYM8bsxASYM8bs_AT_OUT_ASYM8bs(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(-128)), AE_MOVDA32(127)); \
  (*((WORD8 *) p_out + (vec_itr + idx_vec)*out_offset + (m_itr + idx_row)*out_stride)) = (WORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \


#endif //XCHAL_HAVE_HIFI1
/*---------------------------------------------------------*/
/* Specific macros needed for extra calculations involved
  for ASYM8b */

/* This is written to match with Tensorflow */
#define ADJUST_ACC_ASYM8b(idx) \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */ \
  ae_int32x2 _ae_int32x2_acc_ ## idx; \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ## idx, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ## idx), out_multiplier, left_shift, right_shift); \
  /* Add output zero point */ \
  (_ae_int32x2_acc_ ## idx) = AE_ADD32S(_ae_int32x2_acc_ ## idx, AE_MOVDA32(out_zero_bias)); \


/* For time batching */
#define ADJUST_ACC_BATCH_ROW_ASYM8b(idx_row) \
  ADJUST_ACC_BATCH_VEC_UNROLL(idx_row); \

/* For time batching */

#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec) \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */ \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec; \
  MPY_BY_QUANT_MULT_X2_OUT32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec), out_multiplier, left_shift, right_shift); \
  /* Add output zero point */ \
  (_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(out_zero_bias)); \

/*---------------------------------------------------------*/
/* ==================================================================================================== */
#if (ROW_UNROLL == 1)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)
#define SETUP_MAT1_X2        UNROLL_SETUP_MAT1_X2(0)
#define SETUP_MAT2_X2        UNROLL_SETUP_MAT2_X2(0)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)
#define KERNEL_MAT1_VEC1_X2  UNROLL_KERNEL_MAT1_VEC1_LOAD_X2(0)          UNROLL_KERNEL_MAT1_VEC1_MAC_X2(0)
#define KERNEL_MAT2_VEC2_X2  UNROLL_KERNEL_MAT2_VEC2_LOAD_X2(0)          UNROLL_KERNEL_MAT2_VEC2_MAC_X2(0)
#define KERNEL_MAT1_VEC1_X2 UNROLL_KERNEL_MAT1_VEC1_X2(0)
#define KERNEL_MAT2_VEC2_X2  UNROLL_KERNEL_MAT2_VEC2_X2(0)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)
#define STORE_ACC            UNROLL_STORE_ACC(0)

#elif (ROW_UNROLL == 2)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)
#define SETUP_MAT1_X2        UNROLL_SETUP_MAT1_X2(0)        UNROLL_SETUP_MAT1_X2(1)
#define SETUP_MAT2_X2        UNROLL_SETUP_MAT2_X2(0)        UNROLL_SETUP_MAT2_X2(1)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)
#define KERNEL_MAT1_VEC1_X2  UNROLL_KERNEL_MAT1_VEC1_X2(0)  UNROLL_KERNEL_MAT1_VEC1_X2(1)
#define KERNEL_MAT2_VEC2_X2  UNROLL_KERNEL_MAT2_VEC2_X2(0)  UNROLL_KERNEL_MAT2_VEC2_X2(1)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)

#elif (ROW_UNROLL == 4)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)           UNROLL_SETUP_MAT2(2)           UNROLL_SETUP_MAT2(3)
#define SETUP_MAT1_X2        UNROLL_SETUP_MAT1_X2(0)        UNROLL_SETUP_MAT1_X2(1)        UNROLL_SETUP_MAT1_X2(2)        UNROLL_SETUP_MAT1_X2(3)
#define SETUP_MAT2_X2        UNROLL_SETUP_MAT2_X2(0)        UNROLL_SETUP_MAT2_X2(1)        UNROLL_SETUP_MAT2_X2(2)        UNROLL_SETUP_MAT2_X2(3)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)     UNROLL_KERNEL_MAT1_VEC1(2)     UNROLL_KERNEL_MAT1_VEC1(3)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)     UNROLL_KERNEL_MAT2_VEC2(2)     UNROLL_KERNEL_MAT2_VEC2(3)
#define KERNEL_MAT1_VEC1_X2  UNROLL_KERNEL_MAT1_VEC1_X2(0)  UNROLL_KERNEL_MAT1_VEC1_X2(1)  UNROLL_KERNEL_MAT1_VEC1_X2(2)  UNROLL_KERNEL_MAT1_VEC1_X2(3)
#define KERNEL_MAT2_VEC2_X2  UNROLL_KERNEL_MAT2_VEC2_X2(0)  UNROLL_KERNEL_MAT2_VEC2_X2(1)  UNROLL_KERNEL_MAT2_VEC2_X2(2)  UNROLL_KERNEL_MAT2_VEC2_X2(3)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)

#elif (ROW_UNROLL == 8)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)            UNROLL_SETUP_ACC(4)            UNROLL_SETUP_ACC(5)            UNROLL_SETUP_ACC(6)            UNROLL_SETUP_ACC(7)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)           UNROLL_SETUP_MAT1(4)           UNROLL_SETUP_MAT1(5)           UNROLL_SETUP_MAT1(6)           UNROLL_SETUP_MAT1(7)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)           UNROLL_SETUP_MAT2(2)           UNROLL_SETUP_MAT2(3)           UNROLL_SETUP_MAT2(4)           UNROLL_SETUP_MAT2(5)           UNROLL_SETUP_MAT2(6)           UNROLL_SETUP_MAT2(7)
#define SETUP_MAT1_X2        UNROLL_SETUP_MAT1_X2(0)        UNROLL_SETUP_MAT1_X2(1)        UNROLL_SETUP_MAT1_X2(2)        UNROLL_SETUP_MAT1_X2(3)        UNROLL_SETUP_MAT1_X2(4)        UNROLL_SETUP_MAT1_X2(5)        UNROLL_SETUP_MAT1_X2(6)        UNROLL_SETUP_MAT1_X2(7)
#define SETUP_MAT2_X2        UNROLL_SETUP_MAT2_X2(0)        UNROLL_SETUP_MAT2_X2(1)        UNROLL_SETUP_MAT2_X2(2)        UNROLL_SETUP_MAT2_X2(3)        UNROLL_SETUP_MAT2_X2(4)        UNROLL_SETUP_MAT2_X2(5)        UNROLL_SETUP_MAT2_X2(6)        UNROLL_SETUP_MAT2_X2(7)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)     UNROLL_KERNEL_MAT1_VEC1(2)     UNROLL_KERNEL_MAT1_VEC1(3)     UNROLL_KERNEL_MAT1_VEC1(4)     UNROLL_KERNEL_MAT1_VEC1(5)     UNROLL_KERNEL_MAT1_VEC1(6)     UNROLL_KERNEL_MAT1_VEC1(7)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)     UNROLL_KERNEL_MAT2_VEC2(2)     UNROLL_KERNEL_MAT2_VEC2(3)     UNROLL_KERNEL_MAT2_VEC2(4)     UNROLL_KERNEL_MAT2_VEC2(5)     UNROLL_KERNEL_MAT2_VEC2(6)     UNROLL_KERNEL_MAT2_VEC2(7)
#define KERNEL_MAT1_VEC1_X2  UNROLL_KERNEL_MAT1_VEC1_X2(0)  UNROLL_KERNEL_MAT1_VEC1_X2(1)  UNROLL_KERNEL_MAT1_VEC1_X2(2)  UNROLL_KERNEL_MAT1_VEC1_X2(3)  UNROLL_KERNEL_MAT1_VEC1_X2(4)  UNROLL_KERNEL_MAT1_VEC1_X2(5)  UNROLL_KERNEL_MAT1_VEC1_X2(6)  UNROLL_KERNEL_MAT1_VEC1_X2(7)
#define KERNEL_MAT2_VEC2_X2  UNROLL_KERNEL_MAT2_VEC2_X2(0)  UNROLL_KERNEL_MAT2_VEC2_X2(1)  UNROLL_KERNEL_MAT2_VEC2_X2(2)  UNROLL_KERNEL_MAT2_VEC2_X2(3)  UNROLL_KERNEL_MAT2_VEC2_X2(4)  UNROLL_KERNEL_MAT2_VEC2_X2(5)  UNROLL_KERNEL_MAT2_VEC2_X2(6)  UNROLL_KERNEL_MAT2_VEC2_X2(7)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)         UNROLL_ADD_BIAS_ACC(4)         UNROLL_ADD_BIAS_ACC(5)         UNROLL_ADD_BIAS_ACC(6)         UNROLL_ADD_BIAS_ACC(7)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)           UNROLL_ADJUST_ACC(4)           UNROLL_ADJUST_ACC(5)           UNROLL_ADJUST_ACC(6)           UNROLL_ADJUST_ACC(7)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)            UNROLL_STORE_ACC(4)            UNROLL_STORE_ACC(5)            UNROLL_STORE_ACC(6)            UNROLL_STORE_ACC(7)

#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH                           UNROLL_SETUP_VEC_BATCH(0)               UNROLL_SETUP_VEC_BATCH(1)
#define SETUP_VEC_BATCH_X2                        UNROLL_SETUP_VEC_BATCH_X2(0)            UNROLL_SETUP_VEC_BATCH_X2(1)

#define SETUP_ACC_BATCH                           UNROLL_ROW_SETUP_ACC_BATCH(0)           UNROLL_ROW_SETUP_ACC_BATCH(1)       UNROLL_ROW_SETUP_ACC_BATCH(2)       UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)       UNROLL_SETUP_ACC_BATCH(idx_row,0)       UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define SETUP_ACC_BATCH_TAIL                      UNROLL_SETUP_ACC_BATCH(0,0)             UNROLL_SETUP_ACC_BATCH(1,0)         UNROLL_SETUP_ACC_BATCH(2,0)         UNROLL_SETUP_ACC_BATCH(3,0)

#define LOAD_VEC_BATCH                            UNROLL_LOAD_VEC_BATCH(0)                UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_VEC_BATCH_X2                         UNROLL_LOAD_VEC_BATCH_X2(0)             UNROLL_LOAD_VEC_BATCH_X2(1)
#define LOAD_MAT1                                 UNROLL_LOAD_ROW_MAT1(0)                 UNROLL_LOAD_ROW_MAT1(1)             UNROLL_LOAD_ROW_MAT1(2)             UNROLL_LOAD_ROW_MAT1(3)
#define LOAD_MAT1_X2                              UNROLL_LOAD_ROW_MAT1_X2(0)              UNROLL_LOAD_ROW_MAT1_X2(1)          UNROLL_LOAD_ROW_MAT1_X2(2)          UNROLL_LOAD_ROW_MAT1_X2(3)

#define KERNEL_MAT1_VEC_BATCH                     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0)     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(2) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(3)
#define KERNEL_MAT1_VEC_BATCH_X2                  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2(0)  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2(1) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2(2) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH_X2(3)
#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row) UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,0) UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,1)
#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL_X2(idx_row) UNROLL_KERNEL_MAT1_VEC_BATCH_X2(idx_row,0) UNROLL_KERNEL_MAT1_VEC_BATCH_X2(idx_row,1)
#define KERNEL_MAT1_VEC_BATCH_TAIL                UNROLL_KERNEL_MAT1_VEC_BATCH(0,0)       UNROLL_KERNEL_MAT1_VEC_BATCH(1,0)   UNROLL_KERNEL_MAT1_VEC_BATCH(2,0)   UNROLL_KERNEL_MAT1_VEC_BATCH(3,0)

#define ADD_BIAS_ACC_BATCH                        UNROLL_ROW_ADD_BIAS_ACC(0)              UNROLL_ROW_ADD_BIAS_ACC(1)          UNROLL_ROW_ADD_BIAS_ACC(2)          UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)    UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)    UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define ADD_BIAS_ACC_BATCH_TAIL                   LOAD_BIAS                               UNROLL_ADD_BIAS_ACC_BATCH(0,0)      LOAD_BIAS                           UNROLL_ADD_BIAS_ACC_BATCH(1,0)      LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(2,0) LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(3,0)

#define STORE_ACC_BATCH                           UNROLL_ROW_STORE_ACC(0)                 UNROLL_ROW_STORE_ACC(1)             UNROLL_ROW_STORE_ACC(2)             UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)       UNROLL_STORE_ACC_BATCH(idx_row,0)       UNROLL_STORE_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_TAIL                      UNROLL_STORE_ACC_BATCH(0,0)             UNROLL_STORE_ACC_BATCH(1,0)         UNROLL_STORE_ACC_BATCH(2,0)         UNROLL_STORE_ACC_BATCH(3,0)

#define ADJUST_ACC_BATCH_TAIL                     UNROLL_ADJUST_ACC_BATCH(0, 0)           UNROLL_ADJUST_ACC_BATCH(1, 0)       UNROLL_ADJUST_ACC_BATCH(2, 0)       UNROLL_ADJUST_ACC_BATCH(3, 0)
#define ADJUST_ACC_BATCH                          UNROLL_ROW_ADJUST_ACC(0)                UNROLL_ROW_ADJUST_ACC(1)                UNROLL_ROW_ADJUST_ACC(2)            UNROLL_ROW_ADJUST_ACC(3)
#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row)      UNROLL_ADJUST_ACC_BATCH(idx_row,0)      UNROLL_ADJUST_ACC_BATCH(idx_row,1)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/

#endif /* __XA_NNLIB_COMMON_MACROS_H__ */
