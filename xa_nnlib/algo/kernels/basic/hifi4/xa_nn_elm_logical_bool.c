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
#include "xa_nn_basic_state.h"

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_logicaland_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    WORD8 *pin1 = (WORD8 *)p_inp1;
    WORD8 *pin2 = (WORD8 *)p_inp2;
    WORD8 *pout = (WORD8 *)p_out;
 
    int remainder_start;
    int i;
    int N = num_elm;

    
    ae_int16x4 vi1, vi2, vo; 
    ae_valign align_src_in1, align_src_in2, align_dst;
    align_src_in1 = AE_LA64_PP(pin1);
    align_src_in2 = AE_LA64_PP(pin2);
    align_dst    = AE_ZALIGN64();

    if(((((unsigned)pin1)&1) == 0) && ((((unsigned)pin2)&1) == 0) && ((((unsigned)pout)&1) == 0))
    {
      int Nby8 =  N >> 3 ; 
      remainder_start = Nby8 << 3;
      
      /* Loop is unrolled by 8, to use LA16X4/SA16X4 */
      for(i=0; i < Nby8; i++)
      {
        AE_LA16X4_IP(vi1, align_src_in1, (ae_int16x4 *)pin1);
        AE_LA16X4_IP(vi2, align_src_in2, (ae_int16x4 *)pin2);
        vo = AE_AND16(vi1,vi2);
        AE_SA16X4_IP(vo, align_dst, (ae_int16x4 *)pout);
      }

    }
    else
    {
      int Nby4 =  N >> 2 ; 
      remainder_start = Nby4 << 2;
      
      /* Loop is unrolled by 4, to use LA8X4/SA8X4 */
      for(i=0; i < Nby4; i++)
      {
        AE_LA8X4S_IP(vi1, align_src_in1, pin1);
        AE_LA8X4S_IP(vi2, align_src_in2, pin2);
        vo = AE_AND16(vi1,vi2);
        AE_SA8X4U_IP(vo, align_dst, (ae_int32 *)pout);
       }
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder loop */
    for(i=remainder_start; i < N; i++){
      AE_L8S_IP(vi1, pin1, 1);
      AE_L8S_IP(vi2, pin2, 1);
      vo = AE_AND16(vi1,vi2);
      AE_S8_0_IP_HIFI1(vo, pout, 1);
    }

    return 0;
}
#else
WORD32 xa_nn_elm_logicaland_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    ae_int24x2 *pin1 = (ae_int24x2 *)p_inp1;
    ae_int24x2 *pin2 = (ae_int24x2 *)p_inp2;
    ae_int24x2 *pout = (ae_int24x2 *)p_out;
    int i;
    int N = num_elm;
    /* Following line divides N by 6. Much faster than compiler implementation. Works for N<32768. */ 
    /* unsigned int Nby6 = (N*10923)>>16;*/
    /* Following works for all int32 N */
    int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(N, 0x2AAAAAAB)));
    int remainder_start = 6*Nby6;

    ae_valign align_src_in1, align_src_in2, align_dst;
    align_src_in1 = AE_LA64_PP(pin1);
    align_src_in2 = AE_LA64_PP(pin2);
    align_dst    = AE_ZALIGN64();

/* Loop is unrolled by 6, to use LA24X2/SA24X2 */
    for(i=0; i < Nby6; i++){
        ae_int24x2 vi1, vi2, vo;
        AE_LA24X2_IP(vi1, align_src_in1, pin1);
        AE_LA24X2_IP(vi2, align_src_in2, pin2);
        vo = AE_AND24(vi1, vi2);
        AE_SA24X2_IP(vo, align_dst, pout);
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder loop */
    #pragma no_unroll
    for(i=remainder_start; i < N; i++){
        p_out[i] = p_inp1[i] & p_inp2[i];
    }

    return 0;
}
#endif

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_logicalor_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    WORD8 *pin1 = (WORD8 *)p_inp1;
    WORD8 *pin2 = (WORD8 *)p_inp2;
    WORD8 *pout = (WORD8 *)p_out;
    int remainder_start;
    int i;
    int N = num_elm;

    
    ae_int16x4 vi1, vi2, vo; 
    ae_valign align_src_in1, align_src_in2, align_dst;
    align_src_in1 = AE_LA64_PP(pin1);
    align_src_in2 = AE_LA64_PP(pin2);
    align_dst    = AE_ZALIGN64();

    if(((((unsigned)pin1)&1) == 0) && ((((unsigned)pin2)&1) == 0) && ((((unsigned)pout)&1) == 0))
    {
      int Nby8 =  N >> 3 ; 
      remainder_start = Nby8 << 3;
      
      /* Loop is unrolled by 8, to use LA16X4/SA16X4 */
      for(i=0; i < Nby8; i++)
      {
        AE_LA16X4_IP(vi1, align_src_in1, (ae_int16x4 *)pin1);
        AE_LA16X4_IP(vi2, align_src_in2, (ae_int16x4 *)pin2);
        vo = AE_OR16(vi1,vi2);
        AE_SA16X4_IP(vo, align_dst, (ae_int16x4 *)pout);
      }
    }
    else
    {
      int Nby4 =  N >> 2 ; 
      remainder_start = Nby4 << 2;
      
      /* Loop is unrolled by 4, to use LA8X4/SA8X4 */
      for(i=0; i < Nby4; i++)
      {
        AE_LA8X4S_IP(vi1, align_src_in1, pin1);
        AE_LA8X4S_IP(vi2, align_src_in2, pin2);
        vo = AE_OR16(vi1,vi2);
        AE_SA8X4U_IP(vo, align_dst, (ae_int32 *)pout);
       }
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder loop */
    for(i=remainder_start; i < N; i++){
      AE_L8S_IP(vi1, pin1, 1);
      AE_L8S_IP(vi2, pin2, 1);
      vo = AE_OR16(vi1,vi2);
      AE_S8_0_IP_HIFI1(vo, pout, 1);
    }

    return 0;
}
#else
WORD32 xa_nn_elm_logicalor_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    ae_int24x2 *pin1 = (ae_int24x2 *)p_inp1;
    ae_int24x2 *pin2 = (ae_int24x2 *)p_inp2;
    ae_int24x2 *pout = (ae_int24x2 *)p_out;
    int i;
    int N = num_elm;
    /* Following line divides N by 6. Much faster than compiler implementation. Works for N<32768. */ 
    /* unsigned int Nby6 = (N*10923)>>16;*/
    /* Following works for all int32 N */
    int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(N, 0x2AAAAAAB)));
    int remainder_start = 6*Nby6;

    ae_valign align_src_in1, align_src_in2, align_dst;
    align_src_in1 = AE_LA64_PP(pin1);
    align_src_in2 = AE_LA64_PP(pin2);
    align_dst    = AE_ZALIGN64();

    /* Loop is unrolled by 6, to use LA24X2/SA24X2 */
    for(i=0; i < Nby6; i++){
        ae_int24x2 vi1, vi2, vo;
        AE_LA24X2_IP(vi1, align_src_in1, pin1);
        AE_LA24X2_IP(vi2, align_src_in2, pin2);
        vo = AE_OR24(vi1, vi2);
        AE_SA24X2_IP(vo, align_dst, pout);
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder loop */
    #pragma no_unroll
    for(i=remainder_start; i < N; i++){
        p_out[i] = p_inp1[i] | p_inp2[i];
    }

    return 0;
}
#endif

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_logicalnot_bool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    WORD8 *pin = (WORD8 *)p_inp;
    WORD8 *pout = (WORD8 *)p_out;

    ae_int16x4 cnst_notbit = 0x01010101;
    ae_int16x4 vi1, vo;

    int remainder_start;
    int i;
    int N = num_elm;

    ae_valign align_src_in, align_dst;
    align_src_in = AE_LA64_PP(pin);
    align_dst    = AE_ZALIGN64();

    if(((((unsigned)pin)&1) == 0) && ((((unsigned)pout)&1) == 0) )
    {
      int Nby8 =  N >> 3 ; 
      remainder_start = Nby8 << 3;
      
      /* Loop is unrolled by 8, to use LA16X4/SA16X4 */
      for(i=0; i < Nby8; i++)
      {
        AE_LA16X4_IP(vi1, align_src_in, (ae_int16x4 *)pin);
        vo = AE_XOR16(vi1, cnst_notbit); /* NOT operation with lsb implemented using XOR operation */
        AE_SA16X4_IP(vo, align_dst, (ae_int16x4 *)pout);
      }
    }
    else
    {
      int Nby4 =  N >> 2 ; 
      remainder_start = Nby4 << 2;

      /* Loop is unrolled by 4, to use LA8X4/SA8X4 */
      for(i=0; i < Nby4; i++)
      {
        AE_LA8X4S_IP(vi1, align_src_in, pin);
        vo = AE_XOR16(vi1, cnst_notbit); /* NOT operation with lsb implemented using XOR operation */
        AE_SA8X4U_IP(vo, align_dst, (ae_int32 *)pout);
      }
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder Loop */
    for(i=remainder_start; i < num_elm; i++)
    {
      AE_L8S_IP(vi1, pin, 1);
      vo = AE_XOR16(vi1,cnst_notbit);
      AE_S8_0_IP_HIFI1(vo, pout, 1);
    }

    return 0;
}
#else
WORD32 xa_nn_elm_logicalnot_bool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    ae_int24x2 *pin = (ae_int24x2 *)p_inp;
    ae_int24x2 *pout = (ae_int24x2 *)p_out;
    int i;
    int N = num_elm;
    /* Following line divides N by 6. Much faster than compiler implementation. Works for N<32768. */ 
    /* unsigned int Nby6 = (N*10923)>>16;*/
    /* Following works for all int32 N */
    int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(N, 0x2AAAAAAB)));
    int remainder_start = 6*Nby6;

    ae_valign align_src_in, align_dst;
    align_src_in = AE_LA64_PP(pin);
    align_dst    = AE_ZALIGN64();

    ae_int24x2 cnst_notbit = 0x01010101;

    /* Loop is unrolled by 6, to use LA24X2/SA24X2 */
    for(i=0; i < Nby6; i++){
        ae_int24x2 vi1, vo;
        AE_LA24X2_IP(vi1, align_src_in, pin);
        vo = AE_XOR24(vi1, cnst_notbit); /* NOT operation with lsb implemented using XOR operation */
        AE_SA24X2_IP(vo, align_dst, pout);
    }
    AE_SA64POS_FP(align_dst, pout);

    /* Remainder Loop */
    #pragma no_unroll
    for(i=remainder_start; i < num_elm; i++){
        p_out[i] = 2 + ~p_inp[i];
    }

    return 0;
}
#endif
