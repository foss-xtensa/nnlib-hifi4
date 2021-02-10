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

WORD32 xa_nn_elm_quantize_asym16s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD8 *out = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  
  ae_int32x2 inp_z_b = AE_MOVDA32(inp_zero_bias);
  ae_int32x2 out_mult = AE_MOVDA32(out_multiplier);
  ae_int32x2 quant_min = AE_MOVDA32(-128);
  ae_int32x2 quant_max = AE_MOVDA32(127);
  
  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 inp0;
    ae_int32x2 inp32, inp10;
    ae_int32x2 unclamped_out32, unclamped_out10;
    ae_int32x2 clamped_out32, clamped_out10;

    AE_LA16X4_IP(inp0, align_inp, (ae_int16x4 *)p_i);

    inp32 = AE_SEXT32X2D16_32(inp0);
    inp10 = AE_SEXT32X2D16_10(inp0);

    inp32 = AE_SUB32S(inp32, inp_z_b);
    inp10 = AE_SUB32S(inp10, inp_z_b);

    // unclamped result
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out32, inp32, out_mult, left_shift, right_shift)
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out10, inp10, out_mult, left_shift, right_shift)
    unclamped_out32 = AE_ADD32(unclamped_out32, out_zero_bias);
    unclamped_out10 = AE_ADD32(unclamped_out10, out_zero_bias);

    // clamped_out
    CLAMP_VAL(clamped_out32, unclamped_out32, quant_min, quant_max)
    CLAMP_VAL(clamped_out10, unclamped_out10, quant_min, quant_max)

    // Store Output
    STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
  }

  // Remainder Loop
  for(i = 0; i < (num_elm & 3); i++)
  {
    int inp;
    ae_int32x2 inp_HL;
    ae_int32x2 unclamped_out_HL;
    ae_int32x2 clamped_out_HL;

    inp = (int)p_i[i];
    inp_HL = AE_MOVDA32(inp);
    inp_HL = AE_SUB32S(inp_HL, inp_z_b);

    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(unclamped_out_HL, inp_HL, out_mult, left_shift, right_shift)
    unclamped_out_HL = AE_ADD32(unclamped_out_HL, out_zero_bias);
    
    // clamped_out
    CLAMP_VAL(clamped_out_HL, unclamped_out_HL, quant_min, quant_max)

    *out++ = (WORD8)(AE_MOVAD32_H(clamped_out_HL));
  }
  return 0;  
}

