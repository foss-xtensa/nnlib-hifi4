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
/* Common helper macros. */
#include "common_fpu.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_round_f32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                WORD32 num_elm
              )
           )
#else
#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_elm_round_f32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  int i = 0;
  xtfloatx2 *inp = (xtfloatx2 *)p_inp;
  xtfloatx2 *out = (xtfloatx2 *)p_out;
  xtfloatx2 x1, x2, y1, y2;

  WUR_FCR(0); // Setting the rounding mode

  if(((((unsigned)p_out) & 7) == 0) && ((((unsigned)p_inp) & 7) == 0))
  {
#pragma no_unroll
    for(i = 0; i < (num_elm >> 2); i++)
    {
      AE_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
      AE_LSX2IP(x2, inp, 2*sizeof(FLOAT32));
      y1 = FIRINT_SX2(x1);
      y2 = FIRINT_SX2(x2);
      AE_SSX2IP(y1, out, 2*sizeof(FLOAT32));
      AE_SSX2IP(y2, out, 2*sizeof(FLOAT32));
    }
  }
  else
  {
    ae_valign inp_a, out_a;

    inp_a = AE_LA64_PP(inp);
    out_a = AE_ZALIGN64();

#pragma no_unroll
    for(i = 0; i < (num_elm >> 2); i++)
    {
      AE_LASX2IP(x1, inp_a, inp);
      AE_LASX2IP(x2, inp_a, inp);
      y1 = FIRINT_SX2(x1);
      y2 = FIRINT_SX2(x2);
      AE_SASX2IP(y1, out_a, out);
      AE_SASX2IP(y2, out_a, out);
    }
    AE_SA64POS_FP(out_a, out);
  }

  // Remainder Loop
  i = num_elm & (~3);
  for(; i < num_elm; i++)
  {
    xtfloat a1, a;
    AE_LSIP(a1, (xtfloat *)inp, sizeof(FLOAT32));
    a = FIRINT_S(a1);
    AE_SSIP(a, (xtfloat *)out, sizeof(FLOAT32));
  }

  return 0;

}
#else
WORD32 xa_nn_elm_round_f32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               WORD32 num_elm)
{
 /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  
  int i = 0;
  xtfloatx2 *inp = (xtfloatx2 *)p_inp;
  xtfloatx2 *out = (xtfloatx2 *)p_out;
  xtfloatx2 x1, y1;

  XT_WUR_FCR(0); // Setting the rounding mode

  if(((((unsigned)p_out) & 7) == 0) && ((((unsigned)p_inp) & 7) == 0))
  {
    for(i = 0; i < (num_elm >> 1); i++)
    {
      XT_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
      y1 = XT_FIRINT_SX2(x1);
      XT_SSX2IP(y1, out, 2*sizeof(FLOAT32));
    }
  }
  else
  {
    ae_valign inp_a, out_a;

    inp_a = XT_LASX2PP(inp);
    out_a = AE_ZALIGN64();

    for(i = 0; i < (num_elm >> 1); i++)
    {
      XT_LASX2IP(x1, inp_a, inp);
      y1 = XT_FIRINT_SX2(x1);
      XT_SASX2IP(y1, out_a, out);
    }
    XT_SASX2POSFP(out_a, out);
  }

  // Remainder Loop
  if (num_elm & 1)
  {
    xtfloat a1, a;
    XT_LSIP(a1, (xtfloat *)inp, 0);
    a = XT_FIRINT_S(a1);
    XT_SSI(a, (xtfloat *)out, 0);
  }

  return 0;
}
#endif
#endif /* !HAVE_VFPU */
