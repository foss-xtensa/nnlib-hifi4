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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_memset_f32_f32,
             (
                FLOAT32 * __restrict__ p_out,
                FLOAT32 val,
                WORD32 num_elm
              )
           )
#else
#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_memset_f32_f32(FLOAT32 * __restrict__ p_out,
                                FLOAT32 val,
                                WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  //__attribute__ ((aligned (16))) FLOAT32 valueArray[2] = {val, val};
  FLOAT32 valueArray[2] = {val, val};
  int i;
  xtfloatx2 *out =  (xtfloatx2 *)p_out;
  xtfloatx2 *inp  =  (xtfloatx2 *) valueArray;
  xtfloatx2 x1;

  //Loading input values
  ae_valign inp_a;
  if( (((unsigned)valueArray) & 7) == 0)
    AE_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
  else
  {
     inp_a = AE_LA64_PP(inp);
     AE_LASX2IP(x1, inp_a, inp);
  }

  if( (((unsigned)p_out) & 7) == 0)
  {
#pragma loop_count factor=2
     for(i=0;i < num_elm>>1;i++)
     {
        AE_SSX2IP(x1, out, 2*sizeof(FLOAT32));
     }
  }
  else
  {
     ae_valign out_a;
     out_a = AE_ZALIGN64();
#pragma concurrent
#pragma loop_count factor=2
     for(i=0;i < num_elm>>1;i++)
     {
        AE_SASX2IP(x1, out_a, out);
     }
     AE_SA64POS_FP(out_a, out);
  }

  // Remainder Loop
  i <<= 1;
  xtfloat a;

  xtfloat *inp_val = (xtfloat *) valueArray;
  AE_LSIP(a, inp_val, sizeof(FLOAT32));
#pragma loop_count min=0,max=1
  for(; i < num_elm; i++)
  {
     AE_SSIP(a, (xtfloat *)out, sizeof(FLOAT32));
  }

  return 0;
}
#else
WORD32 xa_nn_memset_f32_f32(FLOAT32 * __restrict__ p_out,
                                FLOAT32 val,
                                WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  //__attribute__ ((aligned (16))) FLOAT32 valueArray[2] = {val, val};
  FLOAT32 valueArray[2] = {val, val};
  int i;
  xtfloatx2 *out =  (xtfloatx2 *)p_out;
  xtfloatx2 *inp  =  (xtfloatx2 *) valueArray;
  xtfloatx2 x1;

  //Loading input values
  ae_valign inp_a;
  if( (((unsigned)valueArray) & 7) == 0)
    XT_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
  else
  {
     inp_a = AE_LA64_PP(inp);
     XT_LASX2IP(x1, inp_a, inp);
  }

  if( (((unsigned)p_out) & 7) == 0)
  {
#pragma loop_count factor=2
     for(i=0;i < num_elm>>1;i++)
     {
        XT_SSX2IP(x1, out, 2*sizeof(FLOAT32));
     }
  }
  else
  {
     ae_valign out_a;
     out_a = AE_ZALIGN64();
#pragma concurrent
#pragma loop_count factor=2
     for(i=0;i < num_elm>>1;i++)
     {
        XT_SASX2IP(x1, out_a, out);
     }
     AE_SA64POS_FP(out_a, out);
  }

  // Remainder Loop
  i <<= 1;
  xtfloat a;

  xtfloat *inp_val = (xtfloat *) valueArray;
  XT_LSIP(a, inp_val, sizeof(FLOAT32));
#pragma loop_count min=0,max=1
  for(; i < num_elm; i++)
  {
     XT_SSIP(a, (xtfloat *)out, sizeof(FLOAT32));
  }

  return 0;
}
#endif
#endif /* !HAVE_VFPU */
