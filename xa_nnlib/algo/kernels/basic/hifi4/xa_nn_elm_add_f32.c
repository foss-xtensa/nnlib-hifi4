/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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
#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_bcast_macro.h"


#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_add_f32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp1,
                const FLOAT32 *p_inp2,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_add_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx2 *inp1 = (xtfloatx2 *)p_inp1;
    xtfloatx2 *inp2 = (xtfloatx2 *)p_inp2;
    xtfloatx2 *out =  (xtfloatx2 *)p_out;
    xtfloatx2 x1, x2, y;

    if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
    {
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
            y = XT_ADD_SX2(x1, x2);
            XT_SSX2IP( y, out,  2*sizeof(FLOAT32));
        }
    }
    else
    {
        ae_valign inp1_a, inp2_a, out_a;

        inp1_a = XT_LASX2PP(inp1);
        inp2_a = XT_LASX2PP(inp2);
        out_a = AE_ZALIGN64();
        /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            y = XT_ADD_SX2(x1, x2);
            XT_SASX2IP(y, out_a, out);
        }
        XT_SASX2POSFP(out_a, out);
    }
    // Remainder Loop
    if (num_elm & 1)
    {
        xtfloat a1, a2, a;
        XT_LSIP(a1, (xtfloat *)inp1, 0);
        XT_LSIP(a2, (xtfloat *)inp2, 0);
        a = XT_ADD_S(a1, a2);
        XT_SSI(a, (xtfloat *)out, 0);
    }

    return 0;
}
#endif



#if HAVE_VFPU
static void internal_elm_add_broadcast_2D_f32xf32_f32(void * __restrict__ ptr_out,
                    const    void * __restrict__ ptr_inp1,
                    const    void * __restrict__ ptr_inp2,
                    bcast_args_t* args)
{
  WORD32  out_lc = args->out_lc;
  WORD32  in_lc = args->in_lc;
  xtbool sign_flag = args->sign_flag;
  
  int i, j;

  FLOAT32  * __restrict__ p_inp1 = (FLOAT32 *)ptr_inp1;
  FLOAT32  * __restrict__ p_inp2 = (FLOAT32 *)ptr_inp2;
  FLOAT32  * __restrict__  p_out =  (FLOAT32 *)ptr_out;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2;
  xtfloatx2  *__restrict__  p_c =  (xtfloatx2 *)p_out;

  int num_simd2_ops;
  int num_scalar_ops;

  if(out_lc)
  {
    num_simd2_ops = in_lc >> 1;
    num_scalar_ops = in_lc & 1;
  }
  else
  {
    num_simd2_ops = (in_lc >> 2) << 1;
    num_scalar_ops = in_lc & 3;
  }

    xtfloatx2 x1, x2, y;
    xtfloat a0, b0, c0;

  /* For computing inp2 + inp1 */
  if(sign_flag){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      p_c = (xtfloatx2 *)&p_out[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          XT_LSX2IP(x2, p_b, 2 * sizeof(FLOAT32));
          y = XT_ADD_SX2(x2, x1);
          XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32));
        }
      }
      else
      {
        ae_valign vinp1, vinp2, out_a = AE_ZALIGN64();
        vinp1 = XT_LASX2PP(p_a);
        vinp2 = XT_LASX2PP(p_b);
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LASX2IP(x1, vinp1, p_a);
          XT_LASX2IP(x2, vinp2, p_b);
          y = XT_ADD_SX2(x2, x1);
          XT_SASX2IP(y, out_a, p_c);
        }
        XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
        XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
        c0 = XT_ADD_S(b0, a0);
        XT_SSI(c0, (xtfloat *)p_c, 0);
      }
    }
  }
  /* For computing inp1 + inp2 */
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      p_c = (xtfloatx2 *)&p_out[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          XT_LSX2IP(x2, p_b, 2 * sizeof(FLOAT32));
          y = XT_ADD_SX2(x1, x2);
          XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32));
        }
      }
      else
      {
        ae_valign vinp1, vinp2, out_a = AE_ZALIGN64();
        vinp1 = XT_LASX2PP(p_a);
        vinp2 = XT_LASX2PP(p_b);

        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LASX2IP(x1, vinp1, p_a);
          XT_LASX2IP(x2, vinp2, p_b);
          y = XT_ADD_SX2(x1, x2);
          XT_SASX2IP(y, out_a, p_c);
        }
        XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
        XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
        c0 = XT_ADD_S(a0, b0);
        XT_SSI(c0, (xtfloat *)p_c, 0);
      }
    }
  }
}

static void internal_elm_add_broadcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  num_elm = args->num_elm;
  xtbool sign_flag = args->sign_flag;
  
  int i;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2;
  xtfloatx2  *__restrict__  p_c =  (xtfloatx2 *)p_out;

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  xtfloat a0_7, out;
  xtfloatx2 x1, x2, y;
  x2 = XT_LSI((xtfloat *)p_b, 0);

  /* For computing inp2 + inp1 */
  if(sign_flag){
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
        y = XT_ADD_SX2(x2, x1);
        XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valign inp1_a, out_a;
      inp1_a = XT_LASX2PP(p_a);
      out_a = AE_ZALIGN64();
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LASX2IP(x1, inp1_a, p_a);
        y = XT_ADD_SX2(x2, x1);
        XT_SASX2IP(y, out_a, p_c);
      }
      XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
      out = XT_ADD_S(x2, a0_7);
      XT_SSI(out, (xtfloat *)p_c, 0);
    }
  }
  /* For computing inp1 + inp2 */
  else
  {
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
        y = XT_ADD_SX2(x1, x2);
        XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valign inp1_a, out_a;
      inp1_a = XT_LASX2PP(p_a);
      out_a = AE_ZALIGN64();
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LASX2IP(x1, inp1_a, p_a);
        y = XT_ADD_SX2(x1, x2);
        XT_SASX2IP(y, out_a, p_c);
      }
      XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
      out = XT_ADD_S(a0_7, x2);
      XT_SSI(out, (xtfloat *)p_c, 0);
    }
  }
}
#endif
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_add_broadcast_4D_f32xf32_f32,
             (
                      FLOAT32 * p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * p_inp2,
                      const WORD32 *const p_inp2_shape
              )
           )
#else           

WORD32 xa_nn_elm_add_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);

  bcast_args_t args = {0};
  args.out_elm_size = args.inp_elm_size = 4;
  args.multiplier_sign = 1;

  return CALL_BCAST(internal_elm_add_broadcast_2D_f32xf32_f32, 
            internal_elm_add_broadcast_f32xf32_f32,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);

}
#endif
