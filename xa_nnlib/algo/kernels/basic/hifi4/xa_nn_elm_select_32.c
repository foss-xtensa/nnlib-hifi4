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
#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"

#ifndef AE_MOVBA1X2
#ifdef AE_MOVBA2
#define AE_MOVBA1X2(src0, src1) AE_MOVBA2(((src0)<<1)|(src1))
#else
#define AE_MOVBA1X2(src0, src1) AE_EQ32(AE_MOVDA32X2(src0, src1), AE_MOVDA32X2(1, 1)) 
#endif
#endif

#ifndef AE_MOVBA
#ifdef AE_MOVBA1
#define AE_MOVBA AE_MOVBA1
#else
#define AE_MOVBA(inp) inp
#endif
#endif

WORD32 xa_nn_elm_select_32x32_32(WORD32 * __restrict__ p_out,
                               const WORD32 * __restrict__ p_inp1,
                               const WORD32 * __restrict__ p_inp2,
                               const unsigned char *__restrict__ p_condition,
                               WORD32 num_elm)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    ae_int32x2 *inp1 = (ae_int32x2 *)p_inp1;
    ae_int32x2 *inp2 = (ae_int32x2 *)p_inp2;
    ae_int32x2 *out =  (ae_int32x2 *)p_out;
    const unsigned char *condition = p_condition;
    ae_int32x2 x1, x2, y;
    unsigned char con1, con2;
    xtbool2 con;

    if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
    {
        for(i=0;i < num_elm>>1;i++)
        {
            AE_L32X2_IP(x1, inp1, 2*sizeof(WORD32));
            AE_L32X2_IP(x2, inp2, 2*sizeof(WORD32));
            con1 = XT_L8UI(condition, 0);
            condition++;
            con2 = XT_L8UI(condition, 0);
            condition++;
            con = AE_MOVBA1X2(con1, con2);
            AE_MOVT32X2 (y, x1, con);
            AE_MOVF32X2 (y, x2, con);
            AE_S32X2_IP( y, out,  2*sizeof(WORD32));
        }
    }
    else
    {
        ae_valign inp1_a, inp2_a, out_a;

        inp1_a = AE_LA64_PP(inp1);
        inp2_a = AE_LA64_PP(inp2);
        out_a = AE_ZALIGN64();
        /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
        for(i=0;i < num_elm>>1;i++)
        {
            AE_LA32X2_IP(x1, inp1_a, inp1);
            AE_LA32X2_IP(x2, inp2_a, inp2);
            con1 = XT_L8UI(condition, 0);
            condition++;
            con2 = XT_L8UI(condition, 0);
            condition++;
            con = AE_MOVBA1X2(con1, con2);
            AE_MOVT32X2 (y, x1, con);
            AE_MOVF32X2 (y, x2, con);
            AE_SA32X2_IP(y, out_a, out);
        }
        AE_SA64POS_FP(out_a, out);
    }
    // Remainder Loop
    if (num_elm & 1)
    {
        ae_int32 a1, a2, a;
        con1 = XT_L8UI(condition, 0);
        xtbool s = AE_MOVBA(con1);
        a1 = AE_L_32((ae_int32 *)inp1, 0);
        a2 = AE_L_32((ae_int32 *)inp2, 0);
        AE_MOVT_32(a, a1, s);
        AE_MOVF_32(a, a2, s);
        AE_S32_L_I(a, (ae_int32 *)out, 0);
    }
  return 0;
}

static void internal_elm_select_broadcast_32x32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp1,
                    const    WORD32 * __restrict__ p_inp2,
                    const    unsigned char * __restrict__ p_condition,
                             WORD32  num_elm,
                             xtbool  sign_flag)
{
  int i;
  ae_int32x2  * __restrict__ p_a = (ae_int32x2 *)p_inp1;
  ae_int32x2  * __restrict__ p_b = (ae_int32x2 *)p_inp2; 
  ae_int32x2  *__restrict__  p_c =  (ae_int32x2 *)p_out;
  const unsigned char *condition = p_condition;

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  ae_int32 a0_7, out;
  ae_int32x2 x1, x2, y;
  x2 = AE_L32_I((ae_int32 *)p_b, 0);

  unsigned char con1, con2;
  xtbool2 con;

  /* For out = condition ? inp2 :inp1 */
  if(sign_flag){
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        AE_L32X2_IP(x1, p_a, 2 * sizeof(WORD32));
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x2, con);
        AE_MOVF32X2 (y, x1, con);
        AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
      }
    }
    else
    {
      ae_valign inp1_a, out_a;
      inp1_a = AE_LA64_PP(p_a);
      out_a = AE_ZALIGN64();      
      for(i=0; i<num_simd2_ops; i++)
      {
        AE_LA32X2_IP(x1, inp1_a, p_a);
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x2, con);
        AE_MOVF32X2 (y, x1, con);
        AE_SA32X2_IP(y, out_a, p_c);
      }
      AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);   
    }  
    if(num_scalar_ops !=0)
    {
      a0_7 = AE_L_32((ae_int32 *)p_a, 0);
      con1 = XT_L8UI(condition, 0);
      xtbool s = AE_MOVBA(con1);
      AE_MOVT_32(out, x2, s);
      AE_MOVF_32(out, a0_7, s);  
      AE_S32_L_I(out, (ae_int32 *)p_c, 0);
    }
  }
  /* For out = condition ? inp1 :inp2 */
  else
  {
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        AE_L32X2_IP(x1, p_a, 2 * sizeof(WORD32));
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x1, con);
        AE_MOVF32X2 (y, x2, con);
        AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
      }
    }
    else
    {
      ae_valign inp1_a, out_a;
      inp1_a = AE_LA64_PP(p_a);
      out_a = AE_ZALIGN64();       
      for(i=0; i<num_simd2_ops; i++)
      {
        AE_LA32X2_IP(x1, inp1_a, p_a);
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x1, con);
        AE_MOVF32X2 (y, x2, con);
        AE_SA32X2_IP(y, out_a, p_c);
      }
      AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      a0_7 = AE_L_32((ae_int32 *)p_a, 0);
      con1 = XT_L8UI(condition, 0);
      xtbool s = AE_MOVBA(con1);
      AE_MOVT_32(out, a0_7, s);
      AE_MOVF_32(out, x2, s);    
      AE_S32_L_I(out, (ae_int32 *)p_c, 0);
    }    
  }
}

static void internal_elm_select_broadcast_both_32x32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp1,
                    const    WORD32 * __restrict__ p_inp2,
                    const    unsigned char * __restrict__ p_condition,
                             WORD32  num_elm)
{
  int i;
  ae_int32x2  * __restrict__ p_a = (ae_int32x2 *)p_inp1;
  ae_int32x2  * __restrict__ p_b = (ae_int32x2 *)p_inp2; 
  ae_int32x2  *__restrict__  p_c =  (ae_int32x2 *)p_out;
  const unsigned char *condition = p_condition;

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  ae_int32 out;//a0_7,
  ae_int32x2 x1, x2, y;
  x2 = AE_L32_I((ae_int32 *)p_b, 0);
  x1 = AE_L32_I((ae_int32 *)p_a, 0);

  unsigned char con1, con2;
  xtbool2 con;

    if((((unsigned)p_c)&7) == 0)
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x1, con);
        AE_MOVF32X2 (y, x2, con);
        AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
      }
    }
    else
    {
      ae_valign out_a;
      out_a = AE_ZALIGN64();       
      for(i=0; i<num_simd2_ops; i++)
      {
        con1 = XT_L8UI(condition, 0);
        condition++;
        con2 = XT_L8UI(condition, 0);
        condition++;
        con = AE_MOVBA1X2(con1, con2);
        AE_MOVT32X2 (y, x1, con);
        AE_MOVF32X2 (y, x2, con);
        AE_SA32X2_IP(y, out_a, p_c);
      }
      AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      con1 = XT_L8UI(condition, 0);
      xtbool s = AE_MOVBA(con1);
      AE_MOVT_32(out, x1, s);
      AE_MOVF_32(out, x2, s);    
      AE_S32_L_I(out, (ae_int32 *)p_c, 0);
    }
}

static void internal_elm_select_broadcast_2D_32x32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp1,
                    const    WORD32 * __restrict__ p_inp2,
                    const    unsigned char * __restrict__ p_condition,
                             WORD32  out_lc,
                             WORD32  in_lc,
                             xtbool  sign_flag)
{
  int i, j;

  ae_int32x2  * __restrict__ p_a = (ae_int32x2 *)p_inp1;
  ae_int32x2  * __restrict__ p_b = (ae_int32x2 *)p_inp2; 
  ae_int32x2  *__restrict__  p_c =  (ae_int32x2 *)p_out;
  const unsigned char *condition = p_condition;
  
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

    ae_int32x2 x1, x2, y;
    ae_int32 a0, b0, c0;
    unsigned char con1, con2;
    xtbool2 con;
  /* For out = condition ? inp2 :inp1 */   
  if(sign_flag){  
    for(i = 0; i < out_lc; i++)
    {
      p_a = (ae_int32x2 *)&p_inp1[i * in_lc];
      p_b = (ae_int32x2 *)p_inp2;
      p_c = (ae_int32x2 *)&p_out[i * in_lc];
      condition = &p_condition[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_L32X2_IP(x1, p_a, 2 * sizeof(WORD32));
          AE_L32X2_IP(x2, p_b, 2 * sizeof(WORD32));
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x2, con);
          AE_MOVF32X2 (y, x1, con);
          AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
        }
      }
      else
      {
        ae_valign vinp1, vinp2, out_a = AE_ZALIGN64();
        vinp1 = AE_LA64_PP(p_a);
        vinp2 = AE_LA64_PP(p_b);
        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_LA32X2_IP(x1, vinp1, p_a);
          AE_LA32X2_IP(x2, vinp2, p_b);
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x2, con);
          AE_MOVF32X2 (y, x1, con);
          AE_SA32X2_IP(y, out_a, p_c); 
        }
        AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        a0 = AE_L_32((ae_int32 *)p_a, 0);
        b0 = AE_L_32((ae_int32 *)p_b, 0);
        con1 = XT_L8UI(condition, 0);
        xtbool s = AE_MOVBA(con1);
        AE_MOVT_32(c0, b0, s);
        AE_MOVF_32(c0, a0, s);   
        AE_S32_L_I(c0, (ae_int32 *)p_c, 0);
      }      
    }
  }
  /* For out = condition ? inp1 :inp2 */
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (ae_int32x2 *)&p_inp1[i * in_lc];
      p_b = (ae_int32x2 *)p_inp2;
      p_c = (ae_int32x2 *)&p_out[i * in_lc];
      condition = &p_condition[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_L32X2_IP(x1, p_a, 2 * sizeof(WORD32));
          AE_L32X2_IP(x2, p_b, 2 * sizeof(WORD32));
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x1, con);
          AE_MOVF32X2 (y, x2, con);
          AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
        }
      }
      else
      {
        ae_valign vinp1, vinp2, out_a = AE_ZALIGN64();
        vinp1 = AE_LA64_PP(p_a);
        vinp2 = AE_LA64_PP(p_b);

        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_LA32X2_IP(x1, vinp1, p_a);
          AE_LA32X2_IP(x2, vinp2, p_b);
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x1, con);
          AE_MOVF32X2 (y, x2, con);
          AE_SA32X2_IP(y, out_a, p_c); 
        }
        AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        a0 = AE_L_32((ae_int32 *)p_a, 0);
        b0 = AE_L_32((ae_int32 *)p_b, 0);
        con1 = XT_L8UI(condition, 0);
        xtbool s = AE_MOVBA(con1);
        AE_MOVT_32(c0, a0, s);
        AE_MOVF_32(c0, b0, s);   
        AE_S32_L_I(c0, (ae_int32 *)p_c, 0);
      }      
    }  
  }
}

static void internal_elm_select_broadcast_both_2D_32x32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp1,
                    const    WORD32 * __restrict__ p_inp2,
                    const    unsigned char * __restrict__ p_condition,
                             WORD32  out_lc,
                             WORD32  in_lc)
{
  int i, j;

  ae_int32x2  * __restrict__ p_a = (ae_int32x2 *)p_inp1;
  ae_int32x2  * __restrict__ p_b = (ae_int32x2 *)p_inp2; 
  ae_int32x2  *__restrict__  p_c =  (ae_int32x2 *)p_out;
  const unsigned char *condition = p_condition;
  
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

    ae_int32x2 x1, x2, y;
    ae_int32 a0, b0, c0;
    unsigned char con1, con2;
    xtbool2 con;

    for(i = 0; i < out_lc; i++)
    {
      p_a = (ae_int32x2 *)p_inp1;
      p_b = (ae_int32x2 *)p_inp2;
      p_c = (ae_int32x2 *)&p_out[i * in_lc];
      condition = &p_condition[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_L32X2_IP(x1, p_a, 2 * sizeof(WORD32));
          AE_L32X2_IP(x2, p_b, 2 * sizeof(WORD32));
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x1, con);
          AE_MOVF32X2 (y, x2, con);
          AE_S32X2_IP(y, p_c, 2 * sizeof(WORD32)); 
        }
      }
      else
      {
        ae_valign vinp1, vinp2, out_a = AE_ZALIGN64();
        vinp1 = AE_LA64_PP(p_a);
        vinp2 = AE_LA64_PP(p_b);

        for(j = 0; j < num_simd2_ops; j++)
        {
          AE_LA32X2_IP(x1, vinp1, p_a);
          AE_LA32X2_IP(x2, vinp2, p_b);
          con1 = XT_L8UI(condition, 0);
          condition++;
          con2 = XT_L8UI(condition, 0);
          condition++;
          con = AE_MOVBA1X2(con1, con2);
          AE_MOVT32X2 (y, x1, con);
          AE_MOVF32X2 (y, x2, con);
          AE_SA32X2_IP(y, out_a, p_c); 
        }
        AE_SA64POS_FP(out_a, (ae_int32x2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        a0 = AE_L_32((ae_int32 *)p_a, 0);
        b0 = AE_L_32((ae_int32 *)p_b, 0);
        con1 = XT_L8UI(condition, 0);
        xtbool s = AE_MOVBA(con1);
        AE_MOVT_32(c0, a0, s);
        AE_MOVF_32(c0, b0, s);   
        AE_S32_L_I(c0, (ae_int32 *)p_c, 0);
      }      
    }  
}

WORD32 xa_nn_elm_select_broadcast_4D_32x32_32(WORD32 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const WORD32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const WORD32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                      const unsigned char *__restrict__ p_condition,
                      const WORD32 *const p_condition_shape
                      )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_condition, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_condition_shape, -1);

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD32), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_condition, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_condition_shape, sizeof(WORD32), -1);
  /* Check shapes */
  int i;
  xtbool sign_flag;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i]) && ((p_inp1_shape[i] != 1) && (p_inp2_shape[i] != 1)))
    {
      return -1;
    }
  }
  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast_1 = 0;
  int need_broadcast_2 = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
      if(p_inp1_shape[i] == 1)
      {
          inp1_strides[i] = 0;
          need_broadcast_1 = 1;
      }
      else
      {
          inp1_const &= 0;
      }
      if(p_inp2_shape[i] == 1)
      {
          inp2_strides[i] = 0;
          need_broadcast_2 = 1;
      }
      else
      {
          inp2_const &= 0;
      }
  }

  int itr0, itr1, itr2;
  WORD32 *p_out_tmp = p_out;
  const unsigned char *__restrict p_condition_temp = p_condition;
  const WORD32 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD32 *__restrict__ p_inp2_tmp = p_inp2;

  if(!(need_broadcast_1 || need_broadcast_2))
  {
    sign_flag = 0;
    internal_elm_select_broadcast_2D_32x32_32(
                p_out,
                p_inp1,
                p_inp2,
                p_condition,
                1,
                p_out_shape[0] * inp1_strides[0],
                sign_flag);
  }
  else if((inp1_strides[3] == 1)&& (inp2_strides[3] == 1))
  {
    WORD32 in_lc, out_lc;
    sign_flag = 0;
    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if((inp1_strides[2] == 0) && (inp2_strides[2] == 0) )
    {
        in_lc = p_out_shape[3];
        out_lc = p_out_shape[2];
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const WORD32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
          const WORD32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            internal_elm_select_broadcast_both_2D_32x32_32(
                p_out_tmp,
                p_inp1_tmp0,
                p_inp2_tmp0,
                p_condition_temp,
                out_lc,
                in_lc);
            p_out_tmp += in_lc * out_lc;
            p_inp1_tmp0 += inp1_strides[1];
            p_inp2_tmp0 += inp2_strides[1];
            p_condition_temp += in_lc * out_lc;
          }
          p_inp1_tmp += inp1_strides[0];
          p_inp2_tmp += inp2_strides[0];
        }
    }
    else
    {
        if(inp1_strides[2] == 0)
        {
          const WORD32 *tmp;
          tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
          sign_flag = 1;
          int tmp_strides[2];
          tmp_strides[0] = inp1_strides[0];
          tmp_strides[1] = inp1_strides[1];

          inp1_strides[0] = inp2_strides[0];
          inp1_strides[1] = inp2_strides[1];

          inp2_strides[0] = tmp_strides[0];
          inp2_strides[1] = tmp_strides[1];
          in_lc = p_out_shape[3];
          out_lc = p_out_shape[2];
        }
        else if(inp2_strides[2] == 0)
        {
          in_lc = p_out_shape[3];
          out_lc = p_out_shape[2];
        }

        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const WORD32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
          const WORD32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            internal_elm_select_broadcast_2D_32x32_32(
                p_out_tmp,
                p_inp1_tmp0,
                p_inp2_tmp0,
                p_condition_temp,
                out_lc,
                in_lc,
                sign_flag);
            p_out_tmp += in_lc * out_lc;
            p_inp1_tmp0 += inp1_strides[1];
            p_inp2_tmp0 += inp2_strides[1];
            p_condition_temp += in_lc * out_lc;
          }

          p_inp1_tmp += inp1_strides[0];
          p_inp2_tmp += inp2_strides[0];
        }
    }
  }
  else if((inp1_const == 1) && (inp2_const == 1))
  {
    internal_elm_select_broadcast_both_32x32_32(
        p_out_tmp,
        p_inp1_tmp,
        p_inp2_tmp,
        p_condition_temp,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else if((inp1_const && (!need_broadcast_2))||(inp2_const && (!need_broadcast_1)))
  {
        sign_flag = 0;
        if(inp1_const == 1)
        {
          sign_flag = 1;
          const WORD32 *tmp;
          tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
        }
        internal_elm_select_broadcast_32x32_32(
            p_out_tmp,
            p_inp1_tmp,
            p_inp2_tmp,
            p_condition_temp,
            p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3],
            sign_flag);
  }
  else
  {
    sign_flag = 0;
    if((inp1_strides[3] == 0) && (inp2_strides[3] == 0))
    {
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const WORD32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
          const WORD32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            const WORD32 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
            const WORD32 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
            for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
            {
              {
                internal_elm_select_broadcast_both_32x32_32(
                    p_out_tmp,
                    p_inp1_tmp1,
                    p_inp2_tmp1,
                    p_condition_temp,
                    p_out_shape[3]);
              }
              p_out_tmp += p_out_shape[3];
              p_inp1_tmp1 += inp1_strides[2];
              p_inp2_tmp1 += inp2_strides[2];
              p_condition_temp += p_out_shape[3];
            }
            p_inp1_tmp0 += inp1_strides[1];
            p_inp2_tmp0 += inp2_strides[1];
          }
          p_inp1_tmp += inp1_strides[0];
          p_inp2_tmp += inp2_strides[0];
        }
    }
    else
    {
        if(inp1_strides[3] == 0)
        {
          const WORD32 *tmp;
          tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
          sign_flag = 1;
          int tmp_strides[3];
          tmp_strides[0] = inp1_strides[0];
          tmp_strides[1] = inp1_strides[1];
          tmp_strides[2] = inp1_strides[2];

          inp1_strides[0] = inp2_strides[0];
          inp1_strides[1] = inp2_strides[1];
          inp1_strides[2] = inp2_strides[2];

          inp2_strides[0] = tmp_strides[0];
          inp2_strides[1] = tmp_strides[1];
          inp2_strides[2] = tmp_strides[2];
        }
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const WORD32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
          const WORD32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            const WORD32 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
            const WORD32 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
            for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
            {
              { 
                internal_elm_select_broadcast_32x32_32(
                    p_out_tmp,
                    p_inp1_tmp1,
                    p_inp2_tmp1,
                    p_condition_temp,
                    p_out_shape[3], 
                    sign_flag);
              }
              p_out_tmp += p_out_shape[3];
              p_inp1_tmp1 += inp1_strides[2];
              p_inp2_tmp1 += inp2_strides[2];
              p_condition_temp += p_out_shape[3];
            }
            p_inp1_tmp0 += inp1_strides[1];
            p_inp2_tmp0 += inp2_strides[1];
          }
          p_inp1_tmp += inp1_strides[0];
          p_inp2_tmp += inp2_strides[0];
        }
    }
  }
  return 0;
}
