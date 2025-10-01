/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_macros.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common_bcast_macro.h"

#ifndef AE_MOVAB
#ifdef AE_MOVAB1
#define AE_MOVAB AE_MOVAB1
#else
#define AE_MOVAB(x)  (x)  
#endif
#endif

#ifndef AE_MOVAB2
#define AE_MOVAB2(x) (x)
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_compare_f32xf32_f32,
             (
                WORD8 *y,
                const FLOAT32 *x1,
                const FLOAT32 *x2,
                WORD32 N,
                compare_ops_t kernel_type
              )
           )
#else
WORD32 xa_nn_elm_compare_f32xf32_f32(WORD8 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm,
                               compare_ops_t kernel_type)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0) || (kernel_type < 0) || (kernel_type > 5), -1);
    int i;
    xtfloatx2 *inp1 = (xtfloatx2 *)p_inp1;
    xtfloatx2 *inp2 = (xtfloatx2 *)p_inp2;
    UWORD8 *out = (UWORD8 *)p_out;
    xtfloatx2 x1, x2;
    xtbool check;
    
    if(kernel_type == COMPARE_GREATEREQUAL)
    {    
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = XT_OLE_SX2(x2, x1);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = XT_OLE_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == COMPARE_GREATER)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = XT_OLT_SX2(x2, x1);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = XT_OLT_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }  
    }
    else if(kernel_type == COMPARE_LESSEREQUAL)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = XT_OLE_SX2(x1, x2);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;//, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = XT_OLE_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a1, a2);
          
          check = 0;        
          if(a <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == COMPARE_LESSER)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = XT_OLT_SX2(x1, x2);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = XT_OLT_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a1, a2);
          
          check = 0;        
          if(a < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == COMPARE_EQUAL)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          //a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a1 == a2)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == COMPARE_NOTEQUAL)
    {
      ae_int32x2 ones = AE_MOVDA32(1);
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
              
              ae_int32x2 store = AE_ZERO32();
              AE_MOVF32X2(store, ones, check);
              
              *out++ = AE_MOVAD32_H(store);
              *out++ = AE_MOVAD32_L(store);
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *out++ = AE_MOVAD32_H(store);
            *out++ = AE_MOVAD32_L(store);
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }

    return 0;
}
#endif
#if HAVE_VFPU
static void internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ ptr_inp1,
                    const    void * __restrict__ ptr_inp2,
                    bcast_args_t* args)
{
  WORD32 out_lc = args->out_lc;
  WORD32 in_lc = args->in_lc;
  xtbool sign_flag = args->sign_flag;
  compare_ops_t kernel_type = args->kernel_type;
  FLOAT32 *p_inp1 = (FLOAT32*) ptr_inp1;
  FLOAT32 *p_inp2 = (FLOAT32*) ptr_inp2;
  int i, j;

  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2; 
  
  xtbool check;
  
  //xtfloatx2 float_0 = XT_MOV_SX2(AE_ZERO32());

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

    xtfloatx2 x1, x2;
    xtfloat a0, b0, c0;

  /* For computing inp2 - inp1 */   
  if(sign_flag){  
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      UWORD8 *p_c = (UWORD8 *)&p_out[i * in_lc];
      
      if(kernel_type == COMPARE_GREATEREQUAL)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLE_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLE_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_GREATER)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLT_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLT_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_LESSEREQUAL)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLE_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLE_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_LESSER)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLT_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLT_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_EQUAL)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          //c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(a0 == b0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_NOTEQUAL)
      {
        ae_int32x2 ones = AE_MOVDA32(1);
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
    }
  }
  /* For computing inp1 - inp2 */   
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      UWORD8 *p_c = (UWORD8 *)&p_out[i * in_lc];
      
      if(kernel_type == COMPARE_GREATEREQUAL)
      {    
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLE_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLE_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if (kernel_type == COMPARE_GREATER)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLT_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = XT_OLT_SX2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_LESSEREQUAL)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLE_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLE_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_LESSER)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLT_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = XT_OLT_SX2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_EQUAL)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          //c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(a0 == b0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == COMPARE_NOTEQUAL)
      {
        ae_int32x2 ones = AE_MOVDA32(1);
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
    }  
  }
}

static void internal_elm_greater_lesser_equal_broadcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32 num_elm = args->num_elm;
  xtbool sign_flag = args->sign_flag;
  compare_ops_t kernel_type = args->kernel_type;
  int i;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2; 
  
  xtbool check;
  
  UWORD8 * p_c = (UWORD8 *)p_out;
  //xtfloatx2 float_0 = XT_MOV_SX2(AE_ZERO32());

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  xtfloat a0_7, out;
  xtfloatx2 x1, x2;
  x2 = XT_LSI((xtfloat *)p_b, 0);
        
  /* For computing inp2 - inp1 */      
  if(sign_flag){
    if(kernel_type == COMPARE_GREATEREQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLE_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLE_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_GREATER)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLT_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLT_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_LESSEREQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLE_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLE_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_LESSER)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLT_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLT_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_EQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out == 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_NOTEQUAL)
    {
      ae_int32x2 ones = AE_MOVDA32(1);
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store); 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store);
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out != 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
  }
  /* For computing inp1 - inp2 */   
  else
  {
    if(kernel_type == COMPARE_GREATEREQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLE_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLE_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_GREATER)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLT_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = XT_OLT_SX2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_LESSEREQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLE_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLE_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_LESSER)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = XT_OLT_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
        xtbool2 check = XT_OLT_SX2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_EQUAL)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out == 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == COMPARE_NOTEQUAL)
    {
      ae_int32x2 ones = AE_MOVDA32(1);
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store); 
        }
      }
      else
      {
        ae_valign inp1_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = AE_EQ32(XT_AE_MOVINT32X2_FROMXTFLOATX2(x1), XT_AE_MOVINT32X2_FROMXTFLOATX2(x2));
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store);
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out != 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_compare_broadcast_4D_f32xf32_f32,
             (
                      WORD8 * p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * p_inp2,
                      const WORD32 *const p_inp2_shape,
                      compare_ops_t kernel_type
              )
           )
#else           
WORD32 xa_nn_elm_compare_broadcast_4D_f32xf32_f32(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                      compare_ops_t kernel_type)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((kernel_type < 0) || (kernel_type > 5), -1);

  bcast_args_t args = {0};
  args.inp_elm_size = 4;
  args.out_elm_size = 1;
  args.multiplier_sign = 1;
  args.kernel_type = kernel_type;

  return CALL_BCAST(internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32, 
            internal_elm_greater_lesser_equal_broadcast_f32xf32_f32,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
#endif
