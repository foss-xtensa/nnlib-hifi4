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

WORD32 xa_nn_memmove_16( void *pdst,
    const void *psrc,
    WORD32 n)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pdst, -1);
  XA_NNLIB_ARG_CHK_PTR(psrc, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(pdst, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(psrc, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((n <= 0), -1);

  WORD32 L = n;
  WORD32 i;
  const WORD16 *x = (const WORD16*)psrc;
  WORD16 *y = (WORD16*)pdst;
  int loop_count_by_4 = L >> 2;
  ae_int16x4 d_inp;
  ae_valign align_inp_reg, align_out_reg;
  ae_int16x4 *pout;
  const ae_int16x4 *pinp;
  const ae_int16 *pinp_tmp;
  ae_int16 *pout_tmp;

  if(y == x) //no copy needed
    return 0;  

  if (y < x)
  {
    int rem_start = 0;
    pinp = (const  ae_int16x4*)&x[0];
    pout = (ae_int16x4*)&y[0];

    if ( L >= 4 )
    {
      align_inp_reg = AE_LA64_PP(pinp);
      align_out_reg = AE_ZALIGN64();
      for (i = 0; i < loop_count_by_4; i++)
      {
        AE_LA16X4_IP(d_inp, align_inp_reg, pinp);
        AE_SA16X4_IP(d_inp, align_out_reg, pout);
      }
      AE_SA64POS_FP(align_out_reg, pout); /* finalize the stream */
      rem_start = loop_count_by_4 << 2;
    }

    pinp_tmp = (const ae_int16*)pinp;
    pout_tmp = (ae_int16*)pout;
    for (i = rem_start; i < L; i++)
    {
      AE_L16_IP(d_inp, pinp_tmp, 2 );
      AE_S16_0_IP(d_inp, pout_tmp, 2 );
    }
  }
  else
  {
    int rem_start = 0;
    pinp  = (const ae_int16x4*)&x[L-1];
    pout = (ae_int16x4*)&y[L-1];

    if ( L >= 4 )
    {
      align_inp_reg = AE_LA64_PP(pinp);
      align_out_reg = AE_ZALIGN64();
      for (i = 0; i < loop_count_by_4; i++)
      {
        AE_LA16X4_RIP(d_inp, align_inp_reg, pinp);
        AE_SA16X4_RIP(d_inp, align_out_reg, pout);
      }
      AE_SA64NEG_FP(align_out_reg, pout); /* finalize the stream */
      rem_start = loop_count_by_4 << 2;
    }

    pinp_tmp = (const ae_int16*)pinp;
    pout_tmp = (ae_int16*)pout;
    for (i = rem_start; i < L; i++)
    {
      AE_L16_IP(d_inp, pinp_tmp, -2 );
      AE_S16_0_IP(d_inp, pout_tmp, -2 );
    }
  }

  return 0;  
}
