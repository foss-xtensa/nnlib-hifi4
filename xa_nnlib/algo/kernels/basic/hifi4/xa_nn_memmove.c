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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

#if XCHAL_HAVE_HIFI1
WORD32 xa_nn_memmove_8_8( void *pdst,
    const void *psrc,
    WORD32 n)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pdst, -1);
  XA_NNLIB_ARG_CHK_PTR(psrc, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(pdst, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(psrc, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((n <= 0), -1);

  const WORD8 *x = (const WORD8*)psrc;
  WORD8 *y = (WORD8*)pdst;
  int i;
  ae_int16x4 d0;
  WORD8 *pOut;
  const WORD8 *pInp;
  if(y == x) //no copy needed
    return 0;

  if (y < x)
  {
    pInp = (const  WORD8 *)&x[0];
    pOut = (WORD8 *)&y[0];
    ///check for aligned part
    if( ( (((unsigned)pInp)&3)==0  ) &&  ( (((unsigned)pOut)&3)==0  )   )
    {
        for(i=0;i<n>>2;i++)
        {
            AE_L8X4S_IP(d0, pInp, 4*sizeof(WORD8));
            AE_S8X4_IP(d0, pOut, 4*sizeof(WORD8));
        }
    }
    else
    {
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();

        for(i=0;i<n>>2;i++)
        {
            AE_LA8X4S_IP(d0, alignIn, pInp);
            AE_SA8X4U_IP(d0, alignOut, (ae_int32 *)pOut);
        }
        AE_SA64POS_FP(alignOut, pOut);
    }

    i<<=2;//Reminder Loop
    for(;i<n;i++)
    {
        AE_L8S_IP(d0, pInp, sizeof(WORD8));
        AE_S8_0_IP(d0, pOut, sizeof(WORD8));
    }
  }
  else
  {
      pInp = (const  WORD8 *)&x[n-4];
      pOut = (WORD8 *)&y[n-4];

        ///check for aligned part
        if( ( (((unsigned)pInp)&3)==0  ) &&  ( (((unsigned)pOut)&3)==0  )   )
        {

            for(i=0;i<(n>>2);i++)
            {
                AE_L8X4S_IP(d0, pInp, -4*sizeof(WORD8));
                AE_S8X4_IP(d0, pOut, -4*sizeof(WORD8));
            }
            i<<=2;//Reminder Loop
            pInp = ((WORD8*)pInp + 3);
            pOut = ((WORD8*)pOut + 3);

        }
        else
        {
            pInp = (const  WORD8 *)&x[n-1];
            pOut = (WORD8 *)&y[n-1];
            for(i=0; i<n; i++)
            {
                AE_L8S_IP(d0, pInp, -1);
                AE_S8_0_IP(d0, pOut, -1);
            }
        }
        for(;i<n;i++)
        {
            *(WORD8*)pOut = *(WORD8*)pInp;
            pInp = ((WORD8*)pInp - 1);
            pOut = ((WORD8*)pOut - 1);
        }

  }
  return 0;
}
#else
WORD32 xa_nn_memmove_8_8( void *pdst,
    const void *psrc,
    WORD32 n)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pdst, -1);
  XA_NNLIB_ARG_CHK_PTR(psrc, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(pdst, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(psrc, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((n <= 0), -1);

  const WORD8 *x = (const WORD8*)psrc;
  WORD8 *y = (WORD8*)pdst;
  int i;

  if(y == x) //no copy needed
    return 0;  

  if(y < x){
    /* 2-byte aligned pointers use LA16x4 and SA16x4 instructions. Generic version uses LA24X2 and SA24X2 instructions.*/
    if(( (((unsigned)x)&1)==0  ) &&  ( (((unsigned)y)&1)==0  )   ){
        ae_int16x4 *pOut = (ae_int16x4 *)y;
        ae_int16x4 *pInp = (ae_int16x4 *)x;
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int16x4 d0;

        for(i=0;i<n>>3;i++)
        {
          AE_LA16X4_IP(d0, alignIn, pInp);
          AE_SA16X4_IP(d0, alignOut, pOut);
        }
        AE_SA64POS_FP(alignOut, pOut);
        /* remainder loop */
        for(i=n&~7; i < n; i++){
          y[i] = x[i];
        }
    } else {
        ae_int24x2 *pOut = (ae_int24x2 *)y;
        ae_int24x2 *pInp = (ae_int24x2 *)x;
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int24x2 d0;
        unsigned int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        unsigned int remainder_start = 6*Nby6;

        for(i=0;i<Nby6;i++)
        {
          AE_LA24X2_IP(d0, alignIn, pInp);
          AE_SA24X2_IP(d0, alignOut, pOut);
        }
        AE_SA64POS_FP(alignOut, pOut);
        /* remainder loop */
        for(i=remainder_start; i < n; i++){
          y[i] = x[i];
        }
    }
  } else {
    /* 2-byte aligned pointers use LA16x4 and SA16x4 instructions. Generic version uses LA24X2 and SA24X2 instructions.*/
    if(( (((unsigned)x)&1)==0  ) &&  ( (((unsigned)y)&1)==0  )   ){

        if(n&1){ /* In case of odd-valued n, take one iteration out-side of core-loop to ensure 2-byte alignment during _RIP load/stores. */
          y[n-1] = x[n-1];
          n = n-1;
        }
        ae_int16x4 *pOut = (ae_int16x4 *)&y[n-2];
        ae_int16x4 *pInp = (ae_int16x4 *)&x[n-2];
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int16x4 d0;
        unsigned int Nby8 = n>>3;
        unsigned remainder_start = n - (Nby8<<3);

        for(i=0;i<Nby8;i++){
          AE_LA16X4_RIP(d0, alignIn, pInp);
          AE_SA16X4_RIP(d0, alignOut, pOut);
        }
        AE_SA64NEG_FP(alignOut, pOut);
        /* remainder loop */
        for(i=remainder_start-1; i >= 0; i--){
          y[i] = x[i];
        }
    } else {
        ae_int24x2 *pOut = (ae_int24x2 *)&y[n-1];
        ae_int24x2 *pInp = (ae_int24x2 *)&x[n-1];
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int24x2 d0;
        unsigned int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        unsigned int remainder_start = n - 6*Nby6;

        for(i=0;i<Nby6;i++){
          AE_LA24X2_RIP(d0, alignIn, pInp);
          AE_SA24X2_RIP(d0, alignOut, pOut);
        }
        AE_SA64NEG_FP(alignOut, pOut);

        /* remainder loop */
        for(i=remainder_start-1; i >= 0; i--){
          y[i] = x[i];
        }
    }
  }

  return 0;
}
#endif
