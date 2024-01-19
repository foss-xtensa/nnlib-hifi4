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
#include "xtensa/xtensa-versions.h"

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= RI9_HWVERSION )
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
  ae_int8x8 d0;
  ae_int16x4 d1;
  WORD8 *pOut;
  const WORD8 *pInp;
  if(y == x) //no copy needed
    return 0;

  if (y < x)
  {
    pInp = (const  WORD8 *)&x[0];
    pOut = (WORD8 *)&y[0];
    ae_valign alignIn, alignOut;
    alignIn = AE_LA64_PP(pInp);
    alignOut = AE_ZALIGN64();

    for(i=0;i<n>>3;i++)
    {
        AE_LA8X8_IP(d0, alignIn, (const ae_int8x8 *)pInp);
        AE_SA8X8_IP(d0, alignOut, ( ae_int8x8 *)pOut);
    }
    //Reminder Loop
    {
        AE_LAV8X8_XP(d0, alignIn, (const ae_int8x8 *)pInp, (n&7));
        AE_SAV8X8_XP(d0, alignOut, (ae_int8x8 *)pOut, (n&7));
    }
    AE_SA64POS_FP(alignOut, pOut);
  }
  else
  {
    pOut = ( WORD8 *)&y[n-2];
    pInp = (WORD8 *)&x[n-2];

    ///check for aligned part
    if(( (((unsigned)pOut)&1)==0  ) &&  ( (((unsigned)pInp)&1)==0  ) &&(n>7))
    {
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        for(i=0;i<(n>>3);i++)
        {
            AE_LA16X4_RIP(d1, alignIn, (ae_int16x4 *)pInp);
            AE_SA16X4_RIP(d1, alignOut, (ae_int16x4 *)pOut);
        }
        AE_SA64NEG_FP(alignOut, pOut);
        //Reminder Loop
        pInp = ((WORD8*)pInp + 1);
        pOut = ((WORD8*)pOut + 1);
        for(i=0 ;i< (n&7);i++)
        {
            *(WORD8*)pOut = *(WORD8*)pInp;
            pInp = ((WORD8*)pInp - 1);
            pOut = ((WORD8*)pOut - 1);
        }
    }
    else
    {
    pOut = ( WORD8 *)&y[n-1];
    pInp = (WORD8 *)&x[n-1];
        for(i=0; i<n; i++)
        {
            AE_L8_IP(d0, (const ae_int8 *)pInp, -1);
            AE_S8_0_IP(d0, ( ae_int8 *)pOut, -1);
        }
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
        AE_S8_0_IP_HIFI1(d0, pOut, sizeof(WORD8));
    }
  }
  else
  {
    pOut = ( WORD8 *)&y[n-2];
    pInp = (WORD8 *)&x[n-2];

    ///check for aligned part
    if(( (((unsigned)pOut)&1)==0  ) &&  ( (((unsigned)pInp)&1)==0  ) &&(n>7))
    {
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        for(i=0;i<(n>>3);i++)
        {
            AE_LA16X4_RIP(d0, alignIn, (ae_int16x4 *)pInp);
            AE_SA16X4_RIP(d0, alignOut, (ae_int16x4 *)pOut);
        }
        AE_SA64NEG_FP(alignOut, pOut);
        //Reminder Loop
        pInp = ((WORD8*)pInp + 1);
        pOut = ((WORD8*)pOut + 1);
        for(i=0 ;i< (n&7);i++)
        {
            *(WORD8*)pOut = *(WORD8*)pInp;
            pInp = ((WORD8*)pInp - 1);
            pOut = ((WORD8*)pOut - 1);
        }
    }
    else
    {
    pOut = ( WORD8 *)&y[n-1];
    pInp = (WORD8 *)&x[n-1];
        for(i=0; i<n; i++)
        {
            AE_L8S_IP(d0, (WORD8 *)pInp, -1);
            AE_S8_0_IP_HIFI1(d0, (WORD8 *)pOut, -1);
        }
    }
  }
  return 0;
}
#endif
#else
#include <string.h>
void *xa_nn_memcpy(void * dest1,const void *src1, size_t n1)
{
  char *dest = (char *)dest1;
  char *src = (char *)src1;
  int n = (int)n1;
  ae_int16x4 * __restrict d_align_addr, * __restrict s_align_addr;
  int i;
  void *orig_dest = dest;

  if (n < 32) {
    return memcpy(dest, src, n);
  }

  if ( !(((int) dest) %8) && !(((int) src) %8)) { // 64-bit aligned
    s_align_addr = (ae_int16x4 *) src;
    d_align_addr = (ae_int16x4 *) dest;
    for (i=0; i<n>>3; i++) {
        d_align_addr[i] = s_align_addr[i];
    }

    for (i=(n&~7); i<n; i++) {
      dest[i] = src[i];
    }
    return orig_dest;
  }

  if ( (((int) dest) %2) || (((int) src) %2)) { // 16-bit aligned
    if ( (((int) dest) %2) && (((int) src) %2)) { // 16-bit aligned
      *dest++ = *src++;
       n--;
    } else {
      #if 0
      return memcpy(dest, src, n);
      #else
        ae_int24x2 *pOut = (ae_int24x2 *)dest;
        ae_int24x2 *pInp = (ae_int24x2 *)src;
        ae_valign alignIn, alignOut;
        alignIn = AE_LA64_PP(pInp);
        alignOut = AE_ZALIGN64();
        ae_int24x2 d0;
        int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        int remainder_start = 6*Nby6;

        for(i=0;i<Nby6;i++)
        {
          AE_LA24X2_IP(d0, alignIn, pInp);
          AE_SA24X2_IP(d0, alignOut, pOut);
        }
        AE_SA64POS_FP(alignOut, pOut);
        /* remainder loop */
        for(i=remainder_start; i < n; i++){
          dest[i] = src[i];
      }
      return orig_dest;
      #endif
    }
  }
  int n2 = n/2;
  ae_valign d_align = AE_ZALIGN64();
  d_align_addr = (ae_int16x4 *) dest;
  s_align_addr = (ae_int16x4 *) src;
  ae_valign s_align = AE_LA64_PP(s_align_addr);
  ae_int16x4 t,t2;
  for (i=0; i<n2>>3; i++) {
      AE_LA16X4_IP(t, s_align, s_align_addr);
      AE_LA16X4_IP(t2, s_align, s_align_addr);
      AE_SA16X4_IP(t, d_align, d_align_addr);
      AE_SA16X4_IP(t2, d_align, d_align_addr);
  }
  AE_SA64POS_FP(d_align, d_align_addr);
  ae_int16 *s_src = (ae_int16 *) src;
  ae_int16 *s_dest = (ae_int16 *) dest;
  for (i=8*i; i<n2; i++) {
    s_dest[i] = s_src[i];
  }
  if (n % 2) {
    dest[n-1] = src[n-1];
  }
  return orig_dest;
} /* xa_nn_memcpy */

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
        int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        int remainder_start = 6*Nby6;

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
        int Nby8 = n>>3;
        int remainder_start = n - (Nby8<<3);

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
        int Nby6 =  AE_MOVAD32_H(AE_MOVINT32X2_FROMINT64(AE_MUL32_LL(n, 0x2AAAAAAB)));
        int remainder_start = n - 6*Nby6;

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
