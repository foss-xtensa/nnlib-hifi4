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
#ifndef XA_NNLIB_COMMON_INTERNAL_H
#define XA_NNLIB_COMMON_INTERNAL_H

#if XCHAL_HAVE_HIFI1
#define xa_nn_memcpy memcpy
#else
void *xa_nn_memcpy(void * dest,const void *src, size_t n);
#endif

#if XCHAL_HAVE_HIFI1
#if ( XCHAL_HW_VERSION >= 281090 )
#define AE_S8_0_IP_HIFI1(dr, ptr, size) AE_S8_0_IP(AE_MOVINT8X8_FROMINT16X4((dr)), (ae_int8 *)(ptr), (size))
#define AE_S8_0_I_HIFI1(dr, ptr, size)   AE_S8_0_I(AE_MOVINT8X8_FROMINT16X4((dr)), (ae_int8 *)(ptr), (size))
#define AE_S8_0_X_HIFI1(dr, ptr, size)   AE_S8_0_X(AE_MOVINT8X8_FROMINT16X4((dr)), (ae_int8 *)(ptr), (size))
#define AE_S8_0_XP_HIFI1(dr, ptr, size) AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4((dr)), (ae_int8 *)(ptr), (size))
#else
#define AE_S8_0_IP_HIFI1(dr, ptr, size) AE_S8_0_IP((dr), (WORD8 *)(ptr), (size))
#define AE_S8_0_I_HIFI1(dr, ptr, size)   AE_S8_0_I((dr), (WORD8 *)(ptr), (size))
#define AE_S8_0_X_HIFI1(dr, ptr, size)   AE_S8_0_X((dr), (WORD8 *)(ptr), (size))
#define AE_S8_0_XP_HIFI1(dr, ptr, size) AE_S8_0_XP((dr), (WORD8 *)(ptr), (size))
#endif
#endif

#endif /* XA_NNLIB_COMMON_INTERNAL_H */
