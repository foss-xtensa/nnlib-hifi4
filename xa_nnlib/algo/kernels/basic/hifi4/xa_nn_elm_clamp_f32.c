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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common_bcast_macro.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_clamp_f32xf32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                const FLOAT32 *p_min,
                const FLOAT32 *p_max,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_clamp_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               const FLOAT32 * __restrict__ p_min,
                               const FLOAT32 * __restrict__ p_max,
                               WORD32 num_elm)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_min, -1);
    XA_NNLIB_ARG_CHK_PTR(p_max, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_min, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_max, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx2 *inp = (xtfloatx2 *)p_inp;
    xtfloatx2 *min = (xtfloatx2 *)p_min;
    xtfloatx2 *max = (xtfloatx2 *)p_max;
    xtfloatx2 *out =  (xtfloatx2 *)p_out;

    xtfloatx2 x1, d_min, d_max, y;

    if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp)&7) == 0) && ((((unsigned)p_min)&7) == 0) && ((((unsigned)p_max)&7) == 0))
    {
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
            XT_LSX2IP(d_min, min, 2*sizeof(FLOAT32));
            XT_LSX2IP(d_max, max, 2*sizeof(FLOAT32));

            y = XT_MAX_SX2(x1, d_min);
            y = XT_MIN_SX2(y, d_max);

            XT_SSX2IP( y, out,  2*sizeof(FLOAT32));
        }
    }
    else
    {
        ae_valign inp_a, min_a, max_a, out_a;

        inp_a = XT_LASX2PP(inp);
        min_a = XT_LASX2PP(min);
        max_a = XT_LASX2PP(max);
        out_a = AE_ZALIGN64();
        /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LASX2IP(x1, inp_a, inp);
            XT_LASX2IP(d_min, min_a, min);
            XT_LASX2IP(d_max, max_a, max);

            y = XT_MAX_SX2(x1, d_min);
            y = XT_MIN_SX2(y, d_max);

            XT_SASX2IP(y, out_a, out);
        }
        XT_SASX2POSFP(out_a, out);
    }
    // Remainder Loop
    if (num_elm & 1)
    {
        xtfloat a1, a2, a3, a;
        XT_LSIP(a1, (xtfloat *)inp, 0);
        XT_LSIP(a2, (xtfloat *)min, 0);
        XT_LSIP(a3, (xtfloat *)max, 0);
        a = XT_MAX_S(a1, a2); 
        a = XT_MIN_S(a, a3); 
        XT_SSI(a, (xtfloat *)out, 0);
    }
    return 0;
}
#endif