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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"


#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_l2_norm_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_l2_norm_f32(FLOAT32 * __restrict__ p_out,
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

    int i;
    xtfloatx2 *pt_inp = (xtfloatx2 *)p_inp;
    xtfloatx2 *pt_out =  (xtfloatx2 *)p_out;
    ae_valign inp_a, out_a;
    xtfloatx2 d_inpx2, d_outx2, enegx2, eneg_sqrtx2;
    xtfloat d_inp, d_out, eneg, eneg_sqrt;

    /* Calculate energy (squared sum) */
    inp_a = XT_LASX2PP(pt_inp);
    enegx2 = XT_CONST_S(0);
    for(i=0;i < num_elm>>1;i++)
    {
        XT_LASX2IP(d_inpx2, inp_a, pt_inp);
        XT_MADD_SX2(enegx2, d_inpx2, d_inpx2);
    }
    enegx2 = XT_ADD_SX2(enegx2, XT_SEL32_LH_SX2(enegx2, enegx2));
    eneg = XT_LOW_S(enegx2);
    // Remainder Loop
    if (num_elm & 1)
    {
        d_inp = *(xtfloat *)pt_inp;
        XT_MADD_S(eneg, d_inp, d_inp);
    }
    eneg_sqrt = XT_SQRT_S(eneg);
    eneg_sqrtx2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVDA32(XT_RFR(eneg_sqrt)));

    pt_inp = (xtfloatx2 *)p_inp;
    inp_a = XT_LASX2PP(pt_inp);
    out_a = AE_ZALIGN64();
    for(i=0;i < num_elm>>1;i++)
    {
        XT_LASX2IP(d_inpx2, inp_a, pt_inp);
        d_outx2 = XT_DIV_SX2(d_inpx2, eneg_sqrtx2);
        XT_SASX2IP(d_outx2, out_a, pt_out);
    }
    XT_SASX2POSFP(out_a, pt_out);
    // Remainder Loop
    if (num_elm & 1)
    {
        d_inp = *(xtfloat *)pt_inp;
        d_out = XT_DIV_S(d_inp, eneg_sqrt);
        *(xtfloat *)pt_out = d_out;
    }

    return 0;
}
#endif
