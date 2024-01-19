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


#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_div_f32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp1,
                const FLOAT32 *p_inp2,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_div_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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
        y = XT_DIV_SX2(x1, x2);
        XT_SASX2IP(y, out_a, out);
    }
    XT_SASX2POSFP(out_a, out);

    // Remainder Loop
    if (num_elm & 1)
    {
        xtfloat a1, a2, a;
        XT_LSIP(a1, (xtfloat *)inp1, 0);
        XT_LSIP(a2, (xtfloat *)inp2, 0);
        a = XT_DIV_S(a1, a2);
        XT_SSI(a, (xtfloat *)out, 0);
    }

    return 0;
}
#endif

