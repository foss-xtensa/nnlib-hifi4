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
/*
 * xa_nn_elm_minmax_asym8s.c
 */

#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

// out = (in1 > in2 ) ? in1 : in2 ;
WORD32 xa_nn_elm_max_8x8_8( WORD8* __restrict__ p_out,
                      const WORD8* __restrict__ p_in1,
                      const WORD8* __restrict__ p_in2,
                            WORD32              num_element)
{
    return -1;
}


// out = (in1 < in2 ) ? in1 : in2 ;
WORD32 xa_nn_elm_min_8x8_8( WORD8* __restrict__ p_out,
                      const WORD8* __restrict__ p_in1,
                      const WORD8* __restrict__ p_in2,
                            WORD32              num_element)
{
    return -1;
}


