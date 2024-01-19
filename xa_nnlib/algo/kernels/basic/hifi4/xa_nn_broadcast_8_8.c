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
/*
 * xa_nn_broadcast_8_8.c
 */

#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

#include<string.h>
#include<stdbool.h>

/*
 * This file is sourced from ../hifi5/xa_nn_broadcast_8_8.c
 */

#define NUMDIMS_MAX 8

typedef struct bcast_expansion_struct_{
    size_t load_num_elem;
    int    replicate_loadedElm_times;
    int    repeat_operation;
} bcast_expansion_rule ;

static const void* broadcast_node(bcast_expansion_rule *step, unsigned int step_id, void *dest, const void *src );

WORD32 xa_nn_broadcast_8_8( WORD8* __restrict__ p_out,      /* pointer to write broadcasted output data to */
        const int *const out_shape,         /* output shape resulting after broadcast */

        const  WORD8* __restrict__ p_in,    /* pointer to unextended input data */
        const int * const in_shape,         /* input shape */
        int num_dims)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(out_shape, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in, -1);
    XA_NNLIB_ARG_CHK_PTR(in_shape, -1);

    /* IO shape pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(in_shape, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(out_shape, sizeof(WORD32), -1);

    /* Check if number of dims is valid */
    XA_NNLIB_ARG_CHK_COND(num_dims<=0 || num_dims>8, -1);

    int i = 0;

    /* Check for valid IO shapes */
    for(i=0; i<num_dims; i++){
        XA_NNLIB_CHK_COND(in_shape[i]<=0, -1);
        XA_NNLIB_CHK_COND(out_shape[i]<=0, -1);
    }

    /* Check if input shape can be broadcasted to requested output shape */
    for(i=0; i<num_dims; i++){
        if(in_shape[i] != out_shape[i]){
            /* in_shape is either same as out_shape or 1 */
            XA_NNLIB_CHK_COND( in_shape[i] != 1, -1);
        }
    }

    /* bcast_expansion_steps contains a sequence to steps execute for a broadcast op */
    bcast_expansion_rule bcast_expansion_steps[NUMDIMS_MAX] = {{0}};

    int k=0;
    int dim=0;
    const void *res=0;

    int num_elem_load = 1;
    int num_copy_times = 1;
    int num_repeat = 1;

    dim = num_dims-1;
    while(dim>=0){

        /* Find the sub-matrix size */
        while(in_shape[dim] != 1 && dim>=0){
            num_elem_load *= out_shape[dim];
            dim--;
        }

        /* Find the number of times this sub-matrix needs to be copied */
        num_copy_times = 1;
        while(in_shape[dim] == 1 && dim>=0){
            num_copy_times *= out_shape[dim];
            dim--;
        }

        /* Find the number of times the above copy needs to be repeated */
        num_repeat = 1;
        while(in_shape[dim] != 1 && dim>=0){
            num_repeat *= out_shape[dim];
            dim--;
        }

        bcast_expansion_steps[k].load_num_elem  = num_elem_load;
        bcast_expansion_steps[k].replicate_loadedElm_times = num_copy_times;
        bcast_expansion_steps[k].repeat_operation = num_repeat;
        k++;

        num_elem_load = num_elem_load*num_copy_times*num_repeat;
    }

    res = broadcast_node(bcast_expansion_steps, num_dims-1,
            p_out, p_in);
    (void)res; /* Unused return value */

    return 0;
}

static const void* broadcast_node(bcast_expansion_rule *steps, unsigned int step_id,
        void *dst, const void *src) {
    int step_itr=0, rep_itr=0;
    int i=0, j=0, k=0;
    bcast_expansion_rule *step = NULL;

    // ignore steps that are null
    while(steps[step_id].repeat_operation == 0 && step_id>0){
        step_id--;
    }

    // step is now the parent node for this iteration
    step = &steps[step_id];
    size_t numLoadedElm = step->load_num_elem;

    void *cp_dst = dst;
    const void *cp_src = src;
    char *cp_src_temp=NULL;
    char *cp_dst_temp=NULL;

    if(numLoadedElm>32){
        if(step_id > 0){
            for(step_itr=0; step_itr<step->repeat_operation; step_itr++){
                src = broadcast_node(steps, step_id-1, dst, src);
                cp_src = dst;
                cp_dst = dst + numLoadedElm;
                for(rep_itr=1; rep_itr<step->replicate_loadedElm_times; rep_itr++){
                    xa_nn_memcpy(cp_dst, cp_src, numLoadedElm);
                    cp_dst += numLoadedElm;
                }
                dst = cp_dst;
            }
            return src;
        } else {        
            if(numLoadedElm == 1){
                for(j=0; j<step->repeat_operation; j++){
                    memset(cp_dst, *(WORD8 *)cp_src, step->replicate_loadedElm_times);
                    cp_dst += step->replicate_loadedElm_times;
                    cp_src++;
                }
            } else {
                for(j=0; j<step->repeat_operation; j++){
                    for(i=0; i<step->replicate_loadedElm_times; i++){
                        xa_nn_memcpy(cp_dst, cp_src, numLoadedElm);
                        cp_dst += numLoadedElm;
                    }
                    cp_src += numLoadedElm;
                }
            }
            return cp_src;
        }
    }
    else{
        if(step_id > 0){
            for(step_itr=0; step_itr<step->repeat_operation; step_itr++){
                src = broadcast_node(steps, step_id-1, dst, src);
                cp_src = dst;
                cp_dst = dst + numLoadedElm;
                for(rep_itr=1; rep_itr<step->replicate_loadedElm_times; rep_itr++){
                    for(k=0; k<(int)numLoadedElm; k++){
                        cp_src_temp = (char *)cp_src;
                        cp_dst_temp = (char *)cp_dst;
                        cp_dst_temp[k] = cp_src_temp[k];
                    }
                    cp_dst += numLoadedElm;
                }
                dst = cp_dst;
            }
            return src;
        } else {
            if(numLoadedElm == 1){
                for(j=0; j<step->repeat_operation; j++){
                    memset(cp_dst, *(WORD8 *)cp_src, step->replicate_loadedElm_times);
                    cp_dst += step->replicate_loadedElm_times;
                    cp_src++;
                }
            } else {
                for(j=0; j<step->repeat_operation; j++){
                    for(i=0; i<step->replicate_loadedElm_times; i++){
                        for(k=0; k<(int)numLoadedElm; k++){
                            cp_src_temp = (char *)cp_src;
                            cp_dst_temp = (char *)cp_dst;
                            cp_dst_temp[k] = cp_src_temp[k];
                        }
                        cp_dst += numLoadedElm;
                    }
                    cp_src += numLoadedElm;
                }
            }
            return cp_src;
        }
    }
}
