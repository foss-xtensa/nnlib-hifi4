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
#ifndef __XA_NNLIB_COMMON_BCAST_MACRO_H__
#define __XA_NNLIB_COMMON_BCAST_MACRO_H__

#ifndef AE_MOVBA
#ifdef AE_MOVBA1
#define AE_MOVBA AE_MOVBA1
#else
#define AE_MOVBA(inp) inp
#endif
#endif

typedef struct {
  WORD32  out_zero_bias;
  WORD32  out_shift;
  WORD32  out_multiplier;
  WORD32  out_activation_min;
  WORD32  out_activation_max;
  WORD32  inp1_zero_bias;
  WORD32  inp1_left_shift;
  WORD32  inp1_multiplier;
  WORD32  inp2_zero_bias;
  WORD32  inp2_left_shift;
  WORD32  inp2_multiplier;
  WORD32  left_shift;
  WORD32  inp_elm_size;
  WORD32  out_elm_size;
  WORD32  multiplier_sign;
  WORD32  in_lc;
  WORD32  out_lc;
  WORD32  num_elm;
  xtbool  sign_flag;
  compare_ops_t kernel_type;
} bcast_args_t;

typedef void (*internal_2D) (void * __restrict__ p_out,
    const    void * __restrict__ p_inp1,
    const    void * __restrict__ p_inp2,
              bcast_args_t* args);

typedef void (*internal_1D) (void * __restrict__ p_out,
    const    void * __restrict__ p_inp1,
    const    void * __restrict__ p_inp2,
              bcast_args_t* args);

static inline WORD32 CALL_BCAST(internal_2D internal_2D_func,
  internal_1D internal_1D_func,
  void * __restrict__ p_out,
  const WORD32 *const p_out_shape,
  const void * __restrict__ p_inp1,
  const WORD32 *const p_inp1_shape,
  const void * __restrict__ p_inp2,
  const WORD32 *const p_inp2_shape,
  bcast_args_t* args
  ) 
{
  /* Check shapes */
  int i;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0; 
  int inp1_const = 1, inp2_const = 1; 
  for(i = 0; i < 4; i++) 
  { 
    if(p_inp1_shape[i] != p_inp2_shape[i]) 
    { 
      if(p_inp1_shape[i] == 1) 
        inp1_strides[i] = 0; 
      else 
        inp2_strides[i] = 0; 

      need_broadcast = 1; 
    } 
    if(p_inp1_shape[i] != 1) 
      inp1_const &= 0; 
    if(p_inp2_shape[i] != 1) 
      inp2_const &= 0; 
  } 
  int itr0, itr1, itr2; 

  void *p_out_tmp = p_out; 
  const void *__restrict__ p_inp1_tmp = p_inp1;
  const void *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0) 
  { 
    args->sign_flag = AE_MOVBA(0);
    args->in_lc = p_out_shape[0] * inp1_strides[0];
    args->out_lc = 1;
    (*internal_2D_func)(
          p_out,
          p_inp1,
          p_inp2,
          args); 
  } 
  else if(inp1_const == 1 || inp2_const == 1) 
  {
    args->sign_flag = AE_MOVBA(0);
    if(inp1_const == 1)
    { 
      args->sign_flag = AE_MOVBA(1);
      WORD32 tmp_zb, tmp_ls, tmp_mult;
      tmp_zb   = args->inp1_zero_bias;
      tmp_ls   = args->inp1_left_shift;
      tmp_mult = args->inp1_multiplier;
      args->inp1_zero_bias = args->inp2_zero_bias;
      args->inp1_left_shift = args->inp2_left_shift;
      args->inp1_multiplier = args->inp2_multiplier;
      args->inp2_zero_bias = tmp_zb;
      args->inp2_left_shift = tmp_ls;
      args->inp2_multiplier = tmp_mult;
      args->out_multiplier = args->multiplier_sign * args->out_multiplier;
      const void *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    } 
    args->num_elm = p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3];
    (*internal_1D_func)(
        p_out_tmp,
        p_inp1_tmp,
        p_inp2_tmp,
        args); 
  } 
  else if(inp1_strides[3] == inp2_strides[3])
  {
    args->sign_flag = AE_MOVBA(0);
    args->in_lc = p_out_shape[2] * p_out_shape[3];
    args->out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      args->sign_flag = AE_MOVBA(1);
      WORD32 tmp_zb, tmp_ls, tmp_mult;
      tmp_zb   = args->inp1_zero_bias;
      tmp_ls   = args->inp1_left_shift;
      tmp_mult = args->inp1_multiplier;
      args->inp1_zero_bias = args->inp2_zero_bias;
      args->inp1_left_shift = args->inp2_left_shift;
      args->inp1_multiplier = args->inp2_multiplier;
      args->inp2_zero_bias = tmp_zb;
      args->inp2_left_shift = tmp_ls;
      args->inp2_multiplier = tmp_mult;
      args->out_multiplier = args->multiplier_sign * args->out_multiplier;
      const void *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      
      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      
      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      args->in_lc = p_out_shape[3];
      args->out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      args->in_lc = p_out_shape[3];
      args->out_lc = p_out_shape[2];
    }

    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const void *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const void *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        (*internal_2D_func)(
            p_out_tmp,
            p_inp1_tmp0,
            p_inp2_tmp0,
            args);
        p_out_tmp += args->in_lc * args->out_lc * args->out_elm_size;
        p_inp1_tmp0 += inp1_strides[1] * args->inp_elm_size;
        p_inp2_tmp0 += inp2_strides[1] * args->inp_elm_size;
      }
      p_inp1_tmp += inp1_strides[0] * args->inp_elm_size;
      p_inp2_tmp += inp2_strides[0] * args->inp_elm_size;
    }
  } 
  else 
  {
    args->sign_flag = AE_MOVBA(0);
    if(inp1_strides[3] == 0)
    {
      args->sign_flag = AE_MOVBA(1);
      WORD32 tmp_zb, tmp_ls, tmp_mult;
      tmp_zb   = args->inp1_zero_bias;
      tmp_ls   = args->inp1_left_shift;
      tmp_mult = args->inp1_multiplier;
      args->inp1_zero_bias = args->inp2_zero_bias;
      args->inp1_left_shift = args->inp2_left_shift;
      args->inp1_multiplier = args->inp2_multiplier;
      args->inp2_zero_bias = tmp_zb;
      args->inp2_left_shift = tmp_ls;
      args->inp2_multiplier = tmp_mult;
      args->out_multiplier = args->multiplier_sign * args->out_multiplier;
      const void *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];
      
      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    args->num_elm = p_out_shape[3];
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const void *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const void *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const void *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const void *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            (*internal_1D_func)(
                p_out_tmp,
                p_inp1_tmp1,
                p_inp2_tmp1,
                args);
          }
          p_out_tmp += p_out_shape[3] * args->out_elm_size;
          p_inp1_tmp1 += inp1_strides[2] * args->inp_elm_size;
          p_inp2_tmp1 += inp2_strides[2] * args->inp_elm_size;
        }
        p_inp1_tmp0 += inp1_strides[1] * args->inp_elm_size;
        p_inp2_tmp0 += inp2_strides[1] * args->inp_elm_size;
      }
      p_inp1_tmp += inp1_strides[0] * args->inp_elm_size;
      p_inp2_tmp += inp2_strides[0] * args->inp_elm_size;
    } 
  }
  return 0;
}

#endif /* __XA_NNLIB_COMMON_BCAST_MACRO_H__ */
