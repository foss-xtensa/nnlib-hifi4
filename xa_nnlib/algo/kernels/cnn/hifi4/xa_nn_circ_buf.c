/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "common.h"
#include <string.h>
#include "xa_nn_circ_buf.h"
#include "xa_nnlib_common_macros.h"

#include "xa_nnlib_common.h"
int xa_nn_circ_buf_nchw_getsize(
    WORD32 bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width)
{
  int circ_buf_width;
  int size_in_bytes;

  if (0 != ((circ_buf_height - kernel_height) % y_stride))
  {
    return -2;
  }

  circ_buf_width = kernel_width + ((output_width - 1) * x_stride);
  circ_buf_width = XT_MAX(circ_buf_width, x_padding + input_width);

  /* Aligned size independent of bytewidth */
  circ_buf_width = ALIGNED_SIZE(circ_buf_width, 4);

  size_in_bytes = bytewidth*circ_buf_height*circ_buf_width;

  if (0 > size_in_bytes)
  {
    /* If callee of this function interprets received value from here in
     * unsigned value then negative returned value will be interpreted as
     * large positive number which will explode the memory allocations.
     * Callee of this function should take care of the negative returned
     * values. */
    return -3;
  }
  else
  {
    return size_in_bytes;
  }
}

VOID xa_nn_circ_buf_nchw_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width)
{
  /* No. of row in circular buf */
  p_circ_buf->rows       = circ_buf_height;
  p_circ_buf->row_offset = kernel_width + ((output_width - 1) * x_stride);
  p_circ_buf->row_offset = XT_MAX(p_circ_buf->row_offset, x_padding + input_width);
  /* Aligned independent of bytewidth */
  p_circ_buf->row_offset = ALIGNED_SIZE(p_circ_buf->row_offset, 4);
  p_circ_buf->bytewidth  = bytewidth;
  /* Initialize circular buffer pointers */
  p_circ_buf->p_begin    = p_mem;
  p_circ_buf->p_curr     = p_mem;
  p_circ_buf->p_end      = (((char *)p_mem) + p_circ_buf->rows*p_circ_buf->row_offset*bytewidth);
}

void xa_nn_circ_buf_nchw_add_rows(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 left_padding,
    WORD32 input_width,
    WORD32 n_rows,
    WORD32 top_pad,
    WORD32 bottom_pad)
{
    int i;
    int bytewidth = p_circ_buf->bytewidth;

    /* Error checks */
    if (n_rows < (top_pad + bottom_pad))
    {
      return;
    }
    if (p_circ_buf->row_offset < (left_padding + input_width))
    {
      return;
    }

    const WORD8 *p_src = (const WORD8 *)p_inp;
    pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;

    /* Add top padding rows */
    for(i = 0; i < top_pad; i++)
    {
#if 0
        for(j = 0; j < p_circ_buf->row_offset; j++)
        {
            p_dst[j] = 0;
        }
#else
        memset(p_dst, 0, p_circ_buf->row_offset*p_circ_buf->bytewidth);
#endif
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
    }
    /* Add input rows with left and right padding */
    for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
    {
#if 0
        for(j = 0; j < left_padding; j++)
        {
            p_dst[j] = 0;
        }
        for(; j < (left_padding + input_width); j++)
        {
            p_dst[j] = p_src[i*input_width+(j-left_padding)];
        }
        for(; j < p_circ_buf->row_offset; j++)
        {
            p_dst[j] = 0;
        }
#else
        memset(p_dst, 0, left_padding*bytewidth);
        memcpy(&p_dst[left_padding*bytewidth], &p_src[i*input_width*bytewidth], input_width*bytewidth);
        memset(&p_dst[(left_padding+input_width)*bytewidth], 0, (p_circ_buf->row_offset-(left_padding+input_width))*bytewidth);
#endif
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
    }
    /* Add bottom padding rows */
    for(i = 0; i < bottom_pad; i++)
    {
#if 0
        for(j = 0; j < p_circ_buf->row_offset; j++)
        {
            p_dst[j] = 0;
        }
#else
        memset(p_dst, 0, p_circ_buf->row_offset*bytewidth);
#endif
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
    }
    /* Update current pointer for circular buffer */
    p_circ_buf->p_curr = (pVOID)p_dst;
}

void xa_nn_circ_buf_nchw_add_rows_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 left_padding,
    WORD32 input_width,
    WORD32 n_rows,
    WORD32 top_pad,
    WORD32 bottom_pad,
    pVOID p_pad_val)
{
    int i;
    int bytewidth = p_circ_buf->bytewidth;

    /* Error checks */
    if (n_rows < (top_pad + bottom_pad))
    {
      return;
    }
    if (p_circ_buf->row_offset < (left_padding + input_width))
    {
      return;
    }

    if(bytewidth == 1)
    {
        const WORD8 *p_src = (const WORD8 *)p_inp;
        pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;
        WORD8 pad_val = *(pWORD8)p_pad_val;
        /* Add top padding rows */
        for(i = 0; i < top_pad; i++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset);
        }
        /* Add input rows with left and right padding */
        for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
        {
            /* Left padding */
            memset(p_dst, pad_val, left_padding);
            /* Input */
            memcpy(&p_dst[left_padding], &p_src[i*input_width], input_width);
            /* Right padding */
            memset(&p_dst[(left_padding+input_width)], pad_val, (p_circ_buf->row_offset-(left_padding+input_width)));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset);
        }
        /* Add bottom padding rows */
        for(i = 0; i < bottom_pad; i++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset);
        }
        /* Update current pointer for circular buffer */
        p_circ_buf->p_curr = (pVOID)p_dst;
    }
    else if(bytewidth == 2)
    {
        int j;
        pWORD16 p_src = (pWORD16)p_inp;
        pWORD16 p_dst = (pWORD16)p_circ_buf->p_curr;
        WORD16 pad_val = *(WORD16 *)p_pad_val;
        /* Add top padding rows */
        for(i = 0; i < top_pad; i++)
        {
            ae_int16x4 *p_dst16x4 = (ae_int16x4 *)p_dst;
            ae_int16x4 d_pad_val = AE_MOVDA16(pad_val);
            for(j = 0; j < (p_circ_buf->row_offset>>2); j++)
            {
                p_dst16x4[j] = d_pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<1);
        }
        /* Add input rows with left and right padding */
        for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
        {
            /* Left padding */
            for(j = 0; j < left_padding; j++)
            {
                p_dst[j] = pad_val;
            }
            /* Input */
            memcpy(&p_dst[left_padding], &p_src[i*input_width], (input_width<<1));
            /* Right padding */
            for(j = left_padding + input_width; j < p_circ_buf->row_offset; j++)
            {
                p_dst[j] = pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<1);
        }
        /* Add bottom padding rows */
        for(i = 0; i < bottom_pad; i++)
        {
            ae_int16x4 *p_dst16x4 = (ae_int16x4 *)p_dst;
            ae_int16x4 d_pad_val = AE_MOVDA16(pad_val);
            for(j = 0; j < (p_circ_buf->row_offset>>2); j++)
            {
                p_dst16x4[j] = d_pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<1);
        }
        /* Update current pointer for circular buffer */
        p_circ_buf->p_curr = (pVOID)p_dst;
    }
    else if(bytewidth == 4)
    {
        int j;
        pWORD32 p_src = (pWORD32)p_inp;
        pWORD32 p_dst = (pWORD32)p_circ_buf->p_curr;
        WORD32 pad_val = *(WORD32 *)p_pad_val;
        /* Add top padding rows */
        for(i = 0; i < top_pad; i++)
        {
            ae_int32x2 *p_dst32x2 = (ae_int32x2 *)p_dst;
            ae_int32x2 d_pad_val = AE_MOVDA32(pad_val);
            for(j = 0; j < (p_circ_buf->row_offset>>1); j++)
            {
                p_dst32x2[j] = d_pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<2);
        }
        /* Add input rows with left and right padding */
        for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
        {
            /* Left padding */
            for(j = 0; j < left_padding; j++)
            {
                p_dst[j] = pad_val;
            }
            /* Input */
            memcpy(&p_dst[left_padding], &p_src[i*input_width], input_width<<2);
            /* Right padding */
            for(j = left_padding + input_width; j < p_circ_buf->row_offset; j++)
            {
                p_dst[j] = pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<2);
        }
        /* Add bottom padding rows */
        for(i = 0; i < bottom_pad; i++)
        {
            ae_int32x2 *p_dst32x2 = (ae_int32x2 *)p_dst;
            ae_int32x2 d_pad_val = AE_MOVDA32(pad_val);
            for(j = 0; j < (p_circ_buf->row_offset>>1); j++)
            {
                p_dst32x2[j] = d_pad_val;
            }
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset<<2);
        }
        /* Update current pointer for circular buffer */
        p_circ_buf->p_curr = (pVOID)p_dst;
    }
}

int xa_nn_circ_buf_nhwc_getsize(
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height)
{
  int circ_buf_height, circ_buf_channels;
  int size_in_bytes;

  circ_buf_height = kernel_height + ((output_height - 1) * y_stride);
  circ_buf_height = XT_MAX(circ_buf_height, y_padding + input_height);

  if(bytewidth == 4)
  {
    circ_buf_channels = ALIGNED_SIZE(input_channels*channels_multiplier, 2);
  }
  else
  {
    circ_buf_channels = ALIGNED_SIZE(input_channels*channels_multiplier, 4);
  }

  size_in_bytes = bytewidth*circ_buf_height*circ_buf_channels*kernel_width;

  if (0 > size_in_bytes)
  {
    return -1;
  }
  else
  {
    return size_in_bytes;
  }
}

VOID xa_nn_circ_buf_nhwc_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height)
{
  int circ_buf_height;

  circ_buf_height = kernel_height + ((output_height - 1) * y_stride);
  circ_buf_height = XT_MAX(circ_buf_height, y_padding + input_height);
  /* No. of row in circular buf */
  p_circ_buf->rows       = circ_buf_height*kernel_width;
  if(bytewidth == 4)
  {
    p_circ_buf->row_offset = ALIGNED_SIZE(input_channels*channels_multiplier, 2);
  }
  else
  {
    p_circ_buf->row_offset = ALIGNED_SIZE(input_channels*channels_multiplier, 4);
  }
  p_circ_buf->bytewidth  = bytewidth;
  /* Initialize circular buffer pointers */
  p_circ_buf->p_begin    = p_mem;
  p_circ_buf->p_curr     = p_mem;
  p_circ_buf->p_end      = (((char *)p_mem) + p_circ_buf->rows*p_circ_buf->row_offset*bytewidth);
}

void xa_nn_circ_buf_nhwc_add_cols(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad)
{
    int i, j, k;
    int circ_buf_height;
    int bytewidth = p_circ_buf->bytewidth;

    circ_buf_height = kernel_height + ((output_height - 1) * y_stride);
    circ_buf_height = XT_MAX(circ_buf_height, y_padding + input_height);

    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_circ_buf->p_curr, circ_buf_width*p_circ_buf->row_offset*bytewidth);
    const WORD8 *p_src = (const WORD8 *)p_inp;
    pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;

    /* Add left padding */
    for(i = 0; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, i*p_circ_buf->row_offset*circ_buf_width*bytewidth);
        for(j = 0; j < left_pad; j++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Add input strips */
    for(i = 0; i < top_padding; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < n_cols - (left_pad + right_pad); j++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Padding in depth dimension doesn't need to be initialized as output doesn't depend on it */
    for(; i < top_padding + input_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        if(channels_multiplier == 1)
        {
            for(j = 0; j < n_cols - (left_pad + right_pad); j++)
            {
                memcpy(p_dst, &p_src[((i-top_padding)*input_width + j)*input_channels*bytewidth], input_channels*bytewidth);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
            }
        }
        else
        {
            WORD32 cm;
            if(bytewidth == 1)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD8 *p_src1 = p_src;
                    pWORD8 p_dst1 = p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD8 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
            else if(bytewidth == 2)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD16 *p_src1 = (const WORD16 *)p_src;
                    pWORD16 p_dst1 = (pWORD16)p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD16 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
            else if(bytewidth == 4)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD32 *p_src1 = (const WORD32 *)p_src;
                    pWORD32 p_dst1 = (pWORD32)p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD32 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
        }
    }
    for(; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < n_cols - (left_pad + right_pad); j++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Add right padding */
    for(i = 0; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width + (n_cols - right_pad))*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < right_pad; j++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Update current pointer for circular buffer */
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_circ_buf->p_curr, (n_cols-circ_buf_width)*p_circ_buf->row_offset*bytewidth);
}

void xa_nn_circ_buf_nhwc_add_cols_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 top_padding,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 circ_buf_width,
    WORD32 channels_multiplier,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 output_height,
    WORD32 n_cols,
    WORD32 left_pad,
    WORD32 right_pad,
    pVOID  p_pad_val)
{
    int i, j, k;
    int circ_buf_height;
    int bytewidth = p_circ_buf->bytewidth;
    WORD8 pad_val = *(pWORD8)p_pad_val;

    circ_buf_height = kernel_height + ((output_height - 1) * y_stride);
    circ_buf_height = XT_MAX(circ_buf_height, y_padding + input_height);

    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_circ_buf->p_curr, circ_buf_width*p_circ_buf->row_offset*bytewidth);
    const WORD8 *p_src = (const WORD8 *)p_inp;
    pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;

    /* Add left padding */
    for(i = 0; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, i*p_circ_buf->row_offset*circ_buf_width*bytewidth);
        for(j = 0; j < left_pad; j++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Add input strips */
    for(i = 0; i < top_padding; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < n_cols - (left_pad + right_pad); j++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Padding in depth dimension doesn't need to be initialized as output doesn't depend on it */
    for(; i < top_padding + input_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        if(channels_multiplier == 1)
        {
            for(j = 0; j < n_cols - (left_pad + right_pad); j++)
            {
                memcpy(p_dst, &p_src[((i-top_padding)*input_width + j)*input_channels*bytewidth], input_channels*bytewidth);
                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
            }
        }
        else
        {
            WORD32 cm;
            if(bytewidth == 1)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD8 *p_src1 = p_src;
                    pWORD8 p_dst1 = p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD8 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
            else if(bytewidth == 2)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD16 *p_src1 = (const WORD16 *)p_src;
                    pWORD16 p_dst1 = (pWORD16)p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD16 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
            else if(bytewidth == 4)
            {
                for(j = 0; j < n_cols - (left_pad + right_pad); j++)
                {
                    const WORD32 *p_src1 = (const WORD32 *)p_src;
                    pWORD32 p_dst1 = (pWORD32)p_dst;
                    for(k = 0; k < input_channels; k++)
                    {
                        WORD32 val = p_src1[((i-top_padding)*input_width + j)*input_channels + k];
                        for(cm = 0; cm < channels_multiplier; cm++)
                        {
                            p_dst1[k*channels_multiplier + cm] = val;
                        }
                    }
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
                }
            }
        }
    }
    for(; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width+left_pad)*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < n_cols - (left_pad + right_pad); j++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Add right padding */
    for(i = 0; i < circ_buf_height; i++)
    {
        p_dst = (pWORD8)p_circ_buf->p_curr;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (i*circ_buf_width + (n_cols - right_pad))*p_circ_buf->row_offset*bytewidth);
        for(j = 0; j < right_pad; j++)
        {
            memset(p_dst, pad_val, p_circ_buf->row_offset*bytewidth);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, p_circ_buf->row_offset*bytewidth);
        }
    }
    /* Update current pointer for circular buffer */
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_circ_buf->p_curr, (n_cols-circ_buf_width)*p_circ_buf->row_offset*bytewidth);
}
