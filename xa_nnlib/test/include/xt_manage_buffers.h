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
#ifndef __XT_MANAGE_BUFFERS_H__
#define __XT_MANAGE_BUFFERS_H__
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "xa_type_def.h"

#define ASYM8_TYPE -3
#define ASYM8S_TYPE -4
#define SYM8S_TYPE -5
#define ASYM16S_TYPE -7
#define SYM16S_TYPE -8
#define FLOAT_TYPE -1
#define ASYM32S_TYPE -10

typedef struct _buf1D_t{
  void *p;
  int length;
  int precision;
  int bytes_per_element;
}buf1D_t;

typedef struct _buf2D_t{
  void *p;
  int cols;
  int rows;
  int row_offset;
  int precision;
  int bytes_per_element;
}buf2D_t;

buf1D_t *create_buf1D(int len, int precision);
buf2D_t *create_buf2D(int rows, int cols, int row_offset, int precision, int membank_padding);

void free_buf1D(buf1D_t *ptr);
void free_buf2D(buf2D_t *ptr);

int set_rand_inp_buf1D(buf1D_t *ptr_buf1D);
int set_rand_inp_buf2D(buf2D_t *ptr_buf2D);

void write_buf1D(buf1D_t *pbuf, FILE *file,int extensionIndicator, char * var_name);
void write_buf2D(buf2D_t *pbuf, FILE *file,int extensionIndicator, char * var_name);

int compare_buf1D(buf1D_t *pbuf_ref, buf1D_t *pbuf_out, int method, int precision, int sum_length);
int compare_buf2D(buf2D_t *pbuf_ref, buf2D_t *pbuf_out, int method);

int interleave_buf1D_real(buf1D_t *pbuf, buf1D_t *pbuf_interleaved, int length);
int deinterleave_buf1D_real(buf1D_t *pbuf, buf1D_t *pbuf_deinterleaved, int length);
int interleave_buf1D(buf1D_t *pbuf, buf1D_t *pbuf_interleaved, int length);
int deinterleave_buf1D(buf1D_t *pbuf, buf1D_t *pbuf_deinterleaved, int length);

unsigned int datatype_size(int precision);

#endif // __XT_MANAGE_BUFFERS_H__
