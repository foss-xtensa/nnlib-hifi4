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
#include "xt_manage_buffers.h"

int read_buf1D_from_file(FILE *fptr_read_data, buf1D_t *ptr_buf1D); 
int read_buf2D_from_file(FILE *fptr_read_data, buf2D_t *ptr_buf2D, int pad_val); 
int write_buf1D_to_file(FILE *fptr_write_data, buf1D_t *ptr_buf1D); 
int write_buf2D_to_file(FILE *fptr_write_data, buf2D_t *ptr_buf2D); 
int load_matXvec_input_data(int write_file, FILE *fptr_inp, buf2D_t *p_mat1, buf1D_t *p_vec1, 
    buf2D_t *p_mat2, buf1D_t *p_vec2, buf1D_t *p_bias); 
int load_conv2d_std_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel,
    buf1D_t *p_bias, int input_channels, int input_channels_pad, int kernel_pad_val);
int load_conv1d_std_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel, 
    buf1D_t *p_bias, int input_channels, int input_width, int input_channelsXwidth_pad,
    int kernel_pad_val);
int load_conv2d_ds_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel,
    buf1D_t *p_bias, buf1D_t *p_kernel_point, buf1D_t *p_bias_point, int kernel_pad_val);
int load_dilated_conv2d_depth_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel,
    buf1D_t *p_bias, int kernel_pad_val);
int load_conv2d_pt_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp,
    buf1D_t *p_kernel_point, buf1D_t *p_bias_point);
int load_activation_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf1D_t *p_inp_alpha, char *kernel_name); 
int load_pool_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp);
int load_norm_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp);
int load_batch_norm_3D_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf1D_t *p_alpha, buf1D_t *p_beta);
int load_basic_func_data(int write_file, FILE *fptr_inp1, FILE *fptr_inp2, buf1D_t *p_inp1, buf1D_t *p_inp2);
int load_reorg_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp);
int write_output_data(FILE *fptr_out, buf1D_t *p_out); 
FILE* file_open(char *file_path, char *file_name, char *mode, int max_file_name_length);
