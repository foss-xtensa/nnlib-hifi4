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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <xtensa/config/core-isa.h>
#include "xa_type_def.h"
#include "nnlib/xa_nnlib_api.h"
#include "xt_manage_buffers.h"
#include "cmdline_parser.h"
#include "file_io.h"
#include "xa_nnlib_standards.h"

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_KERNEL_NAME_LENGTH 25

#define XA_MAX_CMD_LINE_LENGTH 200
#define XA_MAX_ARGS 100
#define SHAPE_ARGS_LENGTH 80
#define MAX_NUM_INPS_CONCAT 10
#define MAX_NUM_OUTS_SPLIT_V 10
#define MAX_DIMS 8
#define MAX_DIMS_FOR_STRIDED_SLICE 5
#define MAX_DIMS_FOR_CONCAT 6
#define MAX_DIMS_FOR_SPLIT_V 6 //doubtful
#define PARAMFILE "paramfilesimple_reorg.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{

  int help;
  int inp_data_format; // 0 for nhwc; 1 for chw/dhw
  int num_inp_dims;
  int num_pad_dims;
  int num_out_dims;
  int pad_value;
  int input_shape_extended[MAX_DIMS_FOR_STRIDED_SLICE];
  int input_shape[MAX_DIMS];
  int concat_inputs_shape[MAX_NUM_INPS_CONCAT * MAX_DIMS_FOR_CONCAT];
  int split_v_outputs_shape[MAX_NUM_OUTS_SPLIT_V * MAX_DIMS_FOR_SPLIT_V];
  int output_shape[MAX_DIMS];  
  int pad_shape[MAX_DIMS];
  int pad_values[MAX_DIMS];
  int permute_vec[MAX_DIMS];  
  char read_concat_inps_shape_str[MAX_NUM_INPS_CONCAT * SHAPE_ARGS_LENGTH];
  char read_split_v_outs_shape_str[MAX_NUM_OUTS_SPLIT_V * SHAPE_ARGS_LENGTH];
  char read_inp_shape_str[SHAPE_ARGS_LENGTH];
  char read_pad_shape_str[SHAPE_ARGS_LENGTH];
  char read_out_shape_str[SHAPE_ARGS_LENGTH];
  char read_pad_values_str[SHAPE_ARGS_LENGTH];
  char read_permute_vec_str[SHAPE_ARGS_LENGTH];
  int block_sizes[MAX_DIMS-2];
  int crop_or_pad_sizes[2*(MAX_DIMS)-2];
  char read_block_sizes_str[SHAPE_ARGS_LENGTH];
  char read_crop_or_pad_sizes_str[SHAPE_ARGS_LENGTH];
  int num_inputs;
  int num_outputs;
  int input_batch;
  int input_height;
  int input_width;
  int input_channels;
  int block_size;
  int out_batch;
  int out_height;
  int out_width;
  int out_channels;
  int align_corners;
  int half_pixel_centers;
  int inp_precision;
  int out_precision;
  int start_0,stop_0;
  int start_1,stop_1;
  int start_2,stop_2;
  int start_3,stop_3;
  int start_4,stop_4;
  int stride_0,stride_1,stride_2,stride_3,stride_4;
  int axis;
  char kernel_name[MAX_KERNEL_NAME_LENGTH];
  int frames;
  int write_file;
  char read_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_ref_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_out_file_name[XA_MAX_CMD_LINE_LENGTH];
  int verify;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  {
    p_cfg->help     = 0;
    p_cfg->inp_data_format = 0;
    p_cfg->num_inputs = 1;
    p_cfg->num_outputs = 1;
    p_cfg->num_inp_dims  = 4;
    p_cfg->num_pad_dims  = 2;
    p_cfg->num_out_dims  = 4;
    p_cfg->pad_value  = 0;
    p_cfg->input_batch = 1;
    p_cfg->input_height = 16;
    p_cfg->input_width = 16;
    p_cfg->input_channels = 16;
    p_cfg->block_size = 2;
    p_cfg->out_batch = 1;
    p_cfg->out_height = 32;
    p_cfg->out_width = 32;
    p_cfg->out_channels = 4;
    p_cfg->align_corners = 0;
    p_cfg->half_pixel_centers = 0;
    p_cfg->inp_precision = 8;
    p_cfg->out_precision = 8;
    p_cfg->start_0 = 0;
    p_cfg->start_1 = 0;
    p_cfg->start_2 = 0;
    p_cfg->start_3 = 0;
    p_cfg->start_4 = 0;
    p_cfg->stop_0 =  1;
    p_cfg->stop_1 =  1;
    p_cfg->stop_2 =  1;
    p_cfg->stop_3 =  1;
    p_cfg->stop_4 =  1;
    p_cfg->stride_0 = 1;
    p_cfg->stride_1 = 1;
    p_cfg->stride_2 = 1;
    p_cfg->stride_3 = 1;
    p_cfg->stride_4 = 1;
    p_cfg->axis = 0;
    strcpy(p_cfg->kernel_name, "depth_to_space");
    p_cfg->frames   = 2;
    p_cfg->write_file = 0;
    p_cfg->read_inp_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;
    p_cfg->read_concat_inps_shape_str[0] = '\0';
    p_cfg->read_split_v_outs_shape_str[0] = '\0';
    p_cfg->read_inp_shape_str[0] = '\0';
    p_cfg->read_pad_shape_str[0] = '\0';
    p_cfg->read_out_shape_str[0] = '\0';
    p_cfg->read_pad_values_str[0] = '\0';
    p_cfg->read_permute_vec_str[0] = '\0';
    int itr;
    for(itr = 0; itr < MAX_NUM_INPS_CONCAT * MAX_DIMS_FOR_CONCAT; itr++)
    {
      p_cfg->concat_inputs_shape[itr] = 1;
    }
    for(itr = 0; itr < MAX_NUM_OUTS_SPLIT_V * MAX_DIMS_FOR_SPLIT_V; itr++)
    {
      p_cfg->split_v_outputs_shape[itr] = 1;
    }
    for(itr = 0; itr < MAX_DIMS; itr++)
    {
      p_cfg->input_shape[itr] = 1;
      p_cfg->output_shape[itr] = 1;
      p_cfg->pad_values[itr] = 0;
      p_cfg->permute_vec[itr] = 1;
    }
    for(itr = 0; itr < MAX_DIMS_FOR_STRIDED_SLICE; itr++)
    {
      p_cfg->input_shape_extended[itr] = 1;
    }
    p_cfg->pad_shape[0] = 4;
    p_cfg->pad_shape[1] = 2;

    p_cfg->read_block_sizes_str[0] = '\0';
    p_cfg->read_crop_or_pad_sizes_str[0] = '\0';
    for(itr = 0; itr < 4-2; itr++)
    {
      p_cfg->block_sizes[itr] = 1;
      p_cfg->crop_or_pad_sizes[2*itr] = 0;
      p_cfg->crop_or_pad_sizes[2*itr+1] = 0;
    }

    return 0;
  }
  else
  {
    return -1;
  }
}

void show_usage(void)
{
    printf ("Usage xt-run <binary> [Options]\n");
    printf("\t-inp_data_format: data format of input and output, 0 for nhwc; Default=0\n");
    printf("\t-num_inputs: number of inputs for concat kernels; Default=1\n");
    printf("\t-num_ouputs: number of outputs for split_v kernels; Default=1\n");
    printf("\t-num_inp_dims: number of input dimensions; Default=4\n");
    printf("\t-num_pad_dims: number of pad dimensions; Default=2\n");
    printf("\t-num_out_dims: number of output dimensions; Default=4\n");
    printf("\t-pad_value: input to be padded with this pad value; Default=0\n");
    printf("\t-input_batch: input batch; Default=1\n");
    printf("\t-input_height: input height; Default=16\n");
    printf("\t-input_width: input width; Default=16\n");
    printf("\t-input_channels: input channels; Default=16\n");
    printf("\t-block_size: block size; Default=2\n");
    printf("\t-out_batch: output batch; Default=1\n");
    printf("\t-out_height: output height; Default=16\n");
    printf("\t-out_width: output width; Default=16\n");
    printf("\t-out_channels: output channels; Default=4\n");
    printf("\t-align_corners: align corners while resizing, 0 or 1; Default=0\n");
    printf("\t-half_pixel_centers: half pixel center for resizing, 0 or 1; Default=0\n");
    printf("\t-start_0: begin point for dimention 0; Default=0\n");
    printf("\t-start_1: begin point for dimention 1; Default=0\n");
    printf("\t-start_2: begin point for dimention 2; Default=0\n");
    printf("\t-start_3: begin point for dimention 3; Default=0\n");
    printf("\t-start_4: begin point for dimention 4; Default=0\n");
    printf("\t-stop_0: end point for dimention 0; Default=1\n");
    printf("\t-stop_1: end point for dimention 1; Default=1\n");
    printf("\t-stop_2: end point for dimention 2; Default=1\n");
    printf("\t-stop_3: end point for dimention 3; Default=1\n");
    printf("\t-stop_4: end point for dimention 4; Default=1\n");
    printf("\t-stride_0: stride for dimention 0; Default=1\n");
    printf("\t-stride_1: stride for dimention 1; Default=1\n");
    printf("\t-stride_2: stride for dimention 2; Default=1\n");
    printf("\t-stride_3: stride for dimention 3; Default=1\n");
    printf("\t-stride_4: stride for dimention 4; Default=1\n");
    printf("\t-axis: axis dimension for concat or split_v kernel (-num_inp_dims to num_inp_dims-1); Default=0\n");
    printf("\t-inp_precision: 8, 16; Default=8\n");
    printf("\t-out_precision: 8, 16; Default=8\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: depth_to_space, space_to_depth, pad, batch_to_space_nd, space_to_batch_nd, strided_slice, resize_bilinear, resize_nearest_neighbour; Default=""depth_to_space""\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - inp) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - inp) \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
    printf("\t-concat_inps_shape: Takes the input shape dimensions for all inputs (num_inputs * num_inp_dims values space ' ' separated) \n");
    printf("\t-split_v_outs_shape: Takes the output shape dimensions for all outputs (num_outputs * num_out_dims values space ' ' separated) \n");
    printf("\t-inp_shape: Takes the input shape dimensions (num_inp_dims values space ' ' separated) \n");
    printf("\t-pad_shape: Takes the pad shape dimensions (num_pad_dims values space ' ' separated) \n");
    printf("\t-out_shape: Takes the output shape dimensions (num_out_dims values space ' ' separated) \n");
    printf("\t-pad_values: Takes the pad values (prod(pad_shape) values space ' ' separated) \n");
    printf("\t-permute_vec: Takes the permutation values of dimentions for transpose; space ' ' separated) \n");
    printf("\t-block_sizes: Takes the block sizes((num_inp_dims-2) values space ' ' separated) for batch_to_space_nd and space_to_batch_nd kernels \n");
    printf("\t-crop_or_pad_sizes: Takes the crop sizes for batch_to_space_nd or pad sizes for space_to_batch_nd (2*(num_inp_dims-2) values space ' ' separated) \n");
}

void parse_arguments(int argc, char** argv, test_config_t *p_cfg)
{
  int argidx;
  for (argidx=1;argidx<argc;argidx++)
  {
    if(strncmp((argv[argidx]), "-", 1) != 0)
    {
      //err_code = 0;
      printf("Invalid argument: %s\n",argv[argidx]);
      show_usage();
      exit(1);
    }
    ARGTYPE_INDICATE("--help", p_cfg->help);
    ARGTYPE_INDICATE("-help", p_cfg->help);
    ARGTYPE_INDICATE("-h", p_cfg->help);
    ARGTYPE_ONETIME_CONFIG("-inp_data_format",p_cfg->inp_data_format);
    ARGTYPE_ONETIME_CONFIG("-num_inputs", p_cfg->num_inputs);
    ARGTYPE_ONETIME_CONFIG("-num_outputs", p_cfg->num_outputs);
    ARGTYPE_ONETIME_CONFIG("-num_inp_dims", p_cfg->num_inp_dims);                  
    ARGTYPE_ONETIME_CONFIG("-num_pad_dims", p_cfg->num_pad_dims);                     
    ARGTYPE_ONETIME_CONFIG("-num_out_dims", p_cfg->num_out_dims);                   
    ARGTYPE_ONETIME_CONFIG("-pad_value", p_cfg->pad_value);     
    ARGTYPE_ONETIME_CONFIG("-input_batch",p_cfg->input_batch);
    ARGTYPE_ONETIME_CONFIG("-input_height",p_cfg->input_height);
    ARGTYPE_ONETIME_CONFIG("-input_width",p_cfg->input_width);
    ARGTYPE_ONETIME_CONFIG("-input_channels",p_cfg->input_channels);
    ARGTYPE_ONETIME_CONFIG("-block_size",p_cfg->block_size);
    ARGTYPE_ONETIME_CONFIG("-out_batch",p_cfg->out_batch);
    ARGTYPE_ONETIME_CONFIG("-out_height",p_cfg->out_height);
    ARGTYPE_ONETIME_CONFIG("-out_width",p_cfg->out_width);
    ARGTYPE_ONETIME_CONFIG("-out_channels",p_cfg->out_channels);
    ARGTYPE_ONETIME_CONFIG("-align_corners",p_cfg->align_corners);
    ARGTYPE_ONETIME_CONFIG("-half_pixel_centers",p_cfg->half_pixel_centers);
    ARGTYPE_ONETIME_CONFIG("-start_0",p_cfg->start_0);
    ARGTYPE_ONETIME_CONFIG("-stop_0",p_cfg->stop_0);
    ARGTYPE_ONETIME_CONFIG("-start_1",p_cfg->start_1);
    ARGTYPE_ONETIME_CONFIG("-stop_1",p_cfg->stop_1);
    ARGTYPE_ONETIME_CONFIG("-start_2",p_cfg->start_2);
    ARGTYPE_ONETIME_CONFIG("-stop_2",p_cfg->stop_2);
    ARGTYPE_ONETIME_CONFIG("-start_3",p_cfg->start_3);
    ARGTYPE_ONETIME_CONFIG("-stop_3",p_cfg->stop_3);
    ARGTYPE_ONETIME_CONFIG("-start_4",p_cfg->start_4);
    ARGTYPE_ONETIME_CONFIG("-stop_4",p_cfg->stop_4);
    ARGTYPE_ONETIME_CONFIG("-stride_0",p_cfg->stride_0);
    ARGTYPE_ONETIME_CONFIG("-stride_1",p_cfg->stride_1);
    ARGTYPE_ONETIME_CONFIG("-stride_2",p_cfg->stride_2);
    ARGTYPE_ONETIME_CONFIG("-stride_3",p_cfg->stride_3);
    ARGTYPE_ONETIME_CONFIG("-stride_4",p_cfg->stride_4);
    ARGTYPE_ONETIME_CONFIG("-axis",p_cfg->axis);
    ARGTYPE_ONETIME_CONFIG("-inp_precision",p_cfg->inp_precision);
    ARGTYPE_ONETIME_CONFIG("-out_precision",p_cfg->out_precision);
    ARGTYPE_STRING("-kernel_name",p_cfg->kernel_name, MAX_KERNEL_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-frames",p_cfg->frames);
    ARGTYPE_ONETIME_CONFIG("-write_file",p_cfg->write_file);
    ARGTYPE_STRING("-read_inp_file_name",p_cfg->read_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name",p_cfg->read_ref_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_inp_file_name",p_cfg->write_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_out_file_name",p_cfg->write_out_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify",p_cfg->verify);

    ARGTYPE_ONETIME_CONFIG_ARRAY("-concat_inps_shape", p_cfg->concat_inputs_shape, p_cfg->num_inputs * p_cfg->num_inp_dims, p_cfg->read_concat_inps_shape_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-split_v_outs_shape", p_cfg->split_v_outputs_shape, p_cfg->num_outputs * p_cfg->num_out_dims, p_cfg->read_split_v_outs_shape_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-inp_shape", p_cfg->input_shape, p_cfg->num_inp_dims, p_cfg->read_inp_shape_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-pad_shape", p_cfg->pad_shape, p_cfg->num_pad_dims, p_cfg->read_pad_shape_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-out_shape", p_cfg->output_shape, p_cfg->num_out_dims, p_cfg->read_out_shape_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-permute_vec", p_cfg->permute_vec, p_cfg->num_inp_dims, p_cfg->read_permute_vec_str);
    int i, num_pad_values = 1;
    for(i = 0; i < p_cfg->num_pad_dims; i++)
    {
      num_pad_values *= p_cfg->pad_shape[i];
    }
    ARGTYPE_ONETIME_CONFIG_ARRAY("-pad_values", p_cfg->pad_values, num_pad_values, p_cfg->read_pad_values_str);

    ARGTYPE_ONETIME_CONFIG_ARRAY("-block_sizes", p_cfg->block_sizes, p_cfg->num_inp_dims-2, p_cfg->read_block_sizes_str);
    ARGTYPE_ONETIME_CONFIG_ARRAY("-crop_or_pad_sizes", p_cfg->crop_or_pad_sizes, 2*(p_cfg->num_inp_dims-2), p_cfg->read_crop_or_pad_sizes_str);

    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}
#define STRIDED_SLICE_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
  XTPWR_PROFILER_START(0);\
  err = xa_nn_##KERNEL##_int##OPREC( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, \
        cfg.start_0, cfg.stop_0, cfg.start_1, cfg.stop_1, cfg.start_2, cfg.stop_2, cfg.start_3, cfg.stop_3, cfg.start_4, cfg.stop_4, \
        cfg.stride_0,cfg.stride_1,cfg.stride_2,cfg.stride_3,cfg.stride_4, \
        cfg.input_shape_extended[1],cfg.input_shape_extended[2],cfg.input_shape_extended[3],cfg.input_shape_extended[4]);\
        XTPWR_PROFILER_STOP(0);\
  }

#define DEPTH_SPACE_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.block_size, \
        cfg.out_height, cfg.out_width, cfg.out_channels, 0, 0);\
    XTPWR_PROFILER_STOP(0);\
  }

#define BATCH_SPACE_ND_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, cfg.output_shape, (WORD##IPREC *) p_inp->p, \
        cfg.input_shape, cfg.block_sizes, cfg.crop_or_pad_sizes, \
        cfg.num_out_dims, cfg.num_inp_dims);\
    XTPWR_PROFILER_STOP(0);\
  }

#define SPACE_TO_BATCH_ND_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, cfg.output_shape, (WORD##IPREC *) p_inp->p, \
        cfg.input_shape, cfg.block_sizes, cfg.crop_or_pad_sizes, \
        cfg.num_out_dims, cfg.num_inp_dims, cfg.pad_value);\
    XTPWR_PROFILER_STOP(0);\
  }

#define PAD_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, \
        (WORD32 *) p_out_shape,\
        (WORD##IPREC *) p_inp->p, \
        (WORD32 *) p_inp_shape,\
        (WORD32 *) p_pad_values, \
        (WORD32 *) p_pad_shape,\
        cfg.num_out_dims,\
        cfg.num_inp_dims,\
        cfg.num_pad_dims,\
        cfg.pad_value);\
    XTPWR_PROFILER_STOP(0);\
  }

#define TRANSPOSE_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, \
        (WORD32 *) p_out_shape,\
        (WORD##IPREC *) p_inp->p, \
        (WORD32 *) p_inp_shape,\
        (WORD32 *) p_permute_vec, \
        cfg.num_out_dims,\
        cfg.num_inp_dims);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONCAT_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    int itr_i = 0, accum_inp_size = 0; \
    for(itr_i = 0; itr_i < cfg.num_inputs && itr_i < MAX_NUM_INPS_CONCAT; itr_i++) \
    { \
      p_inp_shape_ar[itr_i] = &(cfg.concat_inputs_shape[itr_i * cfg.num_inp_dims]); \
      int itr_d = 0, cur_inp_size = 1; \
      for(itr_d = 0; itr_d < cfg.num_inp_dims && itr_i < MAX_DIMS_FOR_CONCAT; itr_d++) \
      { \
        if(p_inp_shape_ar[itr_i][itr_d] > 0) \
        { \
          cur_inp_size *= p_inp_shape_ar[itr_i][itr_d]; \
        } \
        else \
        { \
          cur_inp_size = 0; \
        } \
      } \
      p_inp_ar[itr_i] = (void *)(((WORD##IPREC *)p_inp->p) + accum_inp_size); \
      accum_inp_size += cur_inp_size; \
    } \
    XTPWR_PROFILER_START(0); \
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, \
        (WORD32 *) (cfg.output_shape), \
        (const WORD##IPREC **) p_inp_ar, \
        (const WORD32 *const *) p_inp_shape_ar, \
        cfg.num_out_dims, \
        cfg.num_inputs, \
        cfg.num_inp_dims, \
        cfg.axis); \
    XTPWR_PROFILER_STOP(0); \
  }
  
#define SPLIT_V_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) {\
    int itr_i = 0, accum_out_size = 0; \
    for(itr_i = 0; itr_i < cfg.num_outputs && itr_i < MAX_NUM_OUTS_SPLIT_V; itr_i++) \
    { \
      p_out_shape_ar[itr_i] = &(cfg.split_v_outputs_shape[itr_i * cfg.num_out_dims]); \
      int itr_d = 0, cur_out_size = 1; \
      for(itr_d = 0; itr_d < cfg.num_out_dims && itr_i < MAX_DIMS_FOR_SPLIT_V; itr_d++) \
      { \
        if(p_out_shape_ar[itr_i][itr_d] > 0) \
        { \
          cur_out_size *= p_out_shape_ar[itr_i][itr_d]; \
        } \
        else \
        { \
          cur_out_size = 0; \
        } \
      } \
      p_out_ar[itr_i] = (void *)(((WORD##IPREC *)p_out->p) + accum_out_size); \
      accum_out_size += cur_out_size; \
    } \
    XTPWR_PROFILER_START(0); \
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC **)p_out_ar, \
        (const WORD32 * const*) p_out_shape_ar, \
        (const WORD##IPREC *) p_inp->p, \
        (const WORD32 *) cfg.input_shape, \
        cfg.num_outputs, \
        cfg.num_out_dims, \
        cfg.num_inp_dims, \
        cfg.axis); \
    XTPWR_PROFILER_STOP(0); \
  }

#define RESIZE_BILINEAR_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) { \
    WORD32 height_scale_10, width_scale_10, height_bias, width_bias; \
    height_scale_10 = ((cfg.input_height << 10) + cfg.out_height / 2) / cfg.out_height; \
    width_scale_10 = ((cfg.input_width << 10)+ cfg.out_width / 2) / cfg.out_width; \
    if(cfg.align_corners && cfg.out_height > 1) \
      height_scale_10 = (((cfg.input_height - 1) << 10) + (cfg.out_height - 1) / 2) / (cfg.out_height - 1); \
    if(cfg.align_corners && cfg.out_width > 1) \
      width_scale_10 = (((cfg.input_width - 1) << 10) + (cfg.out_width - 1) / 2) / (cfg.out_width - 1); \
    height_bias = 0; \
    width_bias = 0; \
    if(cfg.half_pixel_centers) \
    { \
      height_bias = height_scale_10 / 2 - (1 << 9); \
      width_bias = width_scale_10 / 2 - (1 << 9); \
    } \
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, \
        (WORD##IPREC *) p_inp->p, \
        cfg.input_batch, \
        cfg.input_height, \
        cfg.input_width, \
        cfg.input_channels, \
        cfg.out_batch, \
        cfg.out_height, \
        cfg.out_width, \
        cfg.out_channels, \
        height_scale_10, \
        width_scale_10, \
        height_bias, \
        width_bias); \
    XTPWR_PROFILER_STOP(0);\
  }

#define RESIZE_NEAREST_NEIGHBOUR_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision) && (OPREC == p_out->precision)) { \
    float height_scale, width_scale, height_bias, width_bias; \
    height_scale = (cfg.align_corners && cfg.out_height > 1) ? (cfg.input_height - 1) / ((float)(cfg.out_height - 1)) : cfg.input_height / ((float)(cfg.out_height)); \
    width_scale = (cfg.align_corners && cfg.out_width > 1) ? (cfg.input_width - 1) / ((float)(cfg.out_width - 1)) : cfg.input_width / ((float)(cfg.out_width)); \
    height_bias = cfg.half_pixel_centers ? 0.5f : 0.0f; \
    width_bias = cfg.half_pixel_centers ? 0.5f : 0.0f; \
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC##_##OPREC ( \
        (WORD##OPREC *)p_out->p, \
        (WORD##IPREC *) p_inp->p, \
        cfg.input_batch, \
        cfg.input_height, \
        cfg.input_width, \
        cfg.input_channels, \
        cfg.out_batch, \
        cfg.out_height, \
        cfg.out_width, \
        cfg.out_channels, \
        height_scale, \
        width_scale, \
        height_bias, \
        width_bias, \
        cfg.align_corners); \
    XTPWR_PROFILER_STOP(0);\
  }

#if HIFI_VFPU
#define PROCESS_REORG \
    DEPTH_SPACE_KERNEL_FN(depth_to_space, 8, 8) \
    else DEPTH_SPACE_KERNEL_FN(space_to_depth, 8, 8) \
    else BATCH_SPACE_ND_KERNEL_FN(batch_to_space_nd, 8, 8) \
    else SPACE_TO_BATCH_ND_KERNEL_FN(space_to_batch_nd, 8, 8) \
    else PAD_KERNEL_FN(pad, 8, 8) \
    else PAD_KERNEL_FN(pad, 16, 16) \
    else PAD_KERNEL_FN(pad, 32, 32) \
    else STRIDED_SLICE_FN(strided_slice, 16, 16) \
    else STRIDED_SLICE_FN(strided_slice, 32, 32) \
    else STRIDED_SLICE_FN(strided_slice, 8, 8) \
    else TRANSPOSE_KERNEL_FN(transpose, 8, 8) \
    else TRANSPOSE_KERNEL_FN(transpose, 16, 16) \
    else CONCAT_KERNEL_FN(concat, 8, 8) \
    else SPLIT_V_KERNEL_FN(split_v, 8, 8) \
    else RESIZE_BILINEAR_FN(resize_bilinear, 8, 8) \
    else RESIZE_NEAREST_NEIGHBOUR_FN(resize_nearest_neighbour, 8, 8) \
    else {  printf("unsupported reorg operation\n"); return -1;}
#else
#define PROCESS_REORG \
    DEPTH_SPACE_KERNEL_FN(depth_to_space, 8, 8) \
    else DEPTH_SPACE_KERNEL_FN(space_to_depth, 8, 8) \
    else BATCH_SPACE_ND_KERNEL_FN(batch_to_space_nd, 8, 8) \
    else SPACE_TO_BATCH_ND_KERNEL_FN(space_to_batch_nd, 8, 8) \
    else PAD_KERNEL_FN(pad, 8, 8) \
    else PAD_KERNEL_FN(pad, 16, 16) \
    else PAD_KERNEL_FN(pad, 32, 32) \
    else STRIDED_SLICE_FN(strided_slice, 16, 16) \
    else STRIDED_SLICE_FN(strided_slice, 32, 32) \
    else STRIDED_SLICE_FN(strided_slice, 8, 8) \
    else TRANSPOSE_KERNEL_FN(transpose, 8, 8) \
    else TRANSPOSE_KERNEL_FN(transpose, 16, 16) \
    else CONCAT_KERNEL_FN(concat, 8, 8) \
    else SPLIT_V_KERNEL_FN(split_v, 8, 8) \
    else RESIZE_BILINEAR_FN(resize_bilinear, 8, 8) \
    else {  printf("unsupported reorg operation\n"); return -1;}
#endif

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  int inp_size, out_size/*, pad_values_size*/;
  int num_pts=0;
  void *p_inp_ar[MAX_NUM_INPS_CONCAT];
  int *p_inp_shape_ar[MAX_NUM_INPS_CONCAT];
  void *p_out_ar[MAX_NUM_OUTS_SPLIT_V];
  int *p_out_shape_ar[MAX_NUM_OUTS_SPLIT_V];

  test_config_t cfg;

  buf1D_t *p_inp;
  buf1D_t *p_out;
  buf1D_t *p_ref;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref;

  // permute_vec, pad_values and shape pointers for transpose and pad kernel
  WORD32 *p_inp_shape, *p_out_shape, *p_pad_shape, *p_pad_values, *p_permute_vec;

  if(default_config(&cfg))
  {
    return -1;
  }

  fprintf(stderr, "\n--------------------------------------------------------\n");
  fprintf(stderr, "%s library version %s\n", xa_nnlib_get_lib_name_string() , xa_nnlib_get_lib_version_string());
  fprintf(stderr, "API version: %s\n", xa_nnlib_get_lib_api_version_string());
  fprintf(stderr, "Cadence Design Systems, Inc. http://www.cadence.com\n");

  if(argc > 1)
  {
    printf("Parsing CMDLINE\n");
    parse_arguments(argc, argv, &cfg);
    if(1 == cfg.help)
    {
      show_usage();
      return 0;
    }
  }

  if(strcmp(cfg.kernel_name, "pad") == 0)
  {
    inp_size = 1; 
    out_size = 1;
    /* Calculating input , pad_values and output size from respective shapes for Pad ops */
//    pad_values_size = 1;
    int itr;
    for(itr = 0; itr < cfg.num_inp_dims; itr++)
    {
      inp_size *= cfg.input_shape[itr]; 
    }
    for(itr = 0; itr < cfg.num_pad_dims; itr++)
    {
//      pad_values_size *= cfg.pad_shape[itr]; 
    }
    for(itr = 0; itr < cfg.num_out_dims; itr++)
    {
      out_size *= cfg.output_shape[itr]; 
    }
  }
  else if(strcmp(cfg.kernel_name, "strided_slice") == 0)
  {
     int i,rev_itr;
     float out_dim_calc;
     rev_itr = cfg.num_inp_dims;
     if(cfg.num_inp_dims > 4)
     {
        printf("\nIncorrect Arguments. Number of Input Dimentions must be less than or equal to 4. Exiting\n");
        exit(1);
     }
     for(i = 4 ; i > 4 - cfg.num_inp_dims; i--)
     {
        cfg.input_shape_extended[i] = cfg.input_shape[rev_itr -1];
        rev_itr--;
     }
     
     out_dim_calc = (float)(cfg.stop_0 - cfg.start_0)/(float)cfg.stride_0;
     cfg.output_shape[0] = (int)ceil(out_dim_calc);
     out_dim_calc = (float)(cfg.stop_1 - cfg.start_1)/(float)cfg.stride_1;
     cfg.output_shape[1] = (int)ceil(out_dim_calc);
     out_dim_calc = (float)(cfg.stop_2 - cfg.start_2)/(float)cfg.stride_2;
     cfg.output_shape[2] = (int)ceil(out_dim_calc);
     out_dim_calc = (float)(cfg.stop_3 - cfg.start_3)/(float)cfg.stride_3;
     cfg.output_shape[3] = (int)ceil(out_dim_calc);
     out_dim_calc = (float)(cfg.stop_4 - cfg.start_4)/(float)cfg.stride_4;
     cfg.output_shape[4] = (int)ceil(out_dim_calc);
     inp_size = cfg.input_shape_extended[0]* cfg.input_shape_extended[1] *cfg.input_shape_extended[2] * cfg.input_shape_extended[3] * cfg.input_shape_extended[4] ;
     out_size = cfg.output_shape[0] * cfg.output_shape[1] * cfg.output_shape[2] * cfg.output_shape[3] * cfg.output_shape[4];
     if(cfg.output_shape[0] <= 0 || cfg.output_shape[1] <= 0 || cfg.output_shape[2] <= 0 || cfg.output_shape[3] <= 0 || cfg.output_shape[4] <= 0){
       printf("\nInvalid Arguments. While computing output shape, one or more output shape dimentions are coming out to be zero or negative. Exiting\n");    
       exit(1);
     }
  }
  else if(strcmp(cfg.kernel_name, "batch_to_space_nd") == 0
          || strcmp(cfg.kernel_name, "space_to_batch_nd") == 0)
  {
    inp_size = 1; 
    out_size = 1;
    int itr;
    for(itr = 0; itr < cfg.num_inp_dims; itr++)
    {
      inp_size *= cfg.input_shape[itr]; 
    }
    for(itr = 0; itr < cfg.num_out_dims; itr++)
    {
      out_size *= cfg.output_shape[itr]; 
    }
  }
  else if(strcmp(cfg.kernel_name, "transpose") == 0)
  {
    inp_size = 1; 
    out_size = 1;
    /* Calculating input, output size from respective shapes for transpose */
//    pad_values_size = 1;
    int itr;
    for(itr = 0; itr < cfg.num_inp_dims; itr++)
    {
      inp_size *= cfg.input_shape[itr]; 
    }
    for(itr = 0; itr < cfg.num_out_dims; itr++)
    {
      out_size *= cfg.output_shape[itr]; 
    }
  }
  else if(strcmp(cfg.kernel_name, "concat") == 0)
  {
    inp_size = 0; 
    out_size = 1;
    int itr;
    for(itr = 0; itr < cfg.num_inputs; itr++)
    {
      int itr_d, cur_inp_size = 1;
      for(itr_d = 0; itr_d < cfg.num_inp_dims; itr_d++)
      {
        if(cfg.concat_inputs_shape[itr*cfg.num_inp_dims + itr_d] > 0)
          cur_inp_size *= cfg.concat_inputs_shape[itr*cfg.num_inp_dims + itr_d];
        else
          cur_inp_size = 0;
      }
      inp_size += cur_inp_size;
    }
    for(itr = 0; itr < cfg.num_out_dims; itr++)
    {
      out_size *= cfg.output_shape[itr]; 
    }
  }
  else if(strcmp(cfg.kernel_name, "split_v") == 0)
  {
    inp_size = 1; 
    out_size = 0;
    int itr;
    for(itr = 0; itr < cfg.num_outputs; itr++)
    {
      int itr_d, cur_out_size = 1;
      for(itr_d = 0; itr_d < cfg.num_out_dims; itr_d++)
      {
        if(cfg.split_v_outputs_shape[itr*cfg.num_out_dims + itr_d] > 0)
          cur_out_size *= cfg.split_v_outputs_shape[itr*cfg.num_out_dims + itr_d];
        else
          cur_out_size = 0;
      }
      out_size += cur_out_size;
    }
    for(itr = 0; itr < cfg.num_inp_dims; itr++)
    {
      inp_size *= cfg.input_shape[itr]; 
    }
  }
  else if(strcmp(cfg.kernel_name, "resize_bilinear") == 0 || strcmp(cfg.kernel_name, "resize_nearest_neighbour") == 0)
  {
    inp_size = cfg.input_batch * cfg.input_height * cfg.input_width * cfg.input_channels;
    out_size = cfg.out_batch * cfg.out_height * cfg.out_width * cfg.out_channels;
  }
  else
  {
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    out_size = cfg.out_height * cfg.out_width * cfg.out_channels;
  }

  // Set profiler name
  if(cfg.kernel_name[0])
  {
    strcpy(profiler_name,cfg.kernel_name);
  }
  if(cfg.inp_precision == -1)
  {
    sprintf(profiler_params, "_f32");
    strcat(profiler_name, profiler_params);

    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else if(cfg.inp_precision == -3)
  {
    sprintf(profiler_params, "_asym8");
    strcat(profiler_name, profiler_params);
  }
  else if(cfg.inp_precision == -4)
  {
    sprintf(profiler_params, "_asym8s");
    strcat(profiler_name, profiler_params);
  }
  else
  {
    sprintf(profiler_params, "_%d",
        cfg.inp_precision);
    strcat(profiler_name, profiler_params);
  }

  // Set profiler parameters
  if(strcmp(cfg.kernel_name, "pad") == 0)
  {
    sprintf(profiler_params, "input_shape= %s pad_shape= %s output_shape= %s pad_values= %s\n", cfg.read_inp_shape_str, cfg.read_pad_shape_str, cfg.read_out_shape_str, cfg.read_pad_values_str);
  }
  else if(strcmp(cfg.kernel_name, "batch_to_space_nd") == 0)
  {
    sprintf(profiler_params, "input_shape= %s block_sizes= %s crop_sizes= %s output_shape= %s\n", cfg.read_inp_shape_str, cfg.read_block_sizes_str, cfg.read_crop_or_pad_sizes_str, cfg.read_out_shape_str);
  }
  else if(strcmp(cfg.kernel_name, "space_to_batch_nd") == 0)
  {
    sprintf(profiler_params, "input_shape= %s block_sizes= %s pad_sizes= %s output_shape= %s\n", cfg.read_inp_shape_str, cfg.read_block_sizes_str, cfg.read_crop_or_pad_sizes_str, cfg.read_out_shape_str);
  }
  else if(strcmp(cfg.kernel_name, "strided_slice") == 0)
  {
   sprintf(profiler_params,"start = [%d %d %d %d %d] stop = [%d %d %d %d %d] stride = [%d %d %d %d %d] input_shape=%s\n", cfg.start_0, cfg.start_1, cfg.start_2, cfg.start_3, cfg.start_4 , cfg.stop_0, cfg.stop_1, cfg.stop_2, cfg.stop_3, cfg.stop_4 , cfg.stride_0,cfg.stride_1,cfg.stride_2,cfg.stride_3,cfg.stride_4, cfg.read_inp_shape_str);
  }
  else if(strcmp(cfg.kernel_name, "transpose") == 0)
  {
    sprintf(profiler_params, "input_shape= %s output_shape= %s permute_vec= %s\n", cfg.read_inp_shape_str, cfg.read_out_shape_str, cfg.read_permute_vec_str);
  }
  else if(strcmp(cfg.kernel_name, "concat") == 0)
  {
    sprintf(profiler_params, "inputs_shape= %s output_shape= %s axis = %d\n", cfg.read_concat_inps_shape_str, cfg.read_out_shape_str, cfg.axis);
  }
  else if(strcmp(cfg.kernel_name, "split_v") == 0)
  {
    sprintf(profiler_params, "inputs_shape= %s output_shape= %s axis = %d\n", cfg.read_inp_shape_str, cfg.read_split_v_outs_shape_str, cfg.axis);
  }
  else if(strcmp(cfg.kernel_name, "resize_bilinear") == 0 || strcmp(cfg.kernel_name, "resize_nearest_neighbour") == 0)
  {
    sprintf(profiler_params, "input_batch=%d, input_height=%d, input_width=%d, input_channels=%d, out_batch=%d, out_height=%d, out_width=%d, out_channels=%d",
      cfg.input_batch, cfg.input_height, cfg.input_width, cfg.input_channels, cfg.out_batch, cfg.out_height, cfg.out_width, cfg.out_channels);
  }
  else
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, out_height=%d, out_width=%d, out_channels =%d",
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.out_height, cfg.out_width, cfg.out_channels);
  }

  // Open input file
  if(cfg.write_file)
  {
    /* If write_file (generate test vectors) is enabled, random data would be generated and
       used; the input data and output data generated would be written into files.
     */
    fptr_inp = file_open(pb_input_file_path, cfg.write_inp_file_name, "wb", XA_MAX_CMD_LINE_LENGTH);
  }
  else
  {
    /* Else, if input file is specified on command line, input data would be read from it, else
       input data would be read from the default file set in default_config().
     */
    fptr_inp = file_open(pb_input_file_path, cfg.read_inp_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Open output file
  fptr_out = file_open(pb_output_file_path, cfg.write_out_file_name, "wb", XA_MAX_CMD_LINE_LENGTH);

  // Open reference file if verify flag is enabled
  if(cfg.verify)
  {
    p_ref = create_buf1D(out_size, cfg.out_precision);
    fptr_ref = file_open(pb_ref_file_path, cfg.read_ref_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Allocate Memory
  p_inp = create_buf1D(inp_size, cfg.inp_precision);                              VALIDATE_PTR(p_inp);
  p_out = create_buf1D(out_size, cfg.out_precision);                              VALIDATE_PTR(p_out);

  if(strcmp(cfg.kernel_name, "pad") == 0)
  {
    p_inp_shape  = cfg.input_shape;
    p_pad_shape  = cfg.pad_shape;
    p_out_shape  = cfg.output_shape;
    p_pad_values = cfg.pad_values;
    num_pts      = out_size;
  }

  if(strcmp(cfg.kernel_name, "transpose") == 0)
  {
    p_inp_shape   = cfg.input_shape;
    p_out_shape   = cfg.output_shape;
    p_permute_vec = cfg.permute_vec;
    num_pts       = out_size;
  }

  if(!strcmp(cfg.kernel_name,"depth_to_space")
     || !strcmp(cfg.kernel_name,"space_to_depth")
     || !strcmp(cfg.kernel_name,"batch_to_space_nd")
     || !strcmp(cfg.kernel_name,"space_to_batch_nd")
     || !strcmp(cfg.kernel_name,"strided_slice")
     || !strcmp(cfg.kernel_name,"resize_bilinear")
     || !strcmp(cfg.kernel_name,"resize_nearest_neighbour")
     || !strcmp(cfg.kernel_name,"concat")
     || !strcmp(cfg.kernel_name,"split_v"))
  {
    num_pts = out_size;
  }

  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_pts, "cyc/point", 0);

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    load_reorg_input_data(cfg.write_file, fptr_inp, p_inp);

    // Call the cnn kernel_name specified on command line
    PROCESS_REORG;

    if(err)
    {
      fprintf(stdout, "\nKernel returned error (invalid parameters), Performance numbers may be incorrect!\n\n");
      pass_count += !err;
      break;
    }

    XTPWR_PROFILER_UPDATE(0);
    XTPWR_PROFILER_PRINT(0);

    // Write output into file
    write_buf1D_to_file(fptr_out, p_out);

    // If verify flag enabled, compare output against reference
    if(cfg.verify)
    {
      read_buf1D_from_file(fptr_ref, p_ref);
      pass_count += compare_buf1D(p_ref, p_out, cfg.verify, cfg.out_precision, 1);
    }
    else
    {
      pass_count += !err;
    }
  }

  XTPWR_PROFILER_CLOSE(0, (pass_count == cfg.frames), cfg.verify);

  fclose(fptr_inp);
  fclose(fptr_out);

  // Free all buffers
  free_buf1D(p_inp);
  free_buf1D(p_out);

  if(cfg.verify)
  {
    fclose(fptr_ref);
    free_buf1D(p_ref);
  }

  return 0;
}

int main (int argc, char *argv[])
{
    FILE *param_file_id;
    int err_code = 0;

    WORD8 curr_cmd[XA_MAX_ARGS * XA_MAX_CMD_LINE_LENGTH];
    WORD32 fargc, curpos;
    WORD32 processcmd = 0;

    char fargv[XA_MAX_ARGS][XA_MAX_CMD_LINE_LENGTH];

    char *pargv[XA_MAX_ARGS+1];

    if(argc == 1)
    {
        param_file_id = fopen(PARAMFILE, "r");
        if (param_file_id == NULL)
        {
            err_code = -1;
            printf("Error opening Parameter file for reading %s\n",PARAMFILE);
            exit(1);
        }

        /* Process one line at a time */
        while(fgets((char *)curr_cmd, XA_MAX_ARGS * XA_MAX_CMD_LINE_LENGTH, param_file_id))
        {
            curpos = 0;
            fargc = 0;
            /* if it is not a param_file command and if */
            /* CLP processing is not enabled */
            if(curr_cmd[0] != '@' && !processcmd)
            {   /* skip it */
                continue;
            }

            while(sscanf((const char *)curr_cmd + curpos, "%s", fargv[fargc]) != EOF)
            {
                if(fargv[0][0]=='/' && fargv[0][1]=='/')
                    break;
                if(strcmp(fargv[0], "@echo") == 0)
                    break;
                if(strcmp(fargv[fargc], "@New_line") == 0)
                {
                    fgets((char *)curr_cmd + curpos, XA_MAX_CMD_LINE_LENGTH, param_file_id);
                    continue;
                }
                curpos += strlen(fargv[fargc]);
                while(*(curr_cmd + curpos)==' ' || *(curr_cmd + curpos)=='\t')
                    curpos++;
                fargc++;
            }

            if(fargc < 1)   /* for blank lines etc. */
                continue;

            if(strcmp(fargv[0], "@Output_path") == 0)
            {
                if(fargc > 1) strcpy((char *)pb_output_file_path, fargv[1]);
                else strcpy((char *)pb_output_file_path, "");
                continue;
            }

            if(strcmp(fargv[0], "@Input_path") == 0)
            {
                if(fargc > 1) strcpy((char *)pb_input_file_path, fargv[1]);
                else strcpy((char *)pb_input_file_path, "");
                continue;
            }

            if(strcmp(fargv[0], "@Ref_path") == 0)
            {
                if(fargc > 1) strcpy((char *)pb_ref_file_path, fargv[1]);
                else strcpy((char *)pb_ref_file_path, "");
                continue;
            }

            if(strcmp(fargv[0], "@Start") == 0)
            {
                processcmd = 1;
                continue;
            }

            if(strcmp(fargv[0], "@Stop") == 0)
            {
                processcmd = 0;
                continue;
            }

            /* otherwise if this a normal command and its enabled for execution */
            if(processcmd)
            {
                int i;

                pargv[0] = argv[0];
                for(i = 0; i < fargc; i++)
                {
                    fprintf(stdout, "%s ", fargv[i]);
                    pargv[i+1] = fargv[i];
                }

                fprintf(stdout, "\n");

                if(err_code == 0)
                    xa_nn_main_process(fargc+1, pargv);

            }
        }
        fclose(param_file_id);
    }
    else
    {
        int i;

        for(i = 1; i < argc; i++)
        {
            fprintf(stdout, "%s ", argv[i]);

        }

        fprintf(stdout, "\n");

        if(err_code == 0)
            xa_nn_main_process(argc, argv);

    }

    return 0;

}


