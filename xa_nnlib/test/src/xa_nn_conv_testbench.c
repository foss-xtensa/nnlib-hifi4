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
#include <xtensa/config/core-isa.h>
#include "xa_type_def.h"
#include "nnlib/xa_nnlib_api.h"
#include "xt_manage_buffers.h"
#include "cmdline_parser.h"
#include "file_io.h"
#include "xa_nnlib_standards.h"
#include "stdbool.h"

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_KERNEL_NAME_LENGTH 40
#define SCRATCH_SIZE_BYTES         2048*8 //TBD: if not reqd, remove

#define XA_MAX_CMD_LINE_LENGTH 200
#define XA_MAX_ARGS 100
#define PARAMFILE "paramfilesimple_conv.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{

  int help;
  int input_height;
  int input_width;
  int input_channels;
  int kernel_height;
  int kernel_width;
  int kernel_channels;
  int channels_multiplier;
  int out_channels;
  int x_stride;
  int y_stride;
  int x_padding;
  int y_padding;
  int out_height;
  int out_width;
  int bias_shift;
  int acc_shift;
  int inp_data_format;
  int out_data_format;
  int inp_precision;
  int kernel_precision;
  int out_precision;
  int bias_precision;
  int input_zero_bias;
  int kernel_zero_bias;
  int out_multiplier;
  int out_shift;
  int *p_out_multiplier;
  int *p_out_shift;
  int out_zero_bias;
  char kernel_name[MAX_KERNEL_NAME_LENGTH];
  int frames;
  int write_file;
  char read_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_ref_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_out_file_name[XA_MAX_CMD_LINE_LENGTH];
  int verify;
  int dilation_height;
  int dilation_width;
  int pointwise_profile_only;
  int groups;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  { 

    p_cfg->help     = 0;
    p_cfg->input_height = 16;
    p_cfg->input_width = 16;
    p_cfg->kernel_channels = 4;
    p_cfg->input_channels = 4;
    p_cfg->kernel_height = 3;
    p_cfg->kernel_width = 3;
    p_cfg->channels_multiplier = 1;
    p_cfg->out_channels = 4;
    p_cfg->x_stride = 2;
    p_cfg->y_stride = 2;
    p_cfg->x_padding = 2;
    p_cfg->y_padding = 2;
    p_cfg->out_height = 16;
    p_cfg->out_width = 16;
    p_cfg->bias_shift = 7;
    p_cfg->acc_shift = -7;
    p_cfg->inp_data_format = 1;
    p_cfg->out_data_format = 0;
    p_cfg->inp_precision = 16;
    p_cfg->kernel_precision = 8;
    p_cfg->out_precision = 16;
    p_cfg->bias_precision = 16;
    p_cfg->input_zero_bias = -127;
    p_cfg->kernel_zero_bias = -127;
    p_cfg->out_multiplier = 0x40000000;
    p_cfg->out_shift = -8;
    p_cfg->p_out_multiplier = NULL;
    p_cfg->p_out_shift = NULL;
    p_cfg->out_zero_bias = 128;
    strcpy(p_cfg->kernel_name, "conv2d_std");
    p_cfg->frames   = 2;  
    p_cfg->write_file = 0;  
    p_cfg->read_inp_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;
    p_cfg->dilation_height = 1;
    p_cfg->dilation_width = 1;
    p_cfg->pointwise_profile_only = 0;
    p_cfg->groups = 1;
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
    printf("\t-input_height: input height; Default=16\n");
    printf("\t-input_width: input width; Default=16\n");
    printf("\t-input_channels: input channels; Default=4\n");
    printf("\t-kernel_height: kernel height; Default=3\n");
    printf("\t-kernel_width: kernel width; Default=3\n");
    printf("\t-kernel_channels: kernel channels; Default=4\n");
    printf("\t-out_channels: output channels; Default=4\n");
    printf("\t-channels_multiplier: channel multiplier; Default=1\n");
    printf("\t-x_stride: stride in width dimension; Default=2\n");
    printf("\t-y_stride: stride in height dimension; Default=2\n");
    printf("\t-x_padding: left padding in width dimension; Default=2\n");
    printf("\t-y_padding: top padding in height dimension; Default=2\n");
    printf("\t-dilation_height: dilation in height dimension; Default=1\n");
    printf("\t-dilation_width: dilation in width dimension; Default=1\n");
    printf("\t-out_height: output height; Default=16\n");
    printf("\t-out_width: output width; Default=16\n");
    printf("\t-bias_shift: bias left shift; Default=7\n");
    printf("\t-groups: number of groups; Default=1\n");
    printf("\t-acc_shift: accumulator left shift; Default=-7\n");
    printf("\t-inp_data_format: Input data format, 0 (DWH), 1 (WHD); Default=1 (WHD), ignored for conv2d_std and conv1d_std kernels \n");
    printf("\t-out_data_format: Output data format, 0 (DWH), 1 (WHD); Default=0 (DWH)\n");
    printf("\t-inp_precision: 8, 16, -1(single prec float), -2(half prec float), -3(Asymmetric 8-bit unsigned), -4(Asymmetric 8-bit signed), -8(Symmetric 16-bit signed); Default=16\n");
    printf("\t-kernel_precision: 8, 16, -1(single prec float), -2(half prec float), -3(Asymmetric 8-bit), -5(Symmetric 8-bit signed), -12(Symmetric 4-bit signed); Default=8\n");
    printf("\t-out_precision: 8, 16, -1(single prec float), -2(half prec float), -3(Asymmetric 8-bit), -4(Asymmetric 8-bit signed), -8(Symmetric 16-bit signed); Default=16\n");
    printf("\t-bias_precision: 8, 16, 32, 64, -1(single prec float) -2(half prec float); Default=16\n");
    printf("\t-input_zero_bias: input zero zero bias for quantized 8-bit, -255 to 0 (for Asymmetric 8-bit unsigned), -127 to 128 (for Asymmetric 8-bit signed), ignored for symmetric 16-bit signed; Default=-127\n");
    printf("\t-kernel_zero_bias: kernel zero zero_bias for quantized 8-bit, -255 to 0 (for Asymmetric 8-bit unsigned), ignored for symmetric 8-bit signed ; Default=-127\n");
    printf("\t-out_multiplier : Output multiplier in Q31 format for quantized 8-bit, 0x0 to 0x7fffffff; Default=0x40000000\n");
    printf("\t-out_shift : Output shift for quantized 8-bit(asym8u and asym8s), 31 to -31; Default=-8\n");
    printf("\t-out_zero_bias : Output zero bias for quantized 8-bit, 0 to 255 for asym8u, -128 to 127 for asym8s, ignored for symmetric 16-bit signed ; Default=128\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: conv2d_std, dilated_conv2d_std, conv2d_depth, dilated_conv2d_depth, conv2d_point, conv1d_std, transpose_conv , conv2d; Default="" : conv2d_std\n");
    printf("\t-pointwise_profile_only: Applicable only when kernel_name is conv2d_depth, 0 (print conv2d depthwise and pointwise profile info), 1(print only conv2d pointwise profile info); Default=0\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - input, kernel, bias, (pointwise kernel, pointwise bias for depth separable)) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - input, kernel, bias, (pointwise kernel, pointwise bias for depth separable)) \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
}

void parse_arguments(int argc, char** argv, test_config_t *p_cfg)
{
  int argidx;
  for (argidx=1;argidx<argc;argidx++)
  {
    if(strncmp((argv[argidx]), "-", 1) != 0)
    {
      //err_code = 0;
      printf("Invalid argument: %s at index %d\n",argv[argidx], argidx);
      show_usage();
      exit(1);
    }
    ARGTYPE_INDICATE("--help", p_cfg->help);
    ARGTYPE_INDICATE("-help", p_cfg->help);
    ARGTYPE_INDICATE("-h", p_cfg->help);
    ARGTYPE_ONETIME_CONFIG("-input_height",p_cfg->input_height);
    ARGTYPE_ONETIME_CONFIG("-input_width",p_cfg->input_width);
    ARGTYPE_ONETIME_CONFIG("-input_channels",p_cfg->input_channels);
    ARGTYPE_ONETIME_CONFIG("-kernel_channels",p_cfg->kernel_channels);
    ARGTYPE_ONETIME_CONFIG("-kernel_height",p_cfg->kernel_height);
    ARGTYPE_ONETIME_CONFIG("-kernel_width",p_cfg->kernel_width);
    ARGTYPE_ONETIME_CONFIG("-out_channels",p_cfg->out_channels);
    ARGTYPE_ONETIME_CONFIG("-groups",p_cfg->groups);
    ARGTYPE_ONETIME_CONFIG("-channels_multiplier",p_cfg->channels_multiplier);
    ARGTYPE_ONETIME_CONFIG("-x_stride",p_cfg->x_stride);
    ARGTYPE_ONETIME_CONFIG("-y_stride",p_cfg->y_stride);
    ARGTYPE_ONETIME_CONFIG("-x_padding",p_cfg->x_padding);
    ARGTYPE_ONETIME_CONFIG("-y_padding",p_cfg->y_padding);
    ARGTYPE_ONETIME_CONFIG("-out_height",p_cfg->out_height);
    ARGTYPE_ONETIME_CONFIG("-out_width",p_cfg->out_width);
    ARGTYPE_ONETIME_CONFIG("-bias_shift",p_cfg->bias_shift);
    ARGTYPE_ONETIME_CONFIG("-acc_shift",p_cfg->acc_shift);
    ARGTYPE_ONETIME_CONFIG("-inp_data_format",p_cfg->inp_data_format);
    ARGTYPE_ONETIME_CONFIG("-out_data_format",p_cfg->out_data_format);
    ARGTYPE_ONETIME_CONFIG("-inp_precision",p_cfg->inp_precision);
    ARGTYPE_ONETIME_CONFIG("-kernel_precision",p_cfg->kernel_precision);
    ARGTYPE_ONETIME_CONFIG("-out_precision",p_cfg->out_precision);
    ARGTYPE_ONETIME_CONFIG("-bias_precision",p_cfg->bias_precision);
    ARGTYPE_ONETIME_CONFIG("-input_zero_bias",p_cfg->input_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-kernel_zero_bias",p_cfg->kernel_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-out_multiplier",p_cfg->out_multiplier);
    ARGTYPE_ONETIME_CONFIG("-out_shift",p_cfg->out_shift);
    ARGTYPE_ONETIME_CONFIG("-out_zero_bias",p_cfg->out_zero_bias);
    ARGTYPE_STRING("-kernel_name",p_cfg->kernel_name, MAX_KERNEL_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-frames",p_cfg->frames);
    ARGTYPE_ONETIME_CONFIG("-write_file",p_cfg->write_file);
    ARGTYPE_STRING("-read_inp_file_name",p_cfg->read_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name",p_cfg->read_ref_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_inp_file_name",p_cfg->write_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_out_file_name",p_cfg->write_out_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify",p_cfg->verify);
    ARGTYPE_ONETIME_CONFIG("-dilation_height",p_cfg->dilation_height);
    ARGTYPE_ONETIME_CONFIG("-dilation_width",p_cfg->dilation_width);
    ARGTYPE_ONETIME_CONFIG("-pointwise_profile_only",p_cfg->pointwise_profile_only);

    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}



#define CONV_KERNEL_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##KPREC##x##IPREC ( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, (WORD##KPREC *) p_kernel->p, (WORD##BPREC *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.bias_shift, cfg.acc_shift, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_ASYM8_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_asym8xasym8 ( \
        (UWORD8 *)p_out->p, (UWORD8 *) p_inp->p, (UWORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.kernel_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_SYM8S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym8sxasym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_SYM4S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym4sxasym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_UN_KERNEL_SYM8S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym8sxasym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width,cfg.kernel_channels,cfg.dilation_height,cfg.dilation_width,cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias,cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_UN_KERNEL_SYM8SXSYM16S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym8sxsym16s ( \
        (WORD16 *)p_out->p, (WORD16 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD64 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width,cfg.kernel_channels,cfg.dilation_height,cfg.dilation_width,cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        0, cfg.p_out_multiplier, cfg.p_out_shift, 0, \
        cfg.out_data_format,p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_SYM8SXSYM16S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym8sxsym16s ( \
        (WORD16 *)p_out->p, (WORD16 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD64 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        0, cfg.p_out_multiplier, cfg.p_out_shift, 0, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define TRANSPOSE_CONV_KERNEL_SYM8SXASYM8S_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_sym8sxasym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.input_channels, \
        cfg.out_channels, cfg.input_height, cfg.input_width, cfg.kernel_height, cfg.kernel_width, \
        cfg.out_height, cfg.out_width, num_elements, cfg.input_zero_bias, cfg.out_zero_bias, cfg.p_out_shift, cfg.p_out_multiplier, \
        p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define TRANSPOSE_CONV_KERNEL_SYM8SXSYM16S_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_sym8sxsym16s ( \
        (WORD16 *)p_out->p, (WORD16 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD64 *)p_bias->p, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.input_channels, \
        cfg.out_channels, cfg.input_height, cfg.input_width, cfg.kernel_height, cfg.kernel_width, \
        cfg.out_height, cfg.out_width, num_elements, cfg.p_out_shift, cfg.p_out_multiplier, \
        p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define TRANSPOSE_CONV_KERNEL_F32XF32_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, (FLOAT32 *) p_kernel->p, (FLOAT32 *)p_bias->p, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.input_channels, \
        cfg.out_channels, cfg.input_height, cfg.input_width, cfg.kernel_height, cfg.kernel_width, \
        cfg.out_height, cfg.out_width, num_elements, \
        p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_DILATIONAL_KERNEL_SYM8S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_per_chan_sym8sxasym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, (WORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch, cfg.dilation_height, cfg.dilation_width);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV1D_KERNEL_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##KPREC##x##IPREC ( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, (WORD##KPREC *) p_kernel->p, (WORD##BPREC *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, \
        cfg.y_stride, cfg.y_padding, cfg.out_height, \
        cfg.bias_shift, cfg.acc_shift, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV1D_KERNEL_ASYM8_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_asym8xasym8 ( \
        (UWORD8 *)p_out->p, (UWORD8 *) p_inp->p, (UWORD8 *) p_kernel->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, \
        cfg.y_stride, cfg.y_padding, cfg.out_height, \
        cfg.input_zero_bias, cfg.kernel_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV1D_KERNEL_F_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, (FLOAT32 *) p_kernel->p, (FLOAT32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, \
        cfg.y_stride, cfg.y_padding, cfg.out_height, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_F_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, (FLOAT32 *) p_kernel->p, (FLOAT32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_KERNEL_F16_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f16 ( \
        (WORD16 *)p_out->p, (WORD16 *) p_inp->p, (WORD16 *) p_kernel->p, (WORD16 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }


#define CONV_DS_KERNEL_F_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_depthwise_f32 ( \
        (FLOAT32 *)p_dw_out->p, (FLOAT32 *) p_kernel->p, (FLOAT32 *) p_inp->p, (FLOAT32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_f32 ( \
            (FLOAT32 *)p_out->p, (FLOAT32 *) p_kernel_point->p, (FLOAT32 *) p_dw_out->p, (FLOAT32 *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }

#define CONV_DS_KERNEL_F16_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_depthwise_f16 ( \
        (WORD16 *)p_dw_out->p, (WORD16 *) p_kernel->p, (WORD16 *) p_inp->p, (WORD16 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_f16 ( \
            (WORD16 *)p_out->p, (WORD16 *) p_kernel_point->p, (WORD16 *) p_dw_out->p, (WORD16 *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }


#define CONV_DS_KERNEL_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_depthwise_##KPREC##x##IPREC ( \
        (WORD##OPREC *) p_dw_out->p, (WORD##KPREC *) p_kernel->p, (WORD##IPREC *) p_inp->p, (WORD##BPREC *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.acc_shift, cfg.bias_shift, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_##KPREC##x##IPREC ( \
            (WORD##OPREC *) p_out->p, (WORD##KPREC *) p_kernel_point->p, (WORD##IPREC *) p_dw_out->p, (WORD##BPREC *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, \
            cfg.acc_shift, cfg.bias_shift, \
            cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }

#define CONV_DS_KERNEL_ASYM8_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_depthwise_asym8xasym8 ( \
        (UWORD8 *) p_dw_out->p, (UWORD8 *) p_kernel->p, (UWORD8 *) p_inp->p, (WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.kernel_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_asym8xasym8 ( \
            (UWORD8 *) p_out->p, (UWORD8 *) p_kernel_point->p, (UWORD8 *) p_dw_out->p, (WORD32 *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, \
            cfg.input_zero_bias, cfg.kernel_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, \
            cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }

#define CONV_DS_KERNEL_SYM8_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s ( \
        (WORD8 *) p_dw_out->p, (const WORD8 *) p_kernel->p, (const WORD8 *) p_inp->p, (const WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_per_chan_sym8sxasym8s ( \
            (WORD8 *) p_out->p, (WORD8 *) p_kernel_point->p, (WORD8 *) p_dw_out->p, (WORD32 *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, \
            cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
            cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }

#define CONV_DS_KERNEL_SYM8SXSYM16S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0); \
    err = xa_nn_conv2d_depthwise_per_chan_sym8sxsym16s ( \
        (WORD16 *) p_dw_out->p, (const WORD8 *) p_kernel->p, (const WORD16 *) p_inp->p, (const WORD64 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, 0, \
        cfg.p_out_multiplier, cfg.p_out_shift, 0, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch); \
    XTPWR_PROFILER_STOP(0);\
    if(!cfg.pointwise_profile_only) { \
        XTPWR_PROFILER_UPDATE(0); \
        XTPWR_PROFILER_PRINT(0); \
    } \
    if(!err) { \
        XTPWR_PROFILER_START(1);\
        err = xa_nn_conv2d_pointwise_per_chan_sym8sxsym16s ( \
            (WORD16 *) p_out->p, (WORD8 *) p_kernel_point->p, (WORD16 *) p_dw_out->p, (WORD64 *)p_bias_point->p, \
            cfg.out_height, cfg.out_width, cfg.input_channels*cfg.channels_multiplier, cfg.out_channels, 0, \
            cfg.p_out_multiplier, cfg.p_out_shift, 0, \
            cfg.out_data_format); \
        XTPWR_PROFILER_STOP(1);\
        XTPWR_PROFILER_UPDATE(1); \
        XTPWR_PROFILER_PRINT(1); \
    } \
  }

#define DILATED_CONV_DEPTH_KERNEL_F_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_dilated_conv2d_depthwise_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_kernel->p, (FLOAT32 *) p_inp->p, (FLOAT32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, cfg.dilation_height, cfg.dilation_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define CONV_PT_KERNEL_SYM8SXSYM16S_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel_point->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias_point->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_conv2d_pointwise_per_chan_sym8sxsym16s ( \
        (WORD16 *) p_out->p, (WORD8 *) p_kernel_point->p, (WORD16 *) p_inp->p, (WORD64 *)p_bias_point->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.out_channels, \
        0, cfg.p_out_multiplier, cfg.p_out_shift, 0, \
        cfg.out_data_format); \
    XTPWR_PROFILER_STOP(0);\
  }

#define DILATED_CONV_DS_KERNEL_SYM8_PC_FN(KERNEL, KPREC, IPREC, OPREC, BPREC) \
  (!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == p_kernel->precision) && (IPREC == p_inp->precision) && (OPREC == p_out->precision) && (BPREC == p_bias->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_dilated_conv2d_depthwise_per_chan_sym8sxasym8s ( \
        (WORD8 *) p_out->p, (const WORD8 *) p_kernel->p, (const WORD8 *) p_inp->p, (const WORD32 *)p_bias->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, cfg.dilation_height, cfg.dilation_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.input_zero_bias, cfg.p_out_multiplier, cfg.p_out_shift, cfg.out_zero_bias, \
        cfg.inp_data_format, 0 /* out_data_format always DWH*/, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
    } \

#if HIFI_VFPU
#if HIFI_HP_VFPU && hifi5
#define PROCESS_CONV \
    if CONV_KERNEL_FN(conv2d_std, 8, 16, 16, 16) \
    else if CONV_KERNEL_FN(conv2d_std, 8, 8, 8, 8) \
    else if CONV_KERNEL_FN(conv2d_std, 16, 16, 16, 16) \
    else if CONV_KERNEL_ASYM8_FN(conv2d_std, -3, -3, -3, 32) \
    else if CONV_KERNEL_SYM8S_PC_FN(conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_SYM4S_PC_FN(conv2d_std,-12,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8S_PC_FN(conv2d,-5,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8SXSYM16S_PC_FN(conv2d,-5,-8,-8, 64) \
    else if CONV_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_std,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXASYM8S_FN(transpose_conv,-5,-4,-4, 32) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXSYM16S_FN(transpose_conv,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_F32XF32_FN(transpose_conv,-1,-1,-1,-1) \
    else if CONV_DILATIONAL_KERNEL_SYM8S_PC_FN(dilated_conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_F_FN(conv2d_std, -1, -1, -1, -1) \
    else if CONV_KERNEL_F16_FN(conv2d_std, -2, -2, -2, -2) \
    else if CONV_DS_KERNEL_F_FN(conv2d_depth, -1, -1, -1, -1) \
    else if CONV_DS_KERNEL_F16_FN(conv2d_depth, -2, -2, -2, -2) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,16,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,8,8,8) \
    else if CONV_DS_KERNEL_ASYM8_FN(conv2d_depth,-3,-3,-3,32) \
    else if CONV_DS_KERNEL_SYM8_PC_FN(conv2d_depth,-5,-4,-4,32) \
    else if DILATED_CONV_DS_KERNEL_SYM8_PC_FN(dilated_conv2d_depth, -5, -4, -4, 32) \
    else if DILATED_CONV_DEPTH_KERNEL_F_FN(dilated_conv2d_depth, -1, -1, -1, -1) \
    else if CONV_PT_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_point,-5,-8,-8,64) \
    else if CONV_DS_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_depth,-5,-8,-8,64) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 16, 16, 16) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 8, 8, 8) \
    else if CONV1D_KERNEL_FN(conv1d_std, 16, 16, 16, 16) \
    else if CONV1D_KERNEL_ASYM8_FN(conv1d_std, -3, -3, -3, 32) \
    else if CONV1D_KERNEL_F_FN(conv1d_std, -1, -1, -1, -1) \
    else {printf("[Error] [%s] convolution is not supported\n", cfg.kernel_name); return -1;}
#else /* HIFI_HP_VFPU  && hifi5 */
#define PROCESS_CONV \
    if CONV_KERNEL_FN(conv2d_std, 8, 16, 16, 16) \
    else if CONV_KERNEL_FN(conv2d_std, 8, 8, 8, 8) \
    else if CONV_KERNEL_FN(conv2d_std, 16, 16, 16, 16) \
    else if CONV_KERNEL_ASYM8_FN(conv2d_std, -3, -3, -3, 32) \
    else if CONV_KERNEL_SYM8S_PC_FN(conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_SYM4S_PC_FN(conv2d_std,-12,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8S_PC_FN(conv2d,-5,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8SXSYM16S_PC_FN(conv2d,-5,-8,-8, 64) \
    else if CONV_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_std,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXASYM8S_FN(transpose_conv,-5,-4,-4, 32) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXSYM16S_FN(transpose_conv,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_F32XF32_FN(transpose_conv,-1,-1,-1,-1) \
    else if CONV_DILATIONAL_KERNEL_SYM8S_PC_FN(dilated_conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_F_FN(conv2d_std, -1, -1, -1, -1) \
    else if CONV_DS_KERNEL_F_FN(conv2d_depth, -1, -1, -1, -1) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,16,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,8,8,8) \
    else if CONV_DS_KERNEL_ASYM8_FN(conv2d_depth,-3,-3,-3,32) \
    else if CONV_DS_KERNEL_SYM8_PC_FN(conv2d_depth,-5,-4,-4,32) \
    else if DILATED_CONV_DS_KERNEL_SYM8_PC_FN(dilated_conv2d_depth, -5, -4, -4, 32) \
    else if DILATED_CONV_DEPTH_KERNEL_F_FN(dilated_conv2d_depth, -1, -1, -1, -1) \
    else if CONV_PT_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_point,-5,-8,-8,64) \
    else if CONV_DS_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_depth,-5,-8,-8,64) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 16, 16, 16) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 8, 8, 8) \
    else if CONV1D_KERNEL_FN(conv1d_std, 16, 16, 16, 16) \
    else if CONV1D_KERNEL_ASYM8_FN(conv1d_std, -3, -3, -3, 32) \
    else if CONV1D_KERNEL_F_FN(conv1d_std, -1, -1, -1, -1) \
    else {printf("[Error] [%s] convolution is not supported\n", cfg.kernel_name); return -1;}
#endif /* HIFI_HP_VFPU  && hifi5 */
#else /* HIFI_VFPU */
#if HIFI_HP_VFPU && hifi5
#define PROCESS_CONV \
    if CONV_KERNEL_FN(conv2d_std, 8, 16, 16, 16) \
    else if CONV_KERNEL_FN(conv2d_std, 8, 8, 8, 8) \
    else if CONV_KERNEL_FN(conv2d_std, 16, 16, 16, 16) \
    else if CONV_KERNEL_ASYM8_FN(conv2d_std, -3, -3, -3, 32) \
    else if CONV_KERNEL_SYM8S_PC_FN(conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_SYM4S_PC_FN(conv2d_std,-12,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8S_PC_FN(conv2d,-5,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8SXSYM16S_PC_FN(conv2d,-5,-8,-8, 64) \
    else if CONV_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_std,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXASYM8S_FN(transpose_conv,-5,-4,-4, 32) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXSYM16S_FN(transpose_conv,-5,-8,-8, 64) \
    else if CONV_DILATIONAL_KERNEL_SYM8S_PC_FN(dilated_conv2d_std,-5,-4,-4, 32) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,16,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,8,8,8) \
    else if CONV_DS_KERNEL_ASYM8_FN(conv2d_depth,-3,-3,-3,32) \
    else if CONV_DS_KERNEL_SYM8_PC_FN(conv2d_depth,-5,-4,-4,32) \
    else if CONV_KERNEL_F16_FN(conv2d_std, -2, -2, -2, -2) \
    else if CONV_DS_KERNEL_F16_FN(conv2d_depth, -2, -2, -2, -2) \
    else if DILATED_CONV_DS_KERNEL_SYM8_PC_FN(dilated_conv2d_depth, -5, -4, -4, 32) \
    else if CONV_PT_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_point,-5,-8,-8,64) \
    else if CONV_DS_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_depth,-5,-8,-8,64) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 16, 16, 16) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 8, 8, 8) \
    else if CONV1D_KERNEL_FN(conv1d_std, 16, 16, 16, 16) \
    else if CONV1D_KERNEL_ASYM8_FN(conv1d_std, -3, -3, -3, 32) \
    else {printf("[Error] [%s] convolution is not supported\n", cfg.kernel_name); return -1;}
#else /* HIFI_HP_VFPU && hifi5 */
#define PROCESS_CONV \
    if CONV_KERNEL_FN(conv2d_std, 8, 16, 16, 16) \
    else if CONV_KERNEL_FN(conv2d_std, 8, 8, 8, 8) \
    else if CONV_KERNEL_FN(conv2d_std, 16, 16, 16, 16) \
    else if CONV_KERNEL_ASYM8_FN(conv2d_std, -3, -3, -3, 32) \
    else if CONV_KERNEL_SYM8S_PC_FN(conv2d_std,-5,-4,-4, 32) \
    else if CONV_KERNEL_SYM4S_PC_FN(conv2d_std,-12,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8S_PC_FN(conv2d,-5,-4,-4, 32) \
    else if CONV_UN_KERNEL_SYM8SXSYM16S_PC_FN(conv2d,-5,-8,-8, 64) \
    else if CONV_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_std,-5,-8,-8, 64) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXASYM8S_FN(transpose_conv,-5,-4,-4, 32) \
    else if TRANSPOSE_CONV_KERNEL_SYM8SXSYM16S_FN(transpose_conv,-5,-8,-8, 64) \
    else if CONV_DILATIONAL_KERNEL_SYM8S_PC_FN(dilated_conv2d_std,-5,-4,-4, 32) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,16,16,16,16) \
    else if CONV_DS_KERNEL_FN(conv2d_depth,8,8,8,8) \
    else if CONV_DS_KERNEL_ASYM8_FN(conv2d_depth,-3,-3,-3,32) \
    else if CONV_DS_KERNEL_SYM8_PC_FN(conv2d_depth,-5,-4,-4,32) \
    else if DILATED_CONV_DS_KERNEL_SYM8_PC_FN(dilated_conv2d_depth, -5, -4, -4, 32) \
    else if CONV_PT_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_point,-5,-8,-8,64) \
    else if CONV_DS_KERNEL_SYM8SXSYM16S_PC_FN(conv2d_depth,-5,-8,-8,64) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 16, 16, 16) \
    else if CONV1D_KERNEL_FN(conv1d_std, 8, 8, 8, 8) \
    else if CONV1D_KERNEL_FN(conv1d_std, 16, 16, 16, 16) \
    else if CONV1D_KERNEL_ASYM8_FN(conv1d_std, -3, -3, -3, 32) \
    else {printf("[Error] [%s] convolution is not supported\n", cfg.kernel_name); return -1;}
#endif /* HIFI_HP_VFPU && hifi5 */    
#endif /* HIFI_VFPU */

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name_0[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_name_1[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 
  void *p_scratch;
  int inp_size=0, kernel_size, out_size;
  int kernel_size_pad, input_channels_pad,kernel_channels_pad;
  int kernel_channels;
  int input_channelsXwidth_pad;
  int kernel_point_size, dw_out_size;
  int bias_size, bias_point_size;
  int num_elements;

  test_config_t cfg;

  buf1D_t *p_inp;
  buf2D_t *p_kernel;
  buf1D_t *p_kernel_point;
  buf1D_t *p_bias;
  buf1D_t *p_bias_point;
  buf1D_t *p_dw_out;
  buf1D_t *p_out;
  buf1D_t *p_ref;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref;

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

  const char *kernel_names_supported[] = {"conv2d_std", "dilated_conv2d_std", "conv2d_depth", "dilated_conv2d_depth", "conv2d_point", "conv1d_std", "transpose_conv","conv2d"};
  int num_kernel_names = 8;
  int ker_name_itr = 0;
  bool is_ker_name_supported = 0;
  for(ker_name_itr = 0; ker_name_itr < num_kernel_names; ker_name_itr++)
  {
    if(!strcmp(cfg.kernel_name,kernel_names_supported[ker_name_itr]))
    {
      is_ker_name_supported = 1;
      break;
    }
  }
  if(!is_ker_name_supported)
  {
    printf("[Error] : Invalid kernel name\n");
    return -1;
  }

  if( (!strcmp(cfg.kernel_name,"conv2d_std")) || (!strcmp(cfg.kernel_name,"dilated_conv2d_std")) )
  {
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    kernel_size = cfg.kernel_height * cfg.kernel_width * cfg.input_channels;
    if(cfg.inp_precision == -1)
    {
      input_channels_pad = (cfg.input_channels + 2 - 1) & ~(2 - 1);
    }
    else
    {
      if(cfg.inp_precision == PREC_8 || cfg.inp_precision == PREC_ASYM8U || cfg.inp_precision == PREC_ASYM8S || cfg.inp_precision == PREC_SYM16S || cfg.inp_precision == PREC_ASYM16S)
        input_channels_pad = cfg.input_channels;
      else
        input_channels_pad = (cfg.input_channels + 4 - 1) & ~(4 - 1);
    }
    kernel_size_pad = cfg.kernel_height * cfg.kernel_width * input_channels_pad;
    bias_size = cfg.out_channels;
    out_size = cfg.out_height * cfg.out_width * cfg.out_channels;
    if(cfg.inp_precision == -4 || cfg.inp_precision == -8 || cfg.inp_precision == -7)
    {
      cfg.p_out_multiplier = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      cfg.p_out_shift = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      int itr_c;
      for(itr_c = 0; itr_c < cfg.out_channels; itr_c++)
      {
        cfg.p_out_multiplier[itr_c] = cfg.out_multiplier;
        cfg.p_out_shift[itr_c] = cfg.out_shift;
      }
    }
  }
  else if((!strcmp(cfg.kernel_name,"conv2d")))
  {
    if(((cfg.input_channels%cfg.kernel_channels)!=0))
    {
      printf("[Error] : Invalid output channels \n");
      return -1;
    }
    cfg.groups=cfg.input_channels/cfg.kernel_channels;
    if(((cfg.out_channels%cfg.groups)!=0))
    {
      printf("[Error] : Invalid output channels \n");
      return -1;
    }
    
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    kernel_size = cfg.kernel_height * cfg.kernel_width * cfg.kernel_channels;
    if(cfg.inp_precision == -1)
    {
      kernel_channels_pad = (cfg.kernel_channels + 2 - 1) & ~(2 - 1);
    }
    else
    {
      if(cfg.inp_precision == PREC_8 || cfg.inp_precision == PREC_ASYM8U || cfg.inp_precision == PREC_ASYM8S || cfg.inp_precision == PREC_SYM16S || cfg.inp_precision == PREC_ASYM16S)
        kernel_channels_pad = cfg.kernel_channels;
      else
        kernel_channels_pad = (cfg.kernel_channels + 4 - 1) & ~(4 - 1);
    }

    kernel_size_pad = cfg.kernel_height * cfg.kernel_width * kernel_channels_pad;
    bias_size = cfg.out_channels;
    out_size = cfg.out_height * cfg.out_width * cfg.out_channels;
    if(cfg.inp_precision == -4 || cfg.inp_precision == -8 || cfg.inp_precision == -7)
    {
      cfg.p_out_multiplier = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      cfg.p_out_shift = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      int itr_c;
      for(itr_c = 0; itr_c < cfg.out_channels; itr_c++)
      {
        cfg.p_out_multiplier[itr_c] = cfg.out_multiplier;
        cfg.p_out_shift[itr_c] = cfg.out_shift;
      }
    }
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_depth") || !strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
  {
    inp_size          = cfg.input_channels      * cfg.input_height        * cfg.input_width;
    kernel_size       = cfg.channels_multiplier * cfg.input_channels      * cfg.kernel_height  * cfg.kernel_width;

    kernel_channels = cfg.channels_multiplier * cfg.input_channels;
    kernel_size_pad = cfg.kernel_height * cfg.kernel_width * kernel_channels;
    dw_out_size       = cfg.channels_multiplier * cfg.input_channels      * cfg.out_height     * cfg.out_width;
    kernel_point_size = cfg.out_channels        * cfg.channels_multiplier * cfg.input_channels * 1 * 1;

    if(!strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
      out_size = dw_out_size;
    else
      out_size = cfg.out_channels        * cfg.out_height          * cfg.out_width;

    bias_size = cfg.channels_multiplier * cfg.input_channels;
    bias_point_size = cfg.out_channels;
    if(cfg.inp_precision == -4 || cfg.inp_precision == -8)
    {
      //As output channels for depthwise convolution and pointwise
      //convolution are different, we need to allocate space for
      //p_out_multiplier and p_out_shift accordingly
      int temp_channels = (cfg.out_channels > (cfg.input_channels * cfg.channels_multiplier)) ? cfg.out_channels : (cfg.input_channels * cfg.channels_multiplier);
      cfg.p_out_multiplier = (int *)malloc(temp_channels*(sizeof(WORD32)));
      cfg.p_out_shift = (int *)malloc(temp_channels*(sizeof(WORD32)));
      int itr_c;
      for(itr_c = 0; itr_c < temp_channels; itr_c++)
      {
        cfg.p_out_multiplier[itr_c] = cfg.out_multiplier;
        cfg.p_out_shift[itr_c] = cfg.out_shift;
      }
    }
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_point"))
  {
    inp_size          = cfg.input_channels      * cfg.input_height        * cfg.input_width;
    kernel_point_size = cfg.out_channels        * cfg.input_channels * 1 * 1;
    out_size          = cfg.out_channels        * cfg.input_height          * cfg.input_width;
    bias_point_size = cfg.out_channels;
    if(cfg.inp_precision == -8 || cfg.inp_precision == -7)
    {
      cfg.p_out_multiplier = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      cfg.p_out_shift = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      int itr_c;
      for(itr_c = 0; itr_c < cfg.out_channels; itr_c++)
      {
        cfg.p_out_multiplier[itr_c] = cfg.out_multiplier;
        cfg.p_out_shift[itr_c] = cfg.out_shift;
      }
    }
  }
  else if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    kernel_size = cfg.kernel_height * cfg.input_width * cfg.input_channels;
    if(cfg.inp_precision == -1)
    {
      input_channelsXwidth_pad = (cfg.input_channels * cfg.input_width + 2 - 1) & ~(2 - 1);
    }
    else
    {
      input_channelsXwidth_pad = (cfg.input_channels * cfg.input_width + 4 - 1) & ~(4 - 1);
    }
    kernel_size_pad = cfg.kernel_height * input_channelsXwidth_pad;
    bias_size = cfg.out_channels;
    out_size = cfg.out_height * cfg.out_channels;
  }
  else if( !strcmp(cfg.kernel_name,"transpose_conv"))
  {
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    input_channels_pad = cfg.input_channels;
    bias_size = cfg.out_channels;
    out_size = cfg.out_height * cfg.out_width * cfg.out_channels;
    num_elements = out_size;
    if(cfg.inp_precision == -8 || cfg.inp_precision == -4)
    {
      cfg.p_out_multiplier = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      cfg.p_out_shift = (int *)malloc(cfg.out_channels*(sizeof(WORD32)));
      int itr_c;
      for(itr_c = 0; itr_c < cfg.out_channels; itr_c++)
      {
        cfg.p_out_multiplier[itr_c] = cfg.out_multiplier;
        cfg.p_out_shift[itr_c] = cfg.out_shift;
      }
    }
  }

  // Set profiler name 
  if(cfg.kernel_name[0])
  {
    strcpy(profiler_name_0,cfg.kernel_name);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcpy(profiler_name_1,"conv2d_point");
    }
  }
  if((cfg.kernel_precision == -1) || (cfg.inp_precision == -1))
  {
    sprintf(profiler_params, "_f32xf32");
    strcat(profiler_name_0, profiler_params);
    
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name_0);
      return 0;
    }
  }
  else if((cfg.kernel_precision == -2) || (cfg.inp_precision == -2))
  {
    sprintf(profiler_params, "_f16xf16");
    strcat(profiler_name_0, profiler_params);
    
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
    // If HP_VFPU is not supported, return
    if(!HIFI_HP_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name_0);
      return 0;
    }
  }  
  else if((cfg.kernel_precision == -3) && (cfg.inp_precision == -3))
  {
    sprintf(profiler_params, "_asym8xasym8");
    strcat(profiler_name_0, profiler_params);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
  }
  else if((cfg.kernel_precision == -5) && (cfg.inp_precision == -4))
  {
    sprintf(profiler_params, "_sym8sxasym8s");
    strcat(profiler_name_0, profiler_params);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
  }
  else if((cfg.kernel_precision == -12) && (cfg.inp_precision == -4))
  {
    sprintf(profiler_params, "_sym4sxasym8s");
    strcat(profiler_name_0, profiler_params);
  }  
  else if((cfg.kernel_precision == -5) && (cfg.inp_precision == -8))
  {
    sprintf(profiler_params, "_sym8sxsym16s");
    strcat(profiler_name_0, profiler_params);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
  }
  else
  {
    sprintf(profiler_params, "_%dx%d", 
        cfg.kernel_precision, cfg.inp_precision);
    strcat(profiler_name_0, profiler_params);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      strcat(profiler_name_1, profiler_params);
    }
  }
  if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    sprintf(profiler_params, "_nhwc");
    if(cfg.inp_data_format == 0)
      strcat(profiler_name_0, profiler_params);
    if(cfg.out_data_format == 0)
      strcat(profiler_name_1, profiler_params);
  }
  if(!strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
  {
    sprintf(profiler_params, "_nhwc");
    if(cfg.inp_data_format == 0)
      strcat(profiler_name_0, profiler_params);
  }
  if(!strcmp(cfg.kernel_name,"conv2d_point"))
  {
    sprintf(profiler_params, "_nhwc");
    if(cfg.out_data_format == 0)
      strcat(profiler_name_0, profiler_params);
  }
  
  
  // Set profiler parameters
  if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, out_channels=%d, out_height=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, cfg.out_height);
  }
  else if(!strcmp(cfg.kernel_name,"dilated_conv2d_std"))
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, out_channels=%d, out_height=%d, dilation_height=%d, dilation_width=%d, x_stride=%d, y_stride=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, cfg.out_height, cfg.dilation_height, cfg.dilation_width, cfg.x_stride, cfg.y_stride);
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_point"))
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, out_channels=%d, out_height=%d, out_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.out_channels, cfg.input_height, cfg.input_width);
  }
  else if(!strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, channels_multiplier=%d, kernel_width=%d, dilation_height=%d, dilation_width=%d, out_height=%d, out_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.channels_multiplier, cfg.dilation_height, cfg.dilation_width, cfg.out_height, cfg.out_width);
  }
  else
  {
    if(!strcmp(cfg.kernel_name,"conv2d_std"))
    {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, kernel_width=%d, out_channels=%d, out_height=%d, out_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, cfg.out_height, cfg.out_width);
    }
    else
    {
      sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d,kernel_channels=%d, kernel_height=%d, kernel_width=%d, out_channels=%d, out_height=%d, out_width=%d,dilation_height=%d, dilation_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels,cfg.kernel_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, cfg.out_height, cfg.out_width,cfg.dilation_height,cfg.dilation_width);

    }  
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
  if( (!strcmp(cfg.kernel_name,"conv2d_std")) || (!strcmp(cfg.kernel_name,"dilated_conv2d_std")) )
  {
    if(cfg.kernel_precision == -12)
    {
      p_kernel = create_buf2D(((cfg.out_channels * cfg.kernel_height * cfg.kernel_width)/2), cfg.input_channels, input_channels_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    }
    else
    {
      p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height * cfg.kernel_width, cfg.input_channels, input_channels_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    }
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, out_size * kernel_size, "MACs/cyc", 1);
  }
  else if( (!strcmp(cfg.kernel_name,"conv2d")) )
  {
    p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height * cfg.kernel_width, cfg.kernel_channels, kernel_channels_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, out_size * kernel_size, "MACs/cyc", 1);
  }
  else if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height, cfg.input_width * cfg.input_channels, input_channelsXwidth_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, out_size * kernel_size, "MACs/cyc", 1);
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_depth") || !strcmp(cfg.kernel_name, "dilated_conv2d_depth"))
  {
    if(cfg.inp_data_format == 0)
    {
      p_kernel = create_buf2D(cfg.kernel_height * cfg.kernel_width, kernel_channels, kernel_channels, cfg.kernel_precision, 0);         VALIDATE_PTR(p_kernel);
    }
    else if(cfg.inp_data_format == 1)
    {
      p_kernel = create_buf2D(kernel_channels * cfg.kernel_height, cfg.kernel_width, cfg.kernel_width, cfg.kernel_precision, 0);            VALIDATE_PTR(p_kernel);
    }
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                      VALIDATE_PTR(p_bias);

    if(!strcmp(cfg.kernel_name, "conv2d_depth"))
    {
      p_kernel_point = create_buf1D(kernel_point_size, cfg.kernel_precision);  VALIDATE_PTR(p_kernel_point);
      p_bias_point = create_buf1D(bias_point_size, cfg.bias_precision);        VALIDATE_PTR(p_bias_point);
      p_dw_out = create_buf1D(dw_out_size, cfg.out_precision);                 VALIDATE_PTR(p_dw_out);      
    }

    int total_conv2d_depth_MACS = (
       (cfg.channels_multiplier * cfg.input_channels * cfg.out_height * cfg.out_width * cfg.kernel_height * cfg.kernel_width) /* MACs in depthwise */
       );
    int total_conv2d_point_MACS = (
       (cfg.out_channels * cfg.channels_multiplier * cfg.input_channels * cfg.out_height * cfg.out_width * 1 * 1)             /* MACs in pointwise */
       );
    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, total_conv2d_depth_MACS, "MACs/cyc", 1);
    if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      XTPWR_PROFILER_OPEN(1, profiler_name_1, profiler_params, total_conv2d_point_MACS, "MACs/cyc", 1);
    }
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_point"))
  {
    p_kernel_point = create_buf1D(kernel_point_size, cfg.kernel_precision);    VALIDATE_PTR(p_kernel_point);
    p_bias_point = create_buf1D(bias_point_size, cfg.bias_precision);          VALIDATE_PTR(p_bias_point);

    int total_conv2d_point_MACS = (
       (cfg.out_channels * cfg.input_channels * cfg.input_height * cfg.input_width * 1 * 1)             /* MACs in pointwise */
       );
    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, total_conv2d_point_MACS, "MACs/cyc", 1);
  }
  else if( !strcmp(cfg.kernel_name,"transpose_conv"))
  {
    int n_macs = 0;
    for (int in_y = 0; in_y < cfg.input_height; ++in_y)
    {
      for (int in_x = 0; in_x < cfg.input_width; ++in_x)
      {
        const int out_x_orig = in_x*cfg.x_stride - cfg.x_padding;
        const int out_y_orig = in_y*cfg.y_stride - cfg.y_padding;
        int filt_x_min = -out_x_orig; 
        int filt_x_max = cfg.out_width - out_x_orig; 
        int filt_y_min = -out_y_orig; 
        int filt_y_max = cfg.out_height - out_y_orig; 
        filt_x_min = (filt_x_min < cfg.kernel_width) ? filt_x_min : cfg.kernel_width;
        filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
        filt_x_max = (filt_x_max < cfg.kernel_width) ? filt_x_max : cfg.kernel_width;
        filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
        filt_y_min = (filt_y_min < cfg.kernel_height) ? filt_y_min : cfg.kernel_height;
        filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
        filt_y_max = (filt_y_max < cfg.kernel_height) ? filt_y_max : cfg.kernel_height;
        filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;

        n_macs += (filt_x_max - filt_x_min)*(filt_y_max - filt_y_min)*cfg.input_channels*cfg.out_channels;  
      }
    }
    kernel_size_pad = cfg.kernel_height * cfg.kernel_width * cfg.input_channels; /* required for comparison when verify = 1 and prec = float32 */
    p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height * cfg.kernel_width, cfg.input_channels, input_channels_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name_0, profiler_params, n_macs, "MACs/cyc", 1);
  }
  
  // Init
  WORD32 scratch_size=0;

  // Get persistent size and allocate 
  if((!strcmp(cfg.kernel_name,"conv2d_std")))
  {
    if(cfg.kernel_precision == -12)
    {
    scratch_size = xa_nn_conv2d_std_getsize_sym4s(cfg.input_height,cfg.input_channels,cfg.kernel_height,cfg.kernel_width,cfg.y_stride,cfg.y_padding,
        cfg.out_height, cfg.out_channels, cfg.inp_precision); PRINT_VAR(scratch_size)
    }
    else
    {
    scratch_size=xa_nn_conv2d_std_getsize(cfg.input_height
                                          ,cfg.input_width
                                          ,cfg.input_channels
                                          ,cfg.kernel_height
                                          ,cfg.kernel_width
                                          ,cfg.input_channels
                                          ,cfg.y_stride
                                          ,cfg.y_padding
                                          ,cfg.x_stride
                                          ,cfg.x_padding
                                          ,cfg.out_height
                                          ,cfg.out_width
                                          ,cfg.out_channels
                                          ,cfg.inp_precision
                                          ,cfg.kernel_precision
                                          ,cfg.dilation_height
                                          ,cfg.dilation_width
                                          ,cfg.out_data_format
                                          ); PRINT_VAR(scratch_size)
    }
  }
  else if((!strcmp(cfg.kernel_name,"conv2d")))
  {
    scratch_size=xa_nn_conv2d_getsize(cfg.input_height
                                      ,cfg.input_width
                                      ,cfg.input_channels
                                      ,cfg.kernel_height
                                      ,cfg.kernel_width
                                      ,cfg.kernel_channels
                                      ,cfg.dilation_height
                                      ,cfg.dilation_width
                                      ,cfg.y_stride
                                      ,cfg.y_padding
                                      ,cfg.x_stride
                                      ,cfg.x_padding
                                      ,cfg.out_height
                                      ,cfg.out_width
                                      ,cfg.out_channels
                                      ,cfg.inp_precision
                                      ,cfg.kernel_precision
                                      ,cfg.out_data_format
                                    );
        PRINT_VAR(scratch_size)
  }
  else if(!strcmp(cfg.kernel_name,"dilated_conv2d_std"))
  {
    scratch_size = xa_nn_dilated_conv2d_std_getsize(cfg.input_height,cfg.input_channels,cfg.kernel_height,cfg.kernel_width,cfg.y_stride,cfg.y_padding,cfg.out_height,cfg.out_channels,cfg.inp_precision,cfg.dilation_height);
    PRINT_VAR(scratch_size)
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    scratch_size =
      xa_nn_conv2d_depthwise_getsize
      (cfg.input_height
       ,cfg.input_width
       ,cfg.input_channels
       ,cfg.kernel_height
       ,cfg.kernel_width
       ,cfg.channels_multiplier
       ,cfg.x_stride
       ,cfg.y_stride
       ,cfg.x_padding
       ,cfg.y_padding
       ,cfg.out_height
       ,cfg.out_width
       ,cfg.inp_precision
       ,cfg.inp_data_format
      );
    PRINT_VAR(scratch_size)
  }
  else if(!strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
  {
    scratch_size =
      xa_nn_dilated_conv2d_depthwise_getsize
      (cfg.input_height
       ,cfg.input_width
       ,cfg.input_channels
       ,cfg.kernel_height
       ,cfg.kernel_width
       ,cfg.channels_multiplier
       ,cfg.dilation_height
       ,cfg.dilation_width
       ,cfg.x_stride
       ,cfg.y_stride
       ,cfg.x_padding
       ,cfg.y_padding
       ,cfg.out_height
       ,cfg.out_width
       ,cfg.inp_precision
       ,cfg.inp_data_format
      );
    PRINT_VAR(scratch_size)
  }
  else if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    scratch_size = xa_nn_conv1d_std_getsize(cfg.kernel_height,cfg.input_width,cfg.input_channels,cfg.inp_precision); PRINT_VAR(scratch_size)
  }
  else if(!strcmp(cfg.kernel_name,"transpose_conv"))
  {
    scratch_size = xa_nn_transpose_conv_getsize(cfg.input_height,cfg.input_width,cfg.input_channels,cfg.kernel_height,cfg.kernel_width,cfg.x_stride,cfg.y_stride,cfg.out_height,cfg.out_width,cfg.out_channels,cfg.kernel_precision,cfg.out_precision); PRINT_VAR(scratch_size)
  }

  if(strcmp(cfg.kernel_name,"conv2d_point"))
  {
    scratch_size=scratch_size<0?0:scratch_size;
    p_scratch = (xa_nnlib_handle_t)malloc(scratch_size); PRINT_PTR(p_scratch)

    fprintf(stdout, "\nScratch size: %d bytes\n", scratch_size);
  }

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    if( (!strcmp(cfg.kernel_name,"conv2d_std")) || (!strcmp(cfg.kernel_name,"dilated_conv2d_std")) || (!strcmp(cfg.kernel_name,"transpose_conv")))
      load_conv2d_std_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, cfg.input_channels, input_channels_pad, -cfg.kernel_zero_bias);
    else if(!strcmp(cfg.kernel_name,"conv2d"))
      load_conv2d_std_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, cfg.kernel_channels, kernel_channels_pad, -cfg.kernel_zero_bias);
    else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
      load_conv2d_ds_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, p_kernel_point, p_bias_point, -cfg.kernel_zero_bias);
    else if(!strcmp(cfg.kernel_name,"dilated_conv2d_depth"))
      load_dilated_conv2d_depth_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, -cfg.kernel_zero_bias);
    else if(!strcmp(cfg.kernel_name,"conv1d_std"))
      load_conv1d_std_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, cfg.input_channels, cfg.input_width, input_channelsXwidth_pad, -cfg.kernel_zero_bias);
    else if(!strcmp(cfg.kernel_name,"conv2d_point"))
      load_conv2d_pt_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel_point, p_bias_point);

    // Call the cnn kernel_name specified on command line
    PROCESS_CONV;
    if(err)
    {
      fprintf(stdout, "\nKernel returned error (invalid parameters), Performance numbers may be incorrect!\n\n");
      pass_count += !err;
      break;
    }

    /* Since there are 2 profilers; one for conv2d_depth one for conv2d_point,
     * thus the update and print will be done in the PROCESS_CONV macro. */
    if(strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      XTPWR_PROFILER_UPDATE(0);
      XTPWR_PROFILER_PRINT(0);
    }

    // Write output into file
    write_buf1D_to_file(fptr_out, p_out);

    // If verify flag enabled, compare output against reference
    if(cfg.verify)
    {
      read_buf1D_from_file(fptr_ref, p_ref);
      pass_count += compare_buf1D(p_ref, p_out, cfg.verify, cfg.out_precision, kernel_size_pad);
    }
    else
    {
      pass_count += !err;
    }
  }

  if(!(!strcmp(cfg.kernel_name,"conv2d_depth") && cfg.pointwise_profile_only))
  {
    XTPWR_PROFILER_CLOSE(0, (pass_count == cfg.frames), cfg.verify);
  }
  if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    XTPWR_PROFILER_CLOSE(1, (pass_count == cfg.frames), cfg.verify);
  }

  fclose(fptr_inp);
  fclose(fptr_out);

  // Free all buffers
  free_buf1D(p_inp);
  if(strcmp(cfg.kernel_name,"conv2d_point"))
  {
    free_buf2D(p_kernel);
    free_buf1D(p_bias);
  }
  else
  {
    free_buf1D(p_kernel_point);
    free_buf1D(p_bias_point);
  }
  free_buf1D(p_out);
  if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    free_buf1D(p_kernel_point);
    free_buf1D(p_bias_point);
    free_buf1D(p_dw_out);
  }
  if(cfg.inp_precision == -4 || cfg.inp_precision == -8 || cfg.inp_precision == -7)
  {
    free(cfg.p_out_multiplier);
    free(cfg.p_out_shift);
  }

  if(cfg.verify)
  {
    fclose(fptr_ref);
    free_buf1D(p_ref);
  }

  if(strcmp(cfg.kernel_name,"conv2d_point"))
  {
    free(p_scratch);
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


