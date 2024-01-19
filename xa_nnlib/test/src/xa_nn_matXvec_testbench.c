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
#include "nnlib/xa_nnlib_standards.h"
#include "xt_manage_buffers.h"
#include "cmdline_parser.h"
#include "file_io.h"
#include "stdbool.h"


#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_ACTIVATION_NAME_LENGTH 20

#define XA_MAX_CMD_LINE_LENGTH 1000
#define XA_MAX_ARGS 40
#define PARAMFILE "paramfilesimple_matXvec.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{
  int help;
  int rows;
  int cols1;
  int cols2;
  int row_stride1;
  int row_stride2;
  int vec_count;
  int acc_shift;
  int bias_shift;
  int mat_precision;
  int inp_precision;
  int out_precision;
  int bias_precision;
  int mat1_zero_bias;
  int mat2_zero_bias;
  int inp1_zero_bias;
  int inp2_zero_bias;
  int out_multiplier;
  int out_shift;
  int out_zero_bias;
  int out_stride;
  char activation[MAX_ACTIVATION_NAME_LENGTH];
  int membank_padding;
  int frames;
  int write_file;
  char read_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_ref_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_out_file_name[XA_MAX_CMD_LINE_LENGTH];
  int verify;
  int batch;
  int fc;
  int matmul;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  { 

    p_cfg->help     = 0;
    p_cfg->rows     = 32;
    p_cfg->cols1    = 32;
    p_cfg->cols2    = 32;
    p_cfg->row_stride1    = 32;
    p_cfg->row_stride2    = 32;
    p_cfg->vec_count = 1;
    p_cfg->acc_shift = 0;
    p_cfg->bias_shift = 0;
    p_cfg->mat_precision   = 16;
    p_cfg->inp_precision = 16;
    p_cfg->out_precision = 16;
    p_cfg->bias_precision = 16;
    p_cfg->mat1_zero_bias = -128;
    p_cfg->mat2_zero_bias = -128;
    p_cfg->inp1_zero_bias = -128;
    p_cfg->inp2_zero_bias = -128;
    p_cfg->out_multiplier = 0x40000000;
    p_cfg->out_shift = -8;
    p_cfg->out_zero_bias = 128;
    p_cfg->out_stride = 1;
    p_cfg->activation[0] = '\0';  
    p_cfg->membank_padding = 1;
    p_cfg->frames   = 2;  
    p_cfg->write_file = 0;  
    p_cfg->read_inp_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;
    p_cfg->batch = 0;
    p_cfg->fc = 0;
    p_cfg->matmul = 0;

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
    printf("\t-rows : rows of mat1; Default=32\n");
    printf("\t-cols1 : columns of mat1 and rows of mat2; should be multiple of 4 (except for quantized datatype kernels); Default=32\n");
    printf("\t-cols2 : columns of mat2; should be multiple of 4 (except for quantized datatype kernels); Default=32\n");
    printf("\t-row_stride1 : row stride for mat1; Default=32\n");
    printf("\t-row_stride2 : row stride for mat2; Default=32\n");
    printf("\t-vec_count : vec count for time batching; Default=1\n");
    printf("\t-acc_shift : Accumulator left shift; Default=0\n");
    printf("\t-bias_shift : Bias left shift; Default=0\n");
    printf("\t-mat_precision : 8, 16, -1(single prec float), -2(half prec float), -3(asym8u), -5(sym8s) or -13(asym4s); Default=16\n");
    printf("\t-inp_precision : 8, 16, -1(single prec float), -2(half prec float), -3(asym8u), -8(sym16s) or -4(asym8s); Default=16\n");
    printf("\t-out_precision : 8, 16, 32, 64 or -1(single prec float), -2(half prec float), -3(asym8u), -4(asym8s) or -8(sym16s); Default=16\n");
    printf("\t-bias_precision : 8, 16, 64, -1(single prec float) or -2(half prec float); Default=16\n");
    printf("\t-mat1_zero_bias : Matrix1 zero_bias for quantized 8-bit, -255 to 0 for asym8u, ignored for sym8s; Default=-128\n");
    printf("\t-mat2_zero_bias : Matrix2 zero_bias for quantized 8-bit, -255 to 0 for asym8u, ignored for sym8s; Default=-128\n");
    printf("\t-inp1_zero_bias : Input1 zero bias for quantized 8-bit, -255 to 0 for asym8u, -127 to 128 for asym8s, 0 for sym16s; Default=-128\n");
    printf("\t-inp2_zero_bias : Input2 zero bias for quantized 8-bit, -255 to 0 for asym8u, -127 to 128 for asym8s, 0 for sym16s; Default=-128\n");
    printf("\t-out_multiplier : Output multiplier in Q31 format for quantized 8-bit, 0x0 to 0x7fffffff; Default=0x40000000\n");
    printf("\t-out_shift : Output shift for quantized 8-bit(asym8u and asym8s), 31 to -31; Default=-8\n");
    printf("\t-out_zero_bias : Output zero bias for quantized 8-bit, 0 to 255 for asym8u, -128 to 127 for asym8s, 0 for sym16s; Default=128\n");
    printf("\t-out_stride : Stride for storing the output; Default=1\n");
    printf("\t-membank_padding: 0, 1; Default=1\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-activation: sigmoid, tanh; Default="" : bypass i.e. no activation for output.\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - mat1, vec1, mat2, vec2, bias) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - mat1, vec1, mat2, vec2, bias) \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
    printf("\t-batch: Flag to check time batching; 0: Disable, 1: Enable; Default=0\n");
    printf("\t-matmul: Flag for matmul, only xa_nn_matmul_asym8sxasym8s_asym8s; 0: Disable, 1: Enable; Default=0\n");
    printf("\t-fc: Flag for fully connected; 0: Disable, 1: Enable; Default=0\n");
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
    ARGTYPE_ONETIME_CONFIG("-rows",p_cfg->rows);
    ARGTYPE_ONETIME_CONFIG("-cols1",p_cfg->cols1);
    ARGTYPE_ONETIME_CONFIG("-cols2",p_cfg->cols2);
    ARGTYPE_ONETIME_CONFIG("-row_stride1",p_cfg->row_stride1);
    ARGTYPE_ONETIME_CONFIG("-row_stride2",p_cfg->row_stride2);
    ARGTYPE_ONETIME_CONFIG("-vec_count",p_cfg->vec_count);
    ARGTYPE_ONETIME_CONFIG("-acc_shift",p_cfg->acc_shift);
    ARGTYPE_ONETIME_CONFIG("-bias_shift",p_cfg->bias_shift);
    ARGTYPE_ONETIME_CONFIG("-mat_precision",p_cfg->mat_precision);
    ARGTYPE_ONETIME_CONFIG("-inp_precision",p_cfg->inp_precision);
    ARGTYPE_ONETIME_CONFIG("-out_precision",p_cfg->out_precision);
    ARGTYPE_ONETIME_CONFIG("-bias_precision",p_cfg->bias_precision);
    ARGTYPE_ONETIME_CONFIG("-mat1_zero_bias",p_cfg->mat1_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-mat2_zero_bias",p_cfg->mat2_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-inp1_zero_bias",p_cfg->inp1_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-inp2_zero_bias",p_cfg->inp2_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-out_multiplier",p_cfg->out_multiplier);
    ARGTYPE_ONETIME_CONFIG("-out_shift",p_cfg->out_shift);
    ARGTYPE_ONETIME_CONFIG("-out_zero_bias",p_cfg->out_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-out_stride",p_cfg->out_stride);
    ARGTYPE_STRING("-activation",p_cfg->activation, MAX_ACTIVATION_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-membank_padding",p_cfg->membank_padding);
    ARGTYPE_ONETIME_CONFIG("-frames",p_cfg->frames);
    ARGTYPE_ONETIME_CONFIG("-write_file",p_cfg->write_file);
    ARGTYPE_STRING("-read_inp_file_name",p_cfg->read_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name",p_cfg->read_ref_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_inp_file_name",p_cfg->write_inp_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_out_file_name",p_cfg->write_out_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify",p_cfg->verify);
    ARGTYPE_ONETIME_CONFIG("-batch",p_cfg->batch);
    ARGTYPE_ONETIME_CONFIG("-fc",p_cfg->fc);
    ARGTYPE_ONETIME_CONFIG("-matmul",p_cfg->matmul);
    
    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}



#define MAT_VEC_MUL_FN(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_##MPREC##x##VPREC##_##OPREC ( \
          (WORD##OPREC *)p_out->p, (WORD##MPREC *) p_mat1->p, (WORD##MPREC *) p_mat2->p, (WORD##VPREC *)p_vec1->p, (WORD##VPREC *)p_vec2->p, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.acc_shift, cfg.bias_shift);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_ASYM8(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_asym8xasym8_asym8 ( \
          (UWORD8 *)p_out->p, (UWORD8 *) p_mat1->p, (UWORD8 *) p_mat2->p, (UWORD8 *)p_vec1->p, (UWORD8 *)p_vec2->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.mat1_zero_bias, cfg.mat2_zero_bias, cfg.inp1_zero_bias, cfg.inp2_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_ASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_asym8sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *) p_mat2->p, (WORD8 *)p_vec1->p, (WORD8 *)p_vec2->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.mat1_zero_bias, cfg.mat2_zero_bias, cfg.inp1_zero_bias, cfg.inp2_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_SYM8SXASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_sym8sxasym8s_asym8s ( \
          (WORD8 *) p_out->p, (WORD8 *) p_mat1->p, (WORD8 *) p_mat2->p, (WORD8 *)p_vec1->p, (WORD8 *)p_vec2->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.inp1_zero_bias, cfg.inp2_zero_bias, cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_SYM8SXSYM16S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_sym8sxsym16s_sym16s ( \
          (WORD16 *) p_out->p, (WORD8 *) p_mat1->p, (WORD8 *) p_mat2->p, (WORD16 *)p_vec1->p, (WORD16 *)p_vec2->p, (WORD64 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.out_multiplier, cfg.out_shift);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_##MPREC##x##VPREC##_##OPREC ( \
          (WORD##OPREC *)p_out->p, (WORD##MPREC *) p_mat1->p, (WORD##VPREC *)p_vec1->p, (VOID *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.acc_shift, cfg.bias_shift);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_ASYM8(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_asym8xasym8_asym8 ( \
          (UWORD8 *)p_out->p, (UWORD8 *) p_mat1->p, (UWORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.inp1_zero_bias, cfg.mat1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_ASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_asym8sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.mat1_zero_bias, cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_ASYM4S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_asym4sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.mat1_zero_bias, cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, p_scratch->p);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_SYM8SXASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_sym8sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_SYM8SXSYM16S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_sym8sxsym16s_sym16s ( \
          (WORD16 *)p_out->p, (WORD8 *) p_mat1->p, (WORD16 *)p_vec1->p, (WORD64 *)p_bias->p, \
          cfg.cols1, cfg.rows, \
          cfg.out_multiplier, cfg.out_shift);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_OUT_STRIDE_FN_SYM8SXASYM8S_16(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      memset(p_out->p, 0xca, cfg.rows*cfg.out_stride*sizeof(WORD16)); \
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_out_stride_sym8sxasym8s_16 ( \
          (WORD16 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, cfg.out_stride, \
          cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_F32(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_f32 ( \
          (FLOAT32 *)p_out->p, (FLOAT32 *) p_mat1->p, (FLOAT32 *)p_vec1->p, (FLOAT32 *)p_bias->p, \
          cfg.cols1, cfg.rows); \
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FC_FN_F16(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_fully_connected_f16 ( \
          (WORD16 *)p_out->p, (WORD16 *) p_mat1->p, (WORD16 *)p_vec1->p, (WORD16 *)p_bias->p, \
          cfg.cols1, cfg.rows); \
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_BATCH(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      int i;\
      /*WORD##VPREC *pp_vec1[200]; WORD##OPREC *pp_out[200];*/\
      WORD##VPREC **pp_vec1; WORD##OPREC **pp_out;\
      pp_vec1 = (WORD##VPREC **)malloc(sizeof(WORD##VPREC *)*cfg.vec_count);\
      pp_out = (WORD##OPREC **)malloc(sizeof(WORD##OPREC *)*cfg.vec_count);\
      for (i=0; i<cfg.vec_count; i++){\
        *((WORD##VPREC **) pp_vec1 + i) =  ((WORD##VPREC *)p_vec1->p + i*cfg.cols1);\
        *((WORD##OPREC **) pp_out + i) = ((WORD##OPREC *)p_out->p + i*cfg.rows);\
      }\
      err = xa_nn_matXvec_batch_##MPREC##x##VPREC##_##OPREC ( \
          (WORD##OPREC **)pp_out, (WORD##MPREC *) p_mat1->p, (WORD##VPREC **)pp_vec1, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.acc_shift, cfg.bias_shift, cfg.vec_count);\
      free(pp_vec1);\
      free(pp_out);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_ASYM8_BATCH(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      int i;\
      /*WORD##VPREC *pp_vec1[200]; WORD##OPREC *pp_out[200];*/\
      UWORD8 **pp_vec1; UWORD8 **pp_out;\
      pp_vec1 = (UWORD8 **)malloc(sizeof(UWORD8 *)*cfg.vec_count);\
      pp_out = (UWORD8 **)malloc(sizeof(UWORD8 *)*cfg.vec_count);\
      for (i=0; i<cfg.vec_count; i++){\
        *((UWORD8 **) pp_vec1 + i) =  ((UWORD8 *)p_vec1->p + i*cfg.cols1);\
        *((UWORD8 **) pp_out + i) = ((UWORD8 *)p_out->p + i*cfg.rows);\
      }\
      err = xa_nn_matXvec_batch_asym8xasym8_asym8 ( \
          (UWORD8 **)pp_out, (UWORD8 *) p_mat1->p, (UWORD8 **)pp_vec1, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.mat1_zero_bias, cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      free(pp_vec1);\
      free(pp_out);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_8X8_ASYM16S_BATCH(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      /*memset(p_out->p, 0xe8, p_out->length*p_out->bytes_per_element);*/ \
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_acc_batch_sym8sx8_asym16s ( \
          (WORD16 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, cfg.vec_count);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_ACTIVATION_FN(MPREC, VPREC, OPREC, ACTIVATION) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_##MPREC##x##VPREC##_##OPREC##_##ACTIVATION ( \
          (WORD##OPREC *)p_out->p, (WORD##MPREC *) p_mat1->p, (WORD##MPREC *) p_mat2->p, (WORD##VPREC *)p_vec1->p, (WORD##VPREC *)p_vec2->p, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          cfg.acc_shift, cfg.bias_shift, cfg.bias_precision, \
          (VOID *)p_scratch->p);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_F32(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_f32xf32_f32 ( \
          (FLOAT32 *)p_out->p, (FLOAT32 *) p_mat1->p, (FLOAT32 *) p_mat2->p, (FLOAT32 *)p_vec1->p, (FLOAT32 *)p_vec2->p, (FLOAT32 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset); \
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_FN_F32_BATCH(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      int i;\
      /*FLOAT32 *pp_vec1[200]; FLOAT32 *pp_out[200];\*/\
      FLOAT32 **pp_vec1; FLOAT32 **pp_out;\
      pp_vec1 = (FLOAT32 **)malloc(sizeof(FLOAT32 *)*cfg.vec_count);\
      pp_out = (FLOAT32 **)malloc(sizeof(FLOAT32 *)*cfg.vec_count);\
      for (i=0; i<cfg.vec_count; i++){\
        *((FLOAT32 **) pp_vec1 + i) =  ((FLOAT32 *)p_vec1->p + i*cfg.cols1);\
        *((FLOAT32 **) pp_out + i) = ((FLOAT32 *)p_out->p + i*cfg.rows);\
      }\
      err = xa_nn_matXvec_batch_f32xf32_f32( \
          (FLOAT32 **)pp_out, (FLOAT32 *) p_mat1->p, (FLOAT32 **)pp_vec1, (FLOAT32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count);\
      free(pp_vec1);\
      free(pp_out);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MAT_VEC_MUL_ACTIVATION_FN_F32(MPREC, VPREC, OPREC, ACTIVATION) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matXvec_f32xf32_f32_##ACTIVATION ( \
          (FLOAT32 *)p_out->p, (FLOAT32 *) p_mat1->p, (FLOAT32 *) p_mat2->p, (FLOAT32 *)p_vec1->p, (FLOAT32 *)p_vec2->p, (FLOAT32 *)p_bias->p, \
          cfg.rows, cfg.cols1, cfg.cols2, p_mat1->row_offset, p_mat2->row_offset, \
          (FLOAT32 *)p_scratch->p);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MATMUL_FN_ASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_asym8sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.cols1, cfg.rows, 1, \
          cfg.mat1_zero_bias, cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MATMUL_FN_SYM4S_ASYM8S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_asym4sxasym8s_asym8s ( \
          (WORD8 *)p_out->p, (WORD8 *) p_mat1->p, (WORD8 *)p_vec1->p, (WORD32 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.cols1, cfg.rows, 1, \
          cfg.mat1_zero_bias, cfg.inp1_zero_bias, \
          cfg.out_multiplier, cfg.out_shift, cfg.out_zero_bias, p_scratch->p);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MATMUL_FN_SYM8S_SYM16S(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_sym8sxsym16s_sym16s ( \
          (WORD16 *)p_out->p, (WORD8 *) p_mat1->p, (WORD16 *)p_vec1->p, (WORD64 *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.cols1, cfg.rows, 1, \
          0, cfg.out_multiplier, cfg.out_shift, 0);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MATMUL_FN_PLAIN(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_##MPREC##x##VPREC##_##OPREC ( \
          (WORD##OPREC *)p_out->p, (WORD##MPREC *) p_mat1->p, (WORD##VPREC *) p_vec1->p, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.acc_shift, cfg.bias_shift, cfg.vec_count, cfg.cols1, 1, cfg.vec_count);\
      XTPWR_PROFILER_STOP(0);\
    }

#define MATMUL_FN_PLAIN_F32(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_f32xf32_f32( \
          (FLOAT32 *)p_out->p, (FLOAT32 *) p_mat1->p, (FLOAT32 *) p_vec1->p, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.cols1, 1, cfg.vec_count);\
      XTPWR_PROFILER_STOP(0);\
    }
#define MATMUL_FN_PLAIN_F16(MPREC, VPREC, OPREC) \
    if((MPREC == p_mat1->precision) && (VPREC == p_vec1->precision) && (OPREC == p_out->precision)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_matmul_f16xf16_f16( \
          (WORD16 *)p_out->p, (WORD16 *) p_mat1->p, (WORD16 *) p_vec1->p, (VOID *)p_bias->p, \
          cfg.rows, cfg.cols1, p_mat1->row_offset, \
          cfg.vec_count, cfg.cols1, 1, cfg.vec_count);\
      XTPWR_PROFILER_STOP(0);\
    }

#if HIFI_VFPU 
#define PROCESS_MATXVEC \
    MAT_VEC_MUL_ACTIVATION_FN(16, 16, 16, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(16, 16, 16, tanh) \
    else MAT_VEC_MUL_FN(16, 16, 16) \
    else MAT_VEC_MUL_FN(16, 16, 32) \
    else MAT_VEC_MUL_FN(16, 16, 64) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 16, 16, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 16, 16, tanh) \
    else MAT_VEC_MUL_FN(8, 16, 16) \
    else MAT_VEC_MUL_FN(8, 16, 32) \
    else MAT_VEC_MUL_FN(8, 16, 64) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 8, 8, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 8, 8, tanh) \
    else MAT_VEC_MUL_FN(8, 8, 8) \
    else MAT_VEC_MUL_FN(8, 8, 16) \
    else MAT_VEC_MUL_FN(8, 8, 32) \
    else MAT_VEC_MUL_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FN_SYM8SXSYM16S(-5, -8, -8) \
    else MAT_VEC_MUL_OUT_STRIDE_FN_SYM8SXASYM8S_16(-5, -4, 16)  \
    else MAT_VEC_MUL_ACTIVATION_FN_F32(-1, -1, -1, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN_F32(-1, -1, -1, tanh) \
    else MAT_VEC_MUL_FN_F32(-1, -1, -1) \
    else {  printf("unsupported multiplication\n"); return -1;} 
#else
#define PROCESS_MATXVEC \
    MAT_VEC_MUL_ACTIVATION_FN(16, 16, 16, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(16, 16, 16, tanh) \
    else MAT_VEC_MUL_FN(16, 16, 16) \
    else MAT_VEC_MUL_FN(16, 16, 32) \
    else MAT_VEC_MUL_FN(16, 16, 64) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 16, 16, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 16, 16, tanh) \
    else MAT_VEC_MUL_FN(8, 16, 16) \
    else MAT_VEC_MUL_FN(8, 16, 32) \
    else MAT_VEC_MUL_FN(8, 16, 64) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 8, 8, sigmoid) \
    else MAT_VEC_MUL_ACTIVATION_FN(8, 8, 8, tanh) \
    else MAT_VEC_MUL_FN(8, 8, 8) \
    else MAT_VEC_MUL_FN(8, 8, 16) \
    else MAT_VEC_MUL_FN(8, 8, 32) \
    else MAT_VEC_MUL_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FN_SYM8SXSYM16S(-5, -8, -8) \
    else MAT_VEC_MUL_OUT_STRIDE_FN_SYM8SXASYM8S_16(-5, -4, 16)  \
    else {  printf("unsupported multiplication\n"); return -1;} 
#endif

#if HIFI_VFPU 
#if HIFI_HP_VFPU && hifi5
#define PROCESS_MATXVEC_FC \
    MAT_VEC_MUL_FC_FN(16, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 8, 8) \
    else MAT_VEC_MUL_FC_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FC_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FC_FN_ASYM4S(-13, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXSYM16S(-5, -8, -8) \
    else MAT_VEC_MUL_FC_FN_F32(-1, -1, -1) \
    else MAT_VEC_MUL_FC_FN_F16(-2, -2, -2) \
    else {  printf("unsupported multiplication\n"); return -1;} 
#else /* HIFI_HP_VFPU && hifi5 */
#define PROCESS_MATXVEC_FC \
    MAT_VEC_MUL_FC_FN(16, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 8, 8) \
    else MAT_VEC_MUL_FC_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FC_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FC_FN_ASYM4S(-13, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXSYM16S(-5, -8, -8) \
    else MAT_VEC_MUL_FC_FN_F32(-1, -1, -1) \
    else {  printf("unsupported multiplication\n"); return -1;} 
#endif /* HIFI_HP_VFPU && hifi5 */
#else /* HIFI_VFPU */
#if HIFI_HP_VFPU && hifi5 /* HIFI_HP_VFPU && hifi5 */
#define PROCESS_MATXVEC_FC \
    MAT_VEC_MUL_FC_FN(16, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 8, 8) \
    else MAT_VEC_MUL_FC_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FC_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FC_FN_ASYM4S(-13, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXSYM16S(-5, -8, -8) \
    else MAT_VEC_MUL_FC_FN_F32(-2, -2, -2) \
    else {  printf("unsupported multiplication\n"); return -1;}
#else/* HIFI_HP_VFPU && hifi5 */
#define PROCESS_MATXVEC_FC \
    MAT_VEC_MUL_FC_FN(16, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 16, 16) \
    else MAT_VEC_MUL_FC_FN(8, 8, 8) \
    else MAT_VEC_MUL_FC_FN_ASYM8(-3, -3, -3) \
    else MAT_VEC_MUL_FC_FN_ASYM8S(-4, -4, -4) \
    else MAT_VEC_MUL_FC_FN_ASYM4S(-13, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXASYM8S(-5, -4, -4) \
    else MAT_VEC_MUL_FC_FN_SYM8SXSYM16S(-5, -8, -8) \
    else {  printf("unsupported multiplication\n"); return -1;}
#endif/* HIFI_HP_VFPU && hifi5 */
#endif /* HIFI_VFPU */

#if HIFI_VFPU 
#define PROCESS_MATXVEC_BATCH \
    MAT_VEC_MUL_FN_BATCH(16, 16, 64) \
    else MAT_VEC_MUL_FN_BATCH(8, 16, 64) \
    else MAT_VEC_MUL_FN_BATCH(8, 8, 32) \
    else MAT_VEC_MUL_FN_ASYM8_BATCH(-3, -3, -3) \
    else MAT_VEC_MUL_FN_8X8_ASYM16S_BATCH(-5, 8, -7) \
    else MAT_VEC_MUL_FN_F32_BATCH(-1, -1, -1) \
    else {  printf("unsupported multiplication\n"); return -1;} 
#else
#define PROCESS_MATXVEC_BATCH \
    MAT_VEC_MUL_FN_BATCH(16, 16, 64) \
    else MAT_VEC_MUL_FN_BATCH(8, 16, 64) \
    else MAT_VEC_MUL_FN_BATCH(8, 8, 32) \
    else MAT_VEC_MUL_FN_ASYM8_BATCH(-3, -3, -3) \
    else MAT_VEC_MUL_FN_8X8_ASYM16S_BATCH(-5, 8, -7) \
    else {  printf("unsupported multiplication\n"); return -1;} 
#endif

#if HIFI_VFPU
#if HIFI_HP_VFPU && hifi5
#define PROCESS_MATMUL \
    MATMUL_FN_ASYM8S(-4, -4, -4) \
    else MATMUL_FN_SYM8S_SYM16S(-5, -8, -8) \
    else MATMUL_FN_SYM4S_ASYM8S(-13, -4, -4) \
    else MATMUL_FN_PLAIN(8, 16, 16) \
    else MATMUL_FN_PLAIN(16, 16, 16) \
    else MATMUL_FN_PLAIN_F32(-1, -1, -1) \
    else MATMUL_FN_PLAIN_F16(-2, -2, -2) \
    else { printf("unsupported multiplication\n"); return -1;}
#else
#define PROCESS_MATMUL \
    MATMUL_FN_ASYM8S(-4, -4, -4) \
    else MATMUL_FN_SYM8S_SYM16S(-5, -8, -8) \
    else MATMUL_FN_SYM4S_ASYM8S(-13, -4, -4) \
    else MATMUL_FN_PLAIN(8, 16, 16) \
    else MATMUL_FN_PLAIN(16, 16, 16) \
    else MATMUL_FN_PLAIN_F32(-1, -1, -1) \
    else { printf("unsupported multiplication\n"); return -1;}
#endif //HIFI_HP_VFPU && hifi5 end
#else
#if HIFI_HP_VFPU && hifi5
#define PROCESS_MATMUL \
    MATMUL_FN_ASYM8S(-4, -4, -4) \
    else MATMUL_FN_SYM8S_SYM16S(-5, -8, -8) \
    else MATMUL_FN_SYM4S_ASYM8S(-13, -4, -4) \
    else MATMUL_FN_PLAIN(8, 16, 16) \
    else MATMUL_FN_PLAIN(16, 16, 16) \
    else MATMUL_FN_PLAIN_F16(-2, -2, -2) \
    else { printf("unsupported multiplication\n"); return -1;}
#else
#define PROCESS_MATMUL \
    MATMUL_FN_ASYM8S(-4, -4, -4) \
    else MATMUL_FN_SYM8S_SYM16S(-5, -8, -8) \
    else MATMUL_FN_SYM4S_ASYM8S(-13, -4, -4) \
    else MATMUL_FN_PLAIN(8, 16, 16) \
    else MATMUL_FN_PLAIN(16, 16, 16) \
    else { printf("unsupported multiplication\n"); return -1;}
#endif //HIFI_HP_VFPU && hifi5 end
#endif

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  //int i;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 

  test_config_t cfg;

  buf2D_t *p_mat1;
  buf1D_t *p_vec1;
  buf2D_t *p_mat2;
  buf1D_t *p_vec2;
  buf1D_t *p_bias;
  buf1D_t *p_out;
  buf1D_t *p_scratch;
  buf1D_t *ptr_ref;
  int scratch_size = 0;

  /* Some kernels like the *_acc_batch_* require (a one time) initialization
   * of the p_out buffer as they do a p_out = SOME_OP(p_out, SOME_DATA).
   * This flag is used to memset the initial contents of this p_out buffer. 
   */
  bool initialize_p_out_memory = false;

  size_t out_buffer_size = 0;

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
    if(strcmp(cfg.activation, "sigmoid") && strcmp(cfg.activation, "tanh") && strcmp(cfg.activation, ""))
    {
      printf("Invalid activation %s\n", cfg.activation);
      show_usage();
      return 0;
    }
    if(1 == cfg.help)
    {
      show_usage();
      return 0;
    }
  }

  if(cfg.fc == 1){
    /* In fully connected apis, row_stride is equal to cols, thus setting
     * membank_padding to 0. */
    cfg.membank_padding = 0;
    cfg.row_stride1 = cfg.cols1;
    cfg.row_stride2 = cfg.cols2;
  }

  // Set profiler name 
  if((cfg.mat_precision == -1) && (cfg.inp_precision == -1) && (cfg.out_precision == -1))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_f32");
    }
    else if(cfg.matmul == 1) {
      sprintf(profiler_name,"matmul_f32xf32_f32");
    }
    else{
      sprintf(profiler_name,"matXvec%s_f32xf32_f32",(cfg.batch)? "_batch": "");
    }
    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else if((cfg.mat_precision == -2) && (cfg.inp_precision == -2) && (cfg.out_precision == -2))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_f16");
    }
    else if(cfg.matmul == 1) {
      sprintf(profiler_name,"matmul_f16xf16_f16");
    }
    else{
      sprintf(profiler_name,"matXvec%s_f16xf16_f16",(cfg.batch)? "_batch": "");
    }
// If HP_VFPU is not supported, return
    if(!HIFI_HP_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }    
  }
  else if((cfg.mat_precision == -3) && (cfg.inp_precision == -3) && (cfg.out_precision == -3))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_asym8xasym8_asym8");
    }
    else{
      sprintf(profiler_name,"matXvec%s_asym8xasym8_asym8",(cfg.batch)? "_batch": "");
    }
  }
  else if((cfg.mat_precision == -4) && (cfg.inp_precision == -4) && (cfg.out_precision == -4))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_asym8sxasym8s_asym8s");
    }
    else if(cfg.matmul == 1) {
      sprintf(profiler_name,"matmul_asym8sxasym8s_asym8s");
    }
    else {
      sprintf(profiler_name,"matXvec%s_asym8sxasym8s_asym8s",(cfg.batch)? "_batch": "");
    }
  }
  else if((cfg.mat_precision == -13) && (cfg.inp_precision == -4) && (cfg.out_precision == -4))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_asym4sxasym8s_asym8s");
    }
    else if(cfg.matmul == 1) {
      sprintf(profiler_name,"matmul_asym4sxasym8s_asym8s");
    }
    else{
      printf("%s: NOT TESTED\n", profiler_name);
    }
  }
  else if((cfg.mat_precision == -5) && (cfg.inp_precision == -4) && (cfg.out_precision == -4))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_sym8sxasym8s_asym8s");
    }
    else{
      sprintf(profiler_name,"matXvec%s_sym8sxasym8s_asym8s",(cfg.batch)? "_batch": "");
    }
  }
  else if((cfg.mat_precision == -5) && (cfg.inp_precision == -7) && (cfg.out_precision == -7))
  {
    if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_sym8sxasym16s_asym16s");
    }
    else{
      sprintf(profiler_name,"matXvec%s_sym8sxasym16s_asym16s",(cfg.batch)? "_batch": "");
    }
  }
  else if((cfg.mat_precision == -5) && (cfg.inp_precision == -8) && (cfg.out_precision == -8))
  {
    if(cfg.matmul == 1) {
      sprintf(profiler_name,"matmul_sym8sxsym16s_sym16s");
    }
    else if(cfg.fc == 1){
      sprintf(profiler_name,"fully_connected_sym8sxsym16s_sym16s");
    }
    else{
      sprintf(profiler_name,"matXvec%s_sym8sxsym16s_sym16s",(cfg.batch)? "_batch": "");
    }
  }
  else if((cfg.mat_precision == -5) && (cfg.inp_precision == -4) && (cfg.out_precision == 16))
  {
      sprintf(profiler_name,"matXvec%s_sym8sxasym8s_16",(cfg.batch)? "_batch": "");
  }
  else if((cfg.mat_precision == -5) && (cfg.inp_precision == 8) && (cfg.out_precision == -7))
  {
      sprintf(profiler_name,"matXvec%s_sym8sx8_asym16s",(cfg.batch)? "_acc_batch": "");
      initialize_p_out_memory = true;
  }
  else
  {
    if(cfg.fc == 1){
      sprintf(profiler_name, "fully_connected_%dx%d_%d",cfg.mat_precision, cfg.inp_precision, cfg.out_precision); 
    }
    else if(cfg.matmul == 1) {
    sprintf(profiler_name,"matmul_%dx%d_%d", cfg.mat_precision, cfg.inp_precision, cfg.out_precision);
    }
    else{
      sprintf(profiler_name, "matXvec%s_%dx%d_%d",(cfg.batch)? "_batch" : "",cfg.mat_precision, cfg.inp_precision, cfg.out_precision); 
    }
  }
  
  if(cfg.activation[0])
  {
    sprintf(profiler_name,"%s_%s",profiler_name,cfg.activation);
  }
  
  // Set profiler parameters
  if(cfg.batch == 1 || cfg.matmul == 1){
    sprintf(profiler_params, "rows=%d, cols1=%d, bias_prec=%d, vec_count=%d", 
      cfg.rows, cfg.cols1, cfg.bias_precision,cfg.vec_count);
  }
  else{
    sprintf(profiler_params, "rows=%d, cols1=%d, cols2=%d, bias_prec=%d", 
      cfg.rows, cfg.cols1, cfg.cols2, cfg.bias_precision);
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
  
  // The total size of the output buffer (which will be greater than just rows*columns in case of out_stride kernels)
  out_buffer_size = cfg.rows*cfg.vec_count*cfg.out_stride;

  // Open reference file if verify flag is enabled
  if(cfg.verify)
  {
    ptr_ref =  create_buf1D(out_buffer_size, cfg.out_precision); 
    
    fptr_ref = file_open(pb_ref_file_path, cfg.read_ref_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  if((cfg.mat_precision == -13) && (cfg.inp_precision == -4) && (cfg.out_precision == -4))
  {
    // scratch size required for fully_connected_asym4sxasym8s kernel
    scratch_size = 16 + cfg.cols1;
    // matmul requires higher scratch-size
    if(cfg.matmul == 1) {
      scratch_size = 4*(32 + cfg.cols1)+32;
    }
  }
  else
  {
    // scratch size required for matXvec activation kernels
    scratch_size = cfg.rows*4;
  }

  // Allocate Memory
  p_mat1 = create_buf2D(cfg.rows, cfg.cols1, cfg.row_stride1, cfg.mat_precision, cfg.membank_padding);    VALIDATE_PTR(p_mat1);
  p_vec1 = create_buf1D(cfg.cols1*cfg.vec_count, cfg.inp_precision);                                      VALIDATE_PTR(p_vec1);
  p_mat2 = create_buf2D(cfg.rows, cfg.cols2, cfg.row_stride2, cfg.mat_precision, cfg.membank_padding);    VALIDATE_PTR(p_mat2);
  p_vec2 = create_buf1D(cfg.cols2, cfg.inp_precision);                                                    VALIDATE_PTR(p_vec2);
  p_bias = create_buf1D(cfg.rows, cfg.bias_precision);                                                    VALIDATE_PTR(p_bias);
  p_out  = create_buf1D(out_buffer_size, cfg.out_precision);                                              VALIDATE_PTR(p_out);
  p_scratch = create_buf1D(scratch_size, 8);                                                              VALIDATE_PTR(p_scratch);

  if(initialize_p_out_memory){
    memset(p_out->p, 0xE8, p_out->length * p_out->bytes_per_element);
  }

  if(cfg.inp_precision == cfg.out_precision && (!strcmp(cfg.activation, "sigmoid") || !strcmp(cfg.activation, "tanh"))){
    fprintf(stdout, "\nScratch size: %d bytes\n", scratch_size);
  }
  if(cfg.batch == 1 || cfg.matmul == 1){
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, (cfg.rows * cfg.cols1 * cfg.vec_count), "MACs/cyc", 1);
  }
  else if(cfg.fc == 1){
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, (cfg.rows * cfg.cols1), "MACs/cyc", 1);
  }
  else {
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, (cfg.rows * (cfg.cols1 + cfg.cols2)), "MACs/cyc", 1);
  }


  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    load_matXvec_input_data(cfg.write_file, fptr_inp, p_mat1, p_vec1, p_mat2, p_vec2, p_bias);

    // Call the matXvec kernel specified on command line
    if(cfg.batch == 1){
        PROCESS_MATXVEC_BATCH;
    }
    else if(cfg.fc == 1){
        PROCESS_MATXVEC_FC;
    }
    else if(cfg.matmul == 1){
        PROCESS_MATMUL;
    }
    else{
        PROCESS_MATXVEC;
    }

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
      read_buf1D_from_file(fptr_ref, ptr_ref);
      pass_count += compare_buf1D(ptr_ref, p_out, cfg.verify, cfg.out_precision, 1);
    }
    else
    {
      pass_count += !err;
    }
  }

  XTPWR_PROFILER_CLOSE(0, (pass_count == cfg.frames), cfg.verify);
  printf("\r\n");


  fclose(fptr_inp);
  fclose(fptr_out);
  // Free all buffers
  free_buf2D(p_mat1);
  free_buf2D(p_mat2);
  free_buf1D(p_vec1);
  free_buf1D(p_vec2);
  free_buf1D(p_bias);
  free_buf1D(p_out);
  free_buf1D(p_scratch);

  if(cfg.verify)
  {
    fclose(fptr_ref);
    free_buf1D(ptr_ref);
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


