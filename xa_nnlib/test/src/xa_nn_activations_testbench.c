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
#include "nnlib/xa_nnlib_standards.h"
#include "xt_manage_buffers.h"
#include "cmdline_parser.h"
#include "file_io.h"
#include "nnlib/xa_nnlib_api.h"

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_FILE_NAME_LENGTH       256
#define MAX_ACTIVATION_NAME_LENGTH 20

#define XA_MAX_CMD_LINE_LENGTH 300
#define XA_MAX_ARGS 30
#define PARAMFILE "paramfilesimple_activations.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{
// asym8(data type) specific parameters
  int diffmin;
  int input_left_shift; 
  int input_multiplier;
  int input_range_radius;
  int zero_point;
  int inp_zero_bias;
  int alpha_zero_bias;
  int alpha_multiplier;
  int alpha_shift;
  int reluish_multiplier;
  int reluish_shift;
  int out_multiplier;
  int out_shift;
  int out_zero_bias;
  int integer_bits;
  int help;
  int num_elements;
  int relu_threshold;
  int inp_precision;
  int out_precision;
  int activation_min; // used in relu_asym8/16/8 (activation_min_max)
  int activation_max; // used in relu_asym8/16/8
  float activation_min_f32;
  float activation_max_f32;
  char activation[MAX_ACTIVATION_NAME_LENGTH];
  int frames;
  int write_file;
  char read_inp_file_name[MAX_FILE_NAME_LENGTH];
  char read_ref_file_name[MAX_FILE_NAME_LENGTH];
  char write_inp_file_name[MAX_FILE_NAME_LENGTH];
  char write_out_file_name[MAX_FILE_NAME_LENGTH];
  int verify;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  { 
    p_cfg->help     = 0;
    p_cfg->diffmin  = -15;
    p_cfg->input_left_shift = 27;
    p_cfg->input_multiplier = 2060158080;
    p_cfg->input_range_radius = 128;
    p_cfg->zero_point = 0; 
    p_cfg->inp_zero_bias = 0;
    p_cfg->alpha_zero_bias = 0;
    p_cfg->alpha_multiplier = 0x40000000;
    p_cfg->alpha_shift = 0;
    p_cfg->reluish_multiplier = 0x40000000;
    p_cfg->reluish_shift = 0;
    p_cfg->out_multiplier = 0x40000000;
    p_cfg->out_shift = -8;
    p_cfg->out_zero_bias = 0;
    p_cfg->num_elements = 32;
    p_cfg->relu_threshold = (1<<15); // threshold=1, Q16.15
    p_cfg->inp_precision = 32;
    p_cfg->out_precision = 32;
    p_cfg->integer_bits = 3;
    p_cfg->activation_min = 0; 
    p_cfg->activation_max = 127; 
    p_cfg->activation_min_f32 = 0.0; 
    p_cfg->activation_max_f32 = 1.0; 
    strcpy(p_cfg->activation,"sigmoid");
    p_cfg->frames   = 2;  
    p_cfg->write_file = 0;  
    p_cfg->read_inp_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;

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
    printf("\t-num_elements : number of elements; Default=32\n");
    printf("\t-relu_threshold : threshold for relu in Q16.15; Default=32768 (=1 in Q16.15)\n");
    printf("\t-inp_precision : 16, 32, -1(single prec float),-2(half prec float), -3 (asym8u), -4 (asym8s), -7 (asym16s) or -8 (sym16s); Default=32\n");
    printf("\t-out_precision : 16, 32, -1(single prec float),-2(half prec float), -3 (asym8u), -4 (asym8s), -7 (asym16s) or -8 (sym16s); Default=32\n");
    printf("\t-integer_bits : number of integer bits in input for tanh_16_16 (0-6); Default=3\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-activation: sigmoid, tanh, relu, relu_std, relu1, relu6, leaky_relu, prelu, hard_swish, activation_min_max or softmax; Default=sigmoid\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading input \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing input \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
    printf("\t =====================================\n ");
    printf("\t ===== ASYM8 specific parameters =====\n ");
    printf("\t =====================================\n ");
    printf ("\t-diffmin: diffmin; Default=-15\n");
    printf ("\t-input_left_shift: input_left_shift;   Default=27\n");
    printf ("\t-input_multiplier: input_multiplier; Default=2060158080\n");
    printf("\t-activation_max: asym8/16/8 input data activation max; Default=0\n");
    printf("\t-activation_min: asym8/16/8 input data activation min; Default=0\n");
    printf("\t-activation_max_f32: float input data activation max; Default=0\n");
    printf("\t-activation_min_f32: float input data activation min; Default=0\n");
    printf("\t-input_range_radius: sigmoid_asym8 input parameter; Default=128\n");
    printf("\t-zero_point: sigmoid_asym8 input parameter; Default=0\n");
    printf("\t-inp_zero_bias: Zero bias value for input Default=0\n");
    printf("\t-alpha_zero_bias: Prelu parameter - Zero bias value for alpha Default=0\n");
    printf("\t-alpha_multiplier: Leaky Relu and Prelu parameter - Multiplier value for alpha Default=0x40000000\n");
    printf("\t-alpha_shift: Leaky Relu and Prelu parameter - Shift value for alpha Default=0\n");
    printf("\t-reluish_multiplier: Hard Swish parameter - Multiplier value for relu scale Default=0x40000000\n");
    printf("\t-reluish_shift: Hard Swish parameter - Shift value for relu scale Default=0\n");
    printf("\t-out_multiplier: Multiplier value for output Default=0x40000000\n");
    printf("\t-out_shift: Shift value for output Default=0\n");
    printf("\t-out_zero_bias: Zero bias value for output Default=0\n");
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
    ARGTYPE_ONETIME_CONFIG("-diffmin", p_cfg->diffmin);                       
    ARGTYPE_ONETIME_CONFIG("-input_left_shift",p_cfg->input_left_shift);                        
    ARGTYPE_ONETIME_CONFIG("-input_multiplier",p_cfg->input_multiplier);                
    ARGTYPE_ONETIME_CONFIG("-input_range_radius",p_cfg->input_range_radius);                
    ARGTYPE_ONETIME_CONFIG("-zero_point",p_cfg->zero_point);                
    ARGTYPE_ONETIME_CONFIG("-inp_zero_bias",p_cfg->inp_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-alpha_zero_bias",p_cfg->alpha_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-alpha_multiplier",p_cfg->alpha_multiplier);
    ARGTYPE_ONETIME_CONFIG("-alpha_shift",p_cfg->alpha_shift);
    ARGTYPE_ONETIME_CONFIG("-reluish_multiplier",p_cfg->reluish_multiplier);
    ARGTYPE_ONETIME_CONFIG("-reluish_shift",p_cfg->reluish_shift);
    ARGTYPE_ONETIME_CONFIG("-out_multiplier",p_cfg->out_multiplier);
    ARGTYPE_ONETIME_CONFIG("-out_shift",p_cfg->out_shift);
    ARGTYPE_ONETIME_CONFIG("-out_zero_bias",p_cfg->out_zero_bias);
    ARGTYPE_ONETIME_CONFIG("-integer_bits",p_cfg->integer_bits);
    ARGTYPE_ONETIME_CONFIG("-num_elements",p_cfg->num_elements);
    ARGTYPE_ONETIME_CONFIG("-relu_threshold",p_cfg->relu_threshold);
    ARGTYPE_ONETIME_CONFIG("-inp_precision",p_cfg->inp_precision);
    ARGTYPE_ONETIME_CONFIG("-out_precision",p_cfg->out_precision);
    ARGTYPE_ONETIME_CONFIG("-activation_min",p_cfg->activation_min);
    ARGTYPE_ONETIME_CONFIG("-activation_max",p_cfg->activation_max);
    ARGTYPE_ONETIME_CONFIG_F32("-activation_min_f32",p_cfg->activation_min_f32);
    ARGTYPE_ONETIME_CONFIG_F32("-activation_max_f32",p_cfg->activation_max_f32);
    ARGTYPE_STRING("-activation",p_cfg->activation, MAX_ACTIVATION_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-frames",p_cfg->frames);
    ARGTYPE_ONETIME_CONFIG("-write_file",p_cfg->write_file);
    ARGTYPE_STRING("-read_inp_file_name",p_cfg->read_inp_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name",p_cfg->read_ref_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-write_inp_file_name",p_cfg->write_inp_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-write_out_file_name",p_cfg->write_out_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify",p_cfg->verify);

    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}



#define SIGMOID_ASYM8(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8_asym8\
                (\
                    (unsigned char *) p_out->p,\
                    (unsigned char *) p_inp->p,\
                    cfg.zero_point,\
                    cfg.input_range_radius,\
                    cfg.input_multiplier,\
                    cfg.input_left_shift,\
                    cfg.num_elements\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SIGMOID_ASYM8s(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    (WORD8 *) p_inp->p,\
                    cfg.zero_point,\
                    cfg.input_range_radius,\
                    cfg.input_multiplier,\
                    cfg.input_left_shift,\
                    cfg.num_elements\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SIGMOID_SYM16s(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_sym16s_sym16s\
                (\
                    (WORD16 *) p_out->p,\
                    (WORD16 *) p_inp->p,\
                    cfg.input_multiplier,\
                    cfg.input_left_shift,\
                    cfg.num_elements\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define TANH_ASYM8s(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    (WORD8 *) p_inp->p,\
                    cfg.zero_point,\
                    cfg.input_range_radius,\
                    cfg.input_multiplier,\
                    cfg.input_left_shift,\
                    cfg.num_elements\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define TANH_SYM16s(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_sym16s_sym16s\
                (\
                    (WORD16 *) p_out->p,\
                    (WORD16 *) p_inp->p,\
                    cfg.input_multiplier,\
                    cfg.input_left_shift,\
                    cfg.num_elements\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define ACTIVATION_MIN_MAX_ASYM8U_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_activation_min_max_asym8_asym8 ( \
                (UWORD8 *)p_out->p, (UWORD8 *)p_inp->p, \
                cfg.activation_min, cfg.activation_max, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define RELU_ASYM8U_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_relu_asym8u_asym8u ( \
                (UWORD8 *)p_out->p, (UWORD8 *)p_inp->p, \
                cfg.inp_zero_bias, cfg.out_multiplier, \
                cfg.out_shift, cfg.out_zero_bias, \
                cfg.activation_min, cfg.activation_max, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define RELU_ASYM8S_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_relu_asym8s_asym8s ( \
                (WORD8 *)p_out->p, (WORD8 *)p_inp->p, \
                cfg.inp_zero_bias, cfg.out_multiplier, \
                cfg.out_shift, cfg.out_zero_bias, \
                cfg.activation_min, cfg.activation_max, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define PRELU_ASYM8S_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_prelu_asym8s_asym8s ( \
                (WORD8 *)p_out->p, (WORD8 *)p_inp->p, (WORD8 *)p_inp_alpha->p,\
                cfg.inp_zero_bias, cfg.alpha_zero_bias, \
                cfg.alpha_multiplier, cfg.alpha_shift, \
                cfg.out_multiplier, cfg.out_shift, \
                cfg.out_zero_bias, cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define LEAKY_RELU_ASYM8S_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_leaky_relu_asym8s_asym8s ( \
                (WORD8 *)p_out->p, (WORD8 *)p_inp->p,\
                cfg.inp_zero_bias,\
                cfg.alpha_multiplier, cfg.alpha_shift, \
                cfg.out_multiplier, cfg.out_shift, \
                cfg.out_zero_bias, cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define LEAKY_RELU_ASYM16S_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_leaky_relu_asym16s_asym16s ( \
                (WORD16 *)p_out->p, (WORD16 *)p_inp->p,\
                cfg.inp_zero_bias,\
                cfg.alpha_multiplier, cfg.alpha_shift, \
                cfg.out_multiplier, cfg.out_shift, \
                cfg.out_zero_bias, cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define HSWISH_ASYM8S_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_hard_swish_asym8s_asym8s ( \
                (WORD8 *)p_out->p, (WORD8 *)p_inp->p, \
                cfg.inp_zero_bias, \
                cfg.reluish_multiplier, cfg.reluish_shift, \
                cfg.out_multiplier, cfg.out_shift, \
                cfg.out_zero_bias, cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }
#define SOFTMAX_ASYM8(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8_asym8\
                (\
                    (unsigned char *) p_out->p,\
                    (unsigned char *) p_inp->p,\
                    cfg.diffmin,\
                    cfg.input_left_shift,\
                    cfg.input_multiplier,\
                    cfg.num_elements,\
                    (WORD32 *)p_scratch->p\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SOFTMAX_ASYM8s(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    (WORD8 *) p_inp->p,\
                    cfg.diffmin,\
                    cfg.input_left_shift,\
                    cfg.input_multiplier,\
                    cfg.num_elements,\
                    (WORD32 *)p_scratch->p\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SOFTMAX_ASYM8s_16(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_asym8s_16\
                (\
                    (WORD16 *) p_out->p,\
                    (WORD8 *) p_inp->p,\
                    cfg.diffmin,\
                    cfg.input_left_shift,\
                    cfg.input_multiplier,\
                    cfg.num_elements,\
                    (WORD32 *)p_scratch->p\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SOFTMAX_SYM16s_16(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.activation,#KERNEL) && (IPREC == cfg.inp_precision) && (OPREC == p_out->precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_vec_##KERNEL##_sym16s_16\
                (\
                    (WORD16 *) p_out->p,\
                    (WORD16 *) p_inp->p,\
                    cfg.input_left_shift,\
                    cfg.input_multiplier,\
                    cfg.num_elements \
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#if HIFI_VFPU
#define ACTIVATION_MIN_MAX_FN_F32(IPREC,OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_activation_min_max_f32_f32( \
                (FLOAT32 *)p_out->p, (FLOAT32 *)p_inp->p, \
                cfg.activation_min_f32, cfg.activation_max_f32, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }
#else
#define ACTIVATION_MIN_MAX_FN_F32(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
        printf("unsupported activation\n"); return -1;} 
#endif

#define ACTIVATION_MIN_MAX_FN(IPREC,OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_##IPREC##_##OPREC ( \
                (WORD##OPREC *)p_out->p, (WORD##IPREC *)p_inp->p, \
                cfg.activation_min, \
                cfg.activation_max, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }


#define ACTIVATION_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_##IPREC##_##OPREC ( \
                (WORD##OPREC *)p_out->p, (WORD##IPREC *)p_inp->p, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define ACTIVATION_FN_VAR_QFORMAT(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_##IPREC##_##OPREC ( \
                (WORD##OPREC *)p_out->p, (WORD##IPREC *)p_inp->p, \
                cfg.integer_bits, cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#define RELU_FN(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_##IPREC##_##OPREC ( \
                (WORD##OPREC *)p_out->p, (WORD##IPREC *)p_inp->p, \
                cfg.relu_threshold, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }

#if HIFI_VFPU
#define ACTIVATION_FN_F32(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_f32_f32 ( \
                (FLOAT32 *)p_out->p, (FLOAT32 *)p_inp->p, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }
#else
#define ACTIVATION_FN_F32(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
        printf("unsupported activation\n"); return -1;} 
#endif

#if HIFI_HP_VFPU && hifi5
#define ACTIVATION_FN_F16(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_f16_f16 ( \
                (WORD16 *)p_out->p, (WORD16 *)p_inp->p, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }
#else
#define ACTIVATION_FN_F16(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
        printf("unsupported activation\n"); return -1;} 
#endif

#if HIFI_VFPU
#define RELU_FN_F32(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && (OPREC == p_out->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
      XTPWR_PROFILER_START(0);\
      err = xa_nn_vec_##ACTIVATION##_f32_f32 ( \
                (FLOAT32 *)p_out->p, (FLOAT32 *)p_inp->p, \
                cfg.relu_threshold, \
                cfg.num_elements);\
      XTPWR_PROFILER_STOP(0);\
    }
#else
#define RELU_FN_F32(IPREC, OPREC, ACTIVATION) \
    if((IPREC == p_inp->precision) && !strcmp(cfg.activation,#ACTIVATION)) {\
        printf("unsupported activation\n"); return -1;} 
#endif

#define PROCESS_ACTIVATION \
    RELU_FN(16, 16, relu) \
    else RELU_FN(8, 8, relu) \
    else ACTIVATION_FN(8, 8, relu_std) \
    else ACTIVATION_MIN_MAX_FN(8, 8, activation_min_max) \
    else ACTIVATION_FN(16, 16, relu_std) \
    else ACTIVATION_MIN_MAX_FN(16, 16, activation_min_max) \
    else ACTIVATION_FN(32, 16, sigmoid) \
    else ACTIVATION_FN(32, 16, tanh) \
    else ACTIVATION_FN(32, 8, sigmoid) \
    else ACTIVATION_FN(32, 8, tanh) \
    else ACTIVATION_FN(16, 16, sigmoid) \
    else ACTIVATION_FN_VAR_QFORMAT(16, 16, tanh) \
    else ACTIVATION_FN_F32(-1, -1, sigmoid) \
    else ACTIVATION_FN_F32(-1, -1, tanh) \
    else ACTIVATION_FN_F16(-2, -2, sigmoid) \
    else ACTIVATION_FN_F16(-2, -2, tanh) \
    else RELU_FN_F32(-1, -1, relu) \
    else ACTIVATION_FN_F32(-1, -1, relu1) \
    else ACTIVATION_FN_F32(-1, -1, relu6) \
    else ACTIVATION_FN_F32(-1, -1, relu_std) \
    else ACTIVATION_MIN_MAX_FN_F32(-1, -1, activation_min_max) \
    else ACTIVATION_FN_F32(-1, -1, softmax) \
    else ACTIVATION_MIN_MAX_ASYM8U_FN(-3, -3, activation_min_max) \
    else RELU_ASYM8U_FN(-3, -3, relu)\
    else RELU_ASYM8S_FN(-4, -4, relu)\
    else PRELU_ASYM8S_FN(-4, -4, prelu)\
    else LEAKY_RELU_ASYM8S_FN(-4, -4, leaky_relu)\
    else LEAKY_RELU_ASYM16S_FN(-7, -7, leaky_relu)\
    else HSWISH_ASYM8S_FN(-4, -4, hard_swish)\
    else SOFTMAX_ASYM8(softmax, -3, -3) \
    else SOFTMAX_ASYM8s(softmax, -4, -4) \
    else SOFTMAX_ASYM8s_16(softmax, -4, 16) \
    else SOFTMAX_SYM16s_16(softmax, -8, 16) \
    else SIGMOID_ASYM8(sigmoid, -3, -3) \
    else SIGMOID_ASYM8s(sigmoid, -4, -4) \
    else SIGMOID_SYM16s(sigmoid, -8, -8) \
    else TANH_ASYM8s(tanh, -4, -4) \
    else TANH_SYM16s(tanh, -8, -8) \
    else {  printf("unsupported activation\n"); return -1;} 


int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 

  test_config_t cfg;

  buf1D_t *p_inp;
  buf1D_t *p_inp_alpha = NULL;
  buf1D_t *p_out;
  buf1D_t *ptr_ref;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref;
  buf1D_t *p_scratch;
  int scratch_size;

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

  // Update threshold for relu based on input precision
  if((cfg.inp_precision == 16) && (cfg.relu_threshold == (1<<15)))
  {
    cfg.relu_threshold = (1<<7);
  }
  else if((cfg.inp_precision == 8) && (cfg.relu_threshold == (1<<15)))
  {
    cfg.relu_threshold = (1<<3);
  }

  // Set profiler name 
  if((cfg.inp_precision == -1) || (cfg.out_precision == -1))
  {
    sprintf(profiler_name, "%s_f32xf32", cfg.activation);
    
    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else if ((cfg.inp_precision == -2) || (cfg.out_precision == -2))
  {
    sprintf(profiler_name, "%s_f16xf16", cfg.activation);
    
    // If VFPU is not supported, return
    if(!HIFI_HP_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else if((cfg.inp_precision == -3) && (cfg.out_precision == -3))
  {
    sprintf(profiler_name, "%s_asym8xasym8", cfg.activation);
  }
  else if((cfg.inp_precision == -4) && (cfg.out_precision == -4))
  {
    sprintf(profiler_name, "%s_asym8sxasym8s", cfg.activation);
  }
  else if((cfg.inp_precision == -7) && (cfg.out_precision == -7))
  {
    sprintf(profiler_name, "%s_asym16sxasym16s", cfg.activation);
  }
  else if((cfg.inp_precision == -8) && (cfg.out_precision == -8))
  {
    sprintf(profiler_name, "%s_sym16sxsym16s", cfg.activation);
  }
  else if((cfg.inp_precision == -4) && (cfg.out_precision == 16))
  {
    sprintf(profiler_name, "%s_asym8sx16", cfg.activation);
  }
  else if((cfg.inp_precision == -8) && (cfg.out_precision == 16))
  {
    sprintf(profiler_name, "%s_sym16sx16", cfg.activation);
  }
  else
  {
    sprintf(profiler_name, "%s_%dx%d", cfg.activation, cfg.inp_precision, cfg.out_precision);
  }

  // Set profiler parameters
  sprintf(profiler_params, "N=%d", cfg.num_elements);
  
  

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
    ptr_ref =  create_buf1D(cfg.num_elements, cfg.out_precision); 
    
    fptr_ref = file_open(pb_ref_file_path, cfg.read_ref_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Allocate Memory
  p_inp = create_buf1D(cfg.num_elements, cfg.inp_precision); VALIDATE_PTR(p_inp);
  p_out = create_buf1D(cfg.num_elements, cfg.out_precision); VALIDATE_PTR(p_out);

  if(!strcmp(cfg.activation,"prelu"))
  {
    p_inp_alpha = create_buf1D(cfg.num_elements, cfg.inp_precision); VALIDATE_PTR(p_inp_alpha);
  }

  if(!strcmp(cfg.activation,"softmax") && (cfg.inp_precision == -3 || cfg.inp_precision == -4) && (cfg.out_precision == -3 || cfg.out_precision == -4 || cfg.out_precision == 16))
  {
      scratch_size = get_softmax_scratch_size(cfg.inp_precision, cfg.out_precision, cfg.num_elements);
      p_scratch = create_buf1D(scratch_size, 8); VALIDATE_PTR(p_scratch);
  }
  
  
  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, cfg.num_elements, "cyc/point", 0);

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    load_activation_input_data(cfg.write_file, fptr_inp, p_inp, p_inp_alpha, cfg.activation);

    // Call the activation specified on command line
    PROCESS_ACTIVATION;

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

  fclose(fptr_inp);
  fclose(fptr_out);

  // Free all buffers
  free_buf1D(p_inp);
  free_buf1D(p_out);

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



