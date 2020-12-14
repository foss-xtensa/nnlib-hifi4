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

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_KERNEL_NAME_LENGTH 20
#define SCRATCH_SIZE_BYTES         2048*8 //TBD: if not reqd, remove

#define XA_MAX_CMD_LINE_LENGTH 1024
#define XA_MAX_ARGS 100
#define PARAMFILE "paramfilesimple_basic.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{
  int  help;
#if 1  // asym8(data type) specific parameters
  int  output_zero_bias;
  int  output_left_shift;
  int  output_multiplier;
  int  output_activation_min;
  int  output_activation_max;
  int  input1_zero_bias;
  int  input1_left_shift;
  int  input1_multiplier;
  int  input2_zero_bias;
  int  input2_left_shift;
  int  input2_multiplier;
  int  left_shift;
#endif
  int  io_length;
  int  vec_count;
  int  frames;
  int  inp_precision;
  int  out_precision;
  int  write_file;
  char kernel_name[MAX_KERNEL_NAME_LENGTH];
  char read_inp1_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_inp2_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_ref_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp1_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp2_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_out_file_name[XA_MAX_CMD_LINE_LENGTH];
  int  verify;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  { 
    p_cfg->help     = 0;
    p_cfg->output_zero_bias = 127;
    p_cfg->output_left_shift = 1;
    p_cfg->output_multiplier = 0x7fff;
    p_cfg->output_activation_min = 0;
    p_cfg->output_activation_max = 255;
    p_cfg->input1_zero_bias = -127;
    p_cfg->input1_left_shift = 0;
    p_cfg->input1_multiplier = 0x7fff;
    p_cfg->input2_zero_bias = -127;
    p_cfg->input2_left_shift = 0;
    p_cfg->input2_multiplier = 0x7fff;
    p_cfg->left_shift = 0;
    p_cfg->io_length  = 1024;
    p_cfg->vec_count  = 1;
    p_cfg->frames   = 2;  
    p_cfg->inp_precision = -1;
    p_cfg->out_precision = -1;
    strcpy(p_cfg->kernel_name, "elm_add");
    p_cfg->write_file = 0;  
    p_cfg->read_inp1_file_name[0] = '\0';
    p_cfg->read_inp2_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp1_file_name[0]='\0';
    p_cfg->write_inp2_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;

    return 0;
  }
  else
  {
    return -1;
  }
}


void parse_arguments(int argc, char** argv, test_config_t *p_cfg)
{
  int argidx;
  for (argidx=1; argidx<argc; argidx++)
  {
    if(strncmp((argv[argidx]), "-", 1) != 0)
    {
      //err_code = 0;
      printf("Invalid argument: %s\n",argv[argidx]);
      exit(1);
    }
    ARGTYPE_INDICATE("--help", p_cfg->help);
    ARGTYPE_INDICATE("-help", p_cfg->help);
    ARGTYPE_INDICATE("-h", p_cfg->help);
    ARGTYPE_ONETIME_CONFIG("-output_zero_bias", p_cfg->output_zero_bias);                       
    ARGTYPE_ONETIME_CONFIG("-output_left_shift", p_cfg->output_left_shift);                        
    ARGTYPE_ONETIME_CONFIG("-output_multiplier", p_cfg->output_multiplier);                
    ARGTYPE_ONETIME_CONFIG("-output_activation_min", p_cfg->output_activation_min);            
    ARGTYPE_ONETIME_CONFIG("-output_activation_max", p_cfg->output_activation_max);            
    ARGTYPE_ONETIME_CONFIG("-input1_zero_bias", p_cfg->input1_zero_bias);                    
    ARGTYPE_ONETIME_CONFIG("-input1_left_shift", p_cfg->input1_left_shift);                     
    ARGTYPE_ONETIME_CONFIG("-input1_multiplier", p_cfg->input1_multiplier);                
    ARGTYPE_ONETIME_CONFIG("-input2_zero_bias", p_cfg->input2_zero_bias);                    
    ARGTYPE_ONETIME_CONFIG("-input2_left_shift", p_cfg->input2_left_shift);                     
    ARGTYPE_ONETIME_CONFIG("-input2_multiplier", p_cfg->input2_multiplier);                
    ARGTYPE_ONETIME_CONFIG("-left_shift", p_cfg->left_shift);                           
    ARGTYPE_ONETIME_CONFIG("-io_length", p_cfg->io_length);                           
    ARGTYPE_ONETIME_CONFIG("-inp_precision", p_cfg->inp_precision);                        
    ARGTYPE_ONETIME_CONFIG("-out_precision", p_cfg->out_precision);                        
    ARGTYPE_ONETIME_CONFIG("-vec_count", p_cfg->vec_count);                           
    ARGTYPE_ONETIME_CONFIG("-frames", p_cfg->frames);
    ARGTYPE_STRING("-kernel_name", p_cfg->kernel_name, MAX_KERNEL_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-write_file", p_cfg->write_file);
    ARGTYPE_STRING("-read_inp1_file_name", p_cfg->read_inp1_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-read_inp2_file_name", p_cfg->read_inp2_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name", p_cfg->read_ref_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_inp1_file_name", p_cfg->write_inp1_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_inp2_file_name", p_cfg->write_inp2_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_STRING("-write_out_file_name", p_cfg->write_out_file_name, XA_MAX_CMD_LINE_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify", p_cfg->verify);
    
    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n", argv[argidx]);
    exit(1);
  }
}

void show_usage(void)
{
    printf ("Usage xt-run <binary> [Options]\n");
    printf("\t-io_length: input/output vector length; Default=1024\n");
    printf("\t-inp_precision: -4 (asym8s) -3 (asym8u),  -1 (single prec float); Default=-1\n");
    printf("\t-out_precision: -4 (asym8s) -3 (asym8u),  -1 (single prec float); Default=-1\n");
    printf("\t-vec_count: number of input vectors; Default=1\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: elm_add, elm_sub, elm_mul, elm_mul_acc, elm_div, elm_floor, dot_prod; Default=""elem_add""\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp1_file_name: Full filename for reading inputs (order - inp) \n");
    printf("\t-read_inp2_file_name: Full filename for reading inputs (order - inp) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp1_file_name: Full filename for writing inputs (order - inp) \n");
    printf("\t-write_inp2_file_name: Full filename for writing inputs (order - inp) \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
    printf("\t =====================================\n ");
    printf("\t ===== ASYM8 specific parameters =====\n ");
    printf("\t =====================================\n ");
    printf ("\t-output_zero_bias: output zero_bias; Default=127\n");
    printf ("\t-output_left_shift: output_left_shift;   Default=1\n");
    printf ("\t-output_multiplier: output_multiplier; Default=0x7fff\n");
    printf ("\t-output_activation_min: output_activation_min; Default=0\n");
    printf ("\t-output_activation_max: output_activation_max; Default=225\n");
    printf ("\t-input1_zero_bias: input1_zero_bias(Only needed in add_asym8); Default=-127\n");
    printf ("\t-input1_left_shift: input1_left_shift(Only needed in add_asym8); Default=0\n");
    printf ("\t-input1_multiplier: input1_multiplier(Only needed in add_asym8); Default=0x7fff\n");
    printf ("\t-input2_zero_bias: input2_zero_bias(Only needed in add_asym8); Default=-127\n");
    printf ("\t-input2_left_shift: input2_left_shift(Only needed in add_asym8); Default=0\n");          
    printf ("\t-input2_multiplier: input2_multiplier(Only needed in add_asym8); Default=0x7fff\n");   
    printf ("\t-left_shift: global left_shift(Only needed in add_asym8); Default=0\n");
}

#define DOT_PROD_OUT_ASYM8S(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_##IPREC##x##IPREC##_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    (WORD16 *) p_inp1->p,\
                    (WORD16 *) p_inp2->p,\
                    0,\
                    cfg.io_length,\
                    cfg.output_multiplier,\
                    cfg.output_left_shift,\
                    cfg.output_zero_bias,\
                    cfg.vec_count\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define BASIC_FLOAT32(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_f32xf32_f32\
                (\
                    (FLOAT32 *) p_out->p,\
                    (FLOAT32 *) p_inp1->p,\
                    (FLOAT32 *) p_inp2->p,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define FLOOR_F32(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_f32_f32\
                (\
                    (FLOAT32 *) p_out->p,\
                    (FLOAT32 *) p_inp1->p,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define ADD_ASYM8(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8xasym8_asym8\
                (\
                    (unsigned char *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (unsigned char *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    cfg.input1_left_shift,\
                    cfg.input1_multiplier,\
                    (unsigned char *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.input2_left_shift,\
                    cfg.input2_multiplier,\
                    cfg.left_shift,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }
#define ADD_ASYM8S(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8sxasym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (WORD8 *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    cfg.input1_left_shift,\
                    cfg.input1_multiplier,\
                    (WORD8 *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.input2_left_shift,\
                    cfg.input2_multiplier,\
                    cfg.left_shift,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SUB_ASYM8(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8xasym8_asym8\
                (\
                    (unsigned char *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (unsigned char *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    cfg.input1_left_shift,\
                    cfg.input1_multiplier,\
                    (unsigned char *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.input2_left_shift,\
                    cfg.input2_multiplier,\
                    cfg.left_shift,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define SUB_ASYM8S(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8sxasym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (WORD8 *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    cfg.input1_left_shift,\
                    cfg.input1_multiplier,\
                    (WORD8 *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.input2_left_shift,\
                    cfg.input2_multiplier,\
                    cfg.left_shift,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define MUL_ASYM8(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8xasym8_asym8\
                (\
                    (unsigned char *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (unsigned char *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    (unsigned char *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }

#define MUL_ASYM8S(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name, #KERNEL) && (IPREC == cfg.inp_precision) \
     && (OPREC == cfg.out_precision)) {\
    XTPWR_PROFILER_START(0);\
        err = xa_nn_##KERNEL##_asym8sxasym8s_asym8s\
                (\
                    (WORD8 *) p_out->p,\
                    cfg.output_zero_bias,\
                    cfg.output_left_shift,\
                    cfg.output_multiplier,\
                    cfg.output_activation_min,\
                    cfg.output_activation_max,\
                    (WORD8 *) p_inp1->p,\
                    cfg.input1_zero_bias,\
                    (WORD8 *) p_inp2->p,\
                    cfg.input2_zero_bias,\
                    cfg.io_length\
                );\
    XTPWR_PROFILER_STOP(0);\
  }


#if HIFI_VFPU
#define PROCESS_BASIC_FUNC \
    BASIC_FLOAT32(elm_mul, -1, -1) \
    else BASIC_FLOAT32(elm_add, -1, -1) \
    else BASIC_FLOAT32(elm_sub, -1, -1) \
    else BASIC_FLOAT32(elm_mul, -1, -1) \
    else BASIC_FLOAT32(elm_mul_acc, -1, -1) \
    else BASIC_FLOAT32(elm_div, -1, -1) \
    else FLOOR_F32(elm_floor, -1, -1) \
    else MUL_ASYM8(elm_mul, -3, -3) \
    else MUL_ASYM8S(elm_mul, -4, -4) \
    else ADD_ASYM8(elm_add, -3, -3) \
    else ADD_ASYM8S(elm_add, -4, -4) \
    else SUB_ASYM8(elm_sub, -3, -3) \
    else SUB_ASYM8S(elm_sub, -4, -4) \
    else DOT_PROD_OUT_ASYM8S(dot_prod, 16, -4) \
    else {  printf("unsupported basic operation\n"); return -1;}
#else
#define PROCESS_BASIC_FUNC \
    MUL_ASYM8(elm_mul, -3, -3) \
    else MUL_ASYM8S(elm_mul, -4, -4) \
    else ADD_ASYM8(elm_add, -3, -3) \
    else ADD_ASYM8S(elm_add, -4, -4) \
    else SUB_ASYM8(elm_sub, -3, -3) \
    else SUB_ASYM8S(elm_sub, -4, -4) \
    else DOT_PROD_OUT_ASYM8S(dot_prod, 16, -4) \
    else {  printf("unsupported basic operation\n"); return -1;}
#endif

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 

  test_config_t cfg;

  buf1D_t *p_inp1;
  buf1D_t *p_inp2;
  buf1D_t *p_out;
  buf1D_t *ptr_ref;

  FILE *fptr_inp1;
  FILE *fptr_inp2;
  FILE *fptr_out;
  FILE *fptr_ref;

  if(default_config(&cfg))
  {
    return -1;
  }
  
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


  // Set profiler name 
  if(cfg.inp_precision == -1)
  {
    sprintf(profiler_name, "%s_f32", cfg.kernel_name);
    
    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else if(cfg.inp_precision == -3)
  {
    sprintf(profiler_name, "%s_asym8", cfg.kernel_name);
  }
  else if(cfg.inp_precision == -4 && cfg.out_precision == -4)
  {
    sprintf(profiler_name, "%s_asym8s", cfg.kernel_name);
  }
  else if(cfg.inp_precision == 16 && cfg.out_precision == -4)
  {
    sprintf(profiler_name, "%s_16x16_asym8s", cfg.kernel_name);
  }
  else
  {
      printf("Precision not supported\n");
      return -1;
  }

  // Set profiler parameters
  sprintf(profiler_params, "N=%d\n", cfg.io_length);

  // Open input file
  if(cfg.write_file)
  {
    /* If write_file (generate test vectors) is enabled, random data would be generated and
       used; the input data and output data generated would be written into files. 
     */
    fptr_inp1 = file_open(pb_input_file_path, cfg.write_inp1_file_name, "wb", XA_MAX_CMD_LINE_LENGTH);
    fptr_inp2 = file_open(pb_input_file_path, cfg.write_inp2_file_name, "wb", XA_MAX_CMD_LINE_LENGTH);
  }
  else
  {
    /* Else, if input file is specified on command line, input data would be read from it, else
       input data would be read from the default file set in default_config().
     */
    fptr_inp1 = file_open(pb_input_file_path, cfg.read_inp1_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
    fptr_inp2 = file_open(pb_input_file_path, cfg.read_inp2_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Open output file
  fptr_out = file_open(pb_output_file_path, cfg.write_out_file_name, "wb", XA_MAX_CMD_LINE_LENGTH);

  // Open reference file if verify flag is enabled
  if(cfg.verify)
  {
    if(strcmp(cfg.kernel_name, "dot_prod") == 0)
    {
      ptr_ref =  create_buf1D(cfg.vec_count, cfg.out_precision); 
    }
    else
    {
      ptr_ref =  create_buf1D(cfg.io_length * cfg.vec_count, cfg.out_precision); 
    }

    fptr_ref = file_open(pb_ref_file_path, cfg.read_ref_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Allocate Memory
  p_inp1 = create_buf1D(cfg.io_length * cfg.vec_count, cfg.inp_precision); VALIDATE_PTR(p_inp1);
  p_inp2 = create_buf1D(cfg.io_length * cfg.vec_count, cfg.inp_precision); VALIDATE_PTR(p_inp2);
  if(strcmp(cfg.kernel_name, "dot_prod") == 0)
  {
    p_out = create_buf1D(cfg.vec_count, cfg.out_precision); VALIDATE_PTR(p_out);
  }
  else
  {
    p_out = create_buf1D(cfg.io_length * cfg.vec_count, cfg.out_precision); VALIDATE_PTR(p_out);
  }

  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, cfg.io_length * cfg.vec_count, "cyc/point", 0);

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    // load_activation_input_data(cfg.write_file, fptr_inp, p_inp);
    err = load_basic_func_data(cfg.write_file, fptr_inp1, fptr_inp2, p_inp1, p_inp2);

    // Call the activation specified on command line
    PROCESS_BASIC_FUNC
    
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

  XTPWR_PROFILER_CLOSE(0, (pass_count == cfg.frames));

  fclose(fptr_inp1);
  fclose(fptr_inp2);
  fclose(fptr_out);

  // Free all buffers
  free_buf1D(p_inp1);
  free_buf1D(p_inp2);
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


