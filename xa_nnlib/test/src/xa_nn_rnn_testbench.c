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

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_FILE_NAME_LENGTH 256
#define MAX_RNN_NAME_LENGTH 30

#define XA_MAX_CMD_LINE_LENGTH 300
#define XA_MAX_ARGS 30
#define PARAMFILE "paramfilesimple_rnn.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{
  lstm_quant_params quant_params;
  lstm_flags flags;
  gru_quant_params gru_q_params;
  int time_major;
  int inp_size;
  int n_itr;
  int n_batch;
  int n_cell;
  int hidden_size;
  int help;
  int ker_precision;
  int io_precision;
  int cell_precision;
  char kernel_name[MAX_RNN_NAME_LENGTH];
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
    p_cfg->ker_precision = -5;
    p_cfg->io_precision = -4;
    p_cfg->cell_precision = 16;

    /* quantization parameters initialization */
    p_cfg->quant_params.ig_W_out_multiplier = 0x40000000;
    p_cfg->quant_params.fg_W_out_multiplier = 0x40000000;
    p_cfg->quant_params.cg_W_out_multiplier = 0x40000000;
    p_cfg->quant_params.og_W_out_multiplier = 0x40000000;
    p_cfg->quant_params.ig_U_out_multiplier = 0x40000000;
    p_cfg->quant_params.fg_U_out_multiplier = 0x40000000;
    p_cfg->quant_params.cg_U_out_multiplier = 0x40000000;
    p_cfg->quant_params.og_U_out_multiplier = 0x40000000;

    p_cfg->quant_params.ig_W_out_shift = -4;
    p_cfg->quant_params.fg_W_out_shift = -4;
    p_cfg->quant_params.cg_W_out_shift = -4;
    p_cfg->quant_params.og_W_out_shift = -4;
    p_cfg->quant_params.ig_U_out_shift = -4;
    p_cfg->quant_params.fg_U_out_shift = -4;
    p_cfg->quant_params.cg_U_out_shift = -4;
    p_cfg->quant_params.og_U_out_shift = -4;

    p_cfg->quant_params.quantized_cell_clip = 32767;
    p_cfg->quant_params.cell_state_scale = -12;
    p_cfg->quant_params.hidden_multiplier = 0x40000000;
    p_cfg->quant_params.hidden_shift = -9;
    p_cfg->quant_params.input_zero_bias = 0;
    p_cfg->quant_params.hidden_zero_bias = 0;
    
    p_cfg->flags.use_cifg = 0;
    p_cfg->flags.time_major = 0;

    p_cfg->inp_size = 128;
    p_cfg->n_itr = 64;
    p_cfg->n_batch = 16;
    p_cfg->n_cell = 96;
    p_cfg->hidden_size = 96;
    
    p_cfg->gru_q_params.ug_W_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.rg_W_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.ms_W_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.ug_U_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.rg_U_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.ms_U_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.rg_fcU_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.ug_ms_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.ug_hidden_out_multiplier = 0x40000000;
    p_cfg->gru_q_params.hidden_multiplier = 0x40000000;
    
    p_cfg->gru_q_params.ug_W_out_shift = -4;
    p_cfg->gru_q_params.rg_W_out_shift = -4;
    p_cfg->gru_q_params.ms_W_out_shift = -4;
    p_cfg->gru_q_params.ug_U_out_shift = -4;
    p_cfg->gru_q_params.rg_U_out_shift = -4;
    p_cfg->gru_q_params.ms_U_out_shift = -4;
    p_cfg->gru_q_params.rg_fcU_out_shift = -4;
    p_cfg->gru_q_params.ug_ms_out_shift = -4;
    p_cfg->gru_q_params.ug_hidden_out_shift = -4;
    p_cfg->gru_q_params.hidden_shift = -4;
    
    p_cfg->gru_q_params.hidden_zero_bias = 0;
    p_cfg->gru_q_params.input_zero_bias = 0;
    
    p_cfg->time_major = 0;

    strcpy(p_cfg->kernel_name,"lstm");
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


void parse_arguments(int argc, char** argv, test_config_t *p_cfg)
{
  int argidx;
  for (argidx=1;argidx<argc;argidx++)
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
    ARGTYPE_STRING("-kernel_name",p_cfg->kernel_name, MAX_RNN_NAME_LENGTH);
    if(strcmp(p_cfg->kernel_name,"lstm") == 0)
    {
      ARGTYPE_ONETIME_CONFIG("-input_zero_bias",p_cfg->quant_params.input_zero_bias);
      ARGTYPE_ONETIME_CONFIG("-hidden_zero_bias",p_cfg->quant_params.hidden_zero_bias);
      ARGTYPE_ONETIME_CONFIG("-hidden_multiplier",p_cfg->quant_params.hidden_multiplier);
      ARGTYPE_ONETIME_CONFIG("-hidden_shift",p_cfg->quant_params.hidden_shift);
      ARGTYPE_ONETIME_CONFIG("-quantized_cell_clip",p_cfg->quant_params.quantized_cell_clip);
      ARGTYPE_ONETIME_CONFIG("-cell_state_scale",p_cfg->quant_params.cell_state_scale);
      ARGTYPE_ONETIME_CONFIG("-time_major",p_cfg->flags.time_major);
    }
    else
    {
      ARGTYPE_ONETIME_CONFIG("-input_zero_bias",p_cfg->gru_q_params.input_zero_bias);
      ARGTYPE_ONETIME_CONFIG("-hidden_zero_bias",p_cfg->gru_q_params.hidden_zero_bias);
      ARGTYPE_ONETIME_CONFIG("-hidden_multiplier",p_cfg->gru_q_params.hidden_multiplier);
      ARGTYPE_ONETIME_CONFIG("-hidden_shift",p_cfg->gru_q_params.hidden_shift);
      ARGTYPE_ONETIME_CONFIG("-time_major",p_cfg->time_major);
    }
    ARGTYPE_ONETIME_CONFIG("-inp_size",p_cfg->inp_size);
    ARGTYPE_ONETIME_CONFIG("-n_itr",p_cfg->n_itr);
    ARGTYPE_ONETIME_CONFIG("-n_batch",p_cfg->n_batch);
    ARGTYPE_ONETIME_CONFIG("-n_cell",p_cfg->n_cell);
    ARGTYPE_ONETIME_CONFIG("-hidden_size",p_cfg->hidden_size);
    ARGTYPE_ONETIME_CONFIG("-ker_precision",p_cfg->ker_precision);
    ARGTYPE_ONETIME_CONFIG("-io_precision",p_cfg->io_precision);
    ARGTYPE_ONETIME_CONFIG("-cell_precision",p_cfg->cell_precision);
    ARGTYPE_ONETIME_CONFIG("-frames",p_cfg->frames);
    ARGTYPE_ONETIME_CONFIG("-write_file",p_cfg->write_file);
    ARGTYPE_STRING("-read_inp_file_name",p_cfg->read_inp_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-read_ref_file_name",p_cfg->read_ref_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-write_inp_file_name",p_cfg->write_inp_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_STRING("-write_out_file_name",p_cfg->write_out_file_name, MAX_FILE_NAME_LENGTH);
    ARGTYPE_ONETIME_CONFIG("-verify",p_cfg->verify);

    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    exit(1);
  }
}

void show_usage(void)
{
    printf ("Usage xt-run <binary> [Options]\n");
    printf("\t-ker_precision : -5; Default=-5\n");
    printf("\t-io_precision : -4; Default-4\n");
    printf("\t-cell_precision : 16; Default=16\n");
    printf("\t-input_zero_bias : Input zero point; Default=0\n");
    printf("\t-hidden_zero_bias: Hidden layer zero-point; Default=0\n");
    printf("\t-hidden_shift: Hidden Layer shift;   Default=0\n");
    printf("\t-hidden_multiplier: Hidden layer multiplier; Default=0x40000000\n");
    printf("\t-quantized_cell_clip: Clip value for quantized cell; Default=32767\n");
    printf("\t-cell_state_scale: Cell state scale; Default=-12\n");
    printf("\t-inp_size: Number of features in input; Default=128\n");
    printf("\t-n_itr: Number of time iterations; Default=64\n");
    printf("\t-n_batch: Number of elements in batch dimension; Default=16\n");
    printf("\t-n_cell: Number of elements in cell state; Default=96\n");
    printf("\t-hidden_size: Number of elements in hidden state; Default=96\n");
    printf("\t-time_major: Order of input and output 1: time is outer most dimension, 0: batch is outer most dimension Default=0\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: lstm; Default=lstm\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading input \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing input \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
}

#define LSTM_8X8_16(KPREC, IOPREC, CPREC, KERNEL) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == cfg.ker_precision) && (IOPREC == cfg.io_precision) && (CPREC == cfg.cell_precision)) { \
    XTPWR_PROFILER_START(0); \
        err = xa_nn_lstm_sym8sxasym8s_16 \
              (\
                (WORD8 *)p_out->p, \
                (WORD8 *)p_hidden->p, \
                (WORD16 *)p_cell->p, \
                &lstm_weights, \
                &lstm_biases, \
                (WORD8 *)p_inp->p,\
                cfg.inp_size, \
                cfg.n_cell, \
                cfg.n_cell, \
                cfg.n_batch, \
                cfg.n_itr, \
                cfg.n_cell, \
                &cfg.quant_params, \
                &cfg.flags, \
                p_scratch->p \
              ); \
    XTPWR_PROFILER_STOP(0); \
  }
  
#define GRU_8X8(KPREC, IOPREC, KERNEL) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (KPREC == cfg.ker_precision) && (IOPREC == cfg.io_precision)) { \
    XTPWR_PROFILER_START(0); \
        err = xa_nn_gru_sym8sxasym8s \
              (\
                (WORD8 *)p_out->p, \
                (WORD8 *)p_hidden->p, \
                &gru_weights, \
                &gru_biases, \
                (WORD8 *)p_inp->p,\
                cfg.inp_size, \
                cfg.hidden_size, \
                cfg.hidden_size, \
                cfg.n_batch, \
                cfg.n_itr, \
                &cfg.gru_q_params, \
                cfg.time_major, \
                p_scratch->p \
              ); \
    XTPWR_PROFILER_STOP(0); \
  }

#define PROCESS_RNN \
    LSTM_8X8_16(-5, -4, 16, lstm) \
    else GRU_8X8(-5, -4, gru) \
    else {  printf("unsupported RNN kernel\n"); return -1;}

int xa_nn_main_process(int argc, char *argv[])
{
  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];

  test_config_t cfg;

  buf1D_t *p_out;
  buf1D_t *p_hidden;
  buf1D_t *p_cell;
  buf1D_t *p_inp;
  buf1D_t *ptr_ref;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref;

  buf1D_t *p_scratch;
  int scratch_size;



  /*Declare variables for FC, peephole and layer norm buffers*/
  lstm_weights_ptrs lstm_weights;
  lstm_bias_ptrs lstm_biases;
  buf1D_t *p_ig_W;
  buf1D_t *p_fg_W;
  buf1D_t *p_cg_W;
  buf1D_t *p_og_W;
  buf1D_t *p_ig_U;
  buf1D_t *p_fg_U;
  buf1D_t *p_cg_U;
  buf1D_t *p_og_U;

  buf1D_t *p_ig_W_bias;
  buf1D_t *p_fg_W_bias;
  buf1D_t *p_cg_W_bias;
  buf1D_t *p_og_W_bias;
  
  /* declare gru weigth buffers */
  gru_weights_ptrs gru_weights;
  gru_bias_ptrs gru_biases;
  buf1D_t *p_ug_W;
  buf1D_t *p_rg_W;
  buf1D_t *p_ms_W;
  buf1D_t *p_ug_U;
  buf1D_t *p_rg_U;
  buf1D_t *p_ms_U;

  buf1D_t *p_ug_W_bias;
  buf1D_t *p_rg_W_bias;
  buf1D_t *p_ms_W_bias;
  
  buf1D_t *p_ug_U_bias;
  buf1D_t *p_rg_U_bias;
  buf1D_t *p_ms_U_bias;

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
#if defined(USE_HIFI_ACT_TIE) && (defined(AE_SIGMOID16X4X2) || defined(AE_SIGMOID16X4) || defined(AE_TANH16X4X2) || defined(AE_TANH16X4))
  char *ext=".bin";
  char *dot_add = strstr(cfg.read_ref_file_name, ext);
  char* new_ext="_act_tie.bin";
  strcpy(dot_add, new_ext);
#endif
  // Set profiler name
  if(strcmp(cfg.kernel_name,"lstm") == 0 && cfg.ker_precision == -5 && cfg.io_precision == -4 && cfg.cell_precision == 16)
    sprintf(profiler_name, "%s_sym8sxasym8s_%d", cfg.kernel_name, cfg.cell_precision);
  else if(strcmp(cfg.kernel_name,"gru") == 0 && cfg.ker_precision == -5 && cfg.io_precision == -4)
    sprintf(profiler_name, "%s_sym8sxasym8s", cfg.kernel_name);

  // Set profiler parameters
  if(strcmp(cfg.kernel_name,"gru") == 0)
  {
    sprintf(profiler_params, "inp_size=%d, n_itr=%d, n_batch=%d, hidden_size=%d",
          cfg.inp_size, cfg.n_itr, cfg.n_batch, cfg.hidden_size);
  }  
  else
  {
    sprintf(profiler_params, "inp_size=%d, n_itr=%d, n_batch=%d, n_cell=%d",
          cfg.inp_size, cfg.n_itr, cfg.n_batch, cfg.n_cell);
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
    if(strcmp(cfg.kernel_name,"gru") == 0)
      ptr_ref =  create_buf1D((cfg.n_batch * cfg.n_itr * cfg.hidden_size), cfg.io_precision);
    else
      ptr_ref =  create_buf1D((cfg.n_batch * cfg.n_itr * cfg.n_cell), cfg.io_precision);

    fptr_ref = file_open(pb_ref_file_path, cfg.read_ref_file_name, "rb", XA_MAX_CMD_LINE_LENGTH);
  }

  // Allocate Memory
  p_inp = create_buf1D((cfg.n_itr * cfg.n_batch * cfg.inp_size), cfg.io_precision); VALIDATE_PTR(p_inp);
  if(strcmp(cfg.kernel_name,"gru") == 0)
  {
    p_out = create_buf1D((cfg.n_itr * cfg.n_batch * cfg.hidden_size), cfg.io_precision); VALIDATE_PTR(p_out);
    p_hidden = create_buf1D((cfg.n_batch * cfg.hidden_size), cfg.io_precision); VALIDATE_PTR(p_hidden);
    
    p_ug_W = create_buf1D(cfg.hidden_size * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_ug_W);
    p_rg_W = create_buf1D(cfg.hidden_size * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_rg_W);
    p_ms_W = create_buf1D(cfg.hidden_size * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_ms_W);
    p_ug_U = create_buf1D(cfg.hidden_size * cfg.hidden_size, cfg.ker_precision);    VALIDATE_PTR(p_ug_U);
    p_rg_U = create_buf1D(cfg.hidden_size * cfg.hidden_size, cfg.ker_precision);    VALIDATE_PTR(p_rg_U);
    p_ms_U = create_buf1D(cfg.hidden_size * cfg.hidden_size, cfg.ker_precision);    VALIDATE_PTR(p_ms_U);

    p_ug_W_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_ug_W_bias);
    p_rg_W_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_rg_W_bias);
    p_ms_W_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_ms_W_bias);
    p_ug_U_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_ug_U_bias);
    p_rg_U_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_rg_U_bias);
    p_ms_U_bias = create_buf1D(cfg.hidden_size, 32);   VALIDATE_PTR(p_ms_U_bias);

    gru_weights.p_ug_W = (WORD8 *)(p_ug_W->p);
    gru_weights.p_rg_W = (WORD8 *)(p_rg_W->p);
    gru_weights.p_ms_W = (WORD8 *)(p_ms_W->p);
    gru_weights.p_ug_U = (WORD8 *)(p_ug_U->p);
    gru_weights.p_rg_U = (WORD8 *)(p_rg_U->p);
    gru_weights.p_ms_U = (WORD8 *)(p_ms_U->p);

    gru_biases.p_ug_W_bias = (WORD32 *)(p_ug_W_bias->p);
    gru_biases.p_rg_W_bias = (WORD32 *)(p_rg_W_bias->p);
    gru_biases.p_ms_W_bias = (WORD32 *)(p_ms_W_bias->p);
    gru_biases.p_ug_U_bias = (WORD32 *)(p_ug_U_bias->p);
    gru_biases.p_rg_U_bias = (WORD32 *)(p_rg_U_bias->p);
    gru_biases.p_ms_U_bias = (WORD32 *)(p_ms_U_bias->p);

    scratch_size = xa_nn_gru_getsize(cfg.n_batch, cfg.n_itr, cfg.hidden_size, cfg.io_precision);
    p_scratch = create_buf1D(scratch_size, 8); VALIDATE_PTR(p_scratch);
  }
  else
  {
    p_out = create_buf1D((cfg.n_itr * cfg.n_batch * cfg.n_cell), cfg.io_precision); VALIDATE_PTR(p_out);
    p_hidden = create_buf1D((cfg.n_batch * cfg.n_cell), cfg.io_precision); VALIDATE_PTR(p_hidden);
    p_cell = create_buf1D((cfg.n_batch * cfg.n_cell), cfg.cell_precision); VALIDATE_PTR(p_cell);
    
    p_ig_W = create_buf1D(cfg.n_cell * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_ig_W);
    p_fg_W = create_buf1D(cfg.n_cell * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_fg_W);
    p_cg_W = create_buf1D(cfg.n_cell * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_cg_W);
    p_og_W = create_buf1D(cfg.n_cell * cfg.inp_size, cfg.ker_precision);  VALIDATE_PTR(p_og_W);
    p_ig_U = create_buf1D(cfg.n_cell * cfg.n_cell, cfg.ker_precision);    VALIDATE_PTR(p_ig_U);
    p_fg_U = create_buf1D(cfg.n_cell * cfg.n_cell, cfg.ker_precision);    VALIDATE_PTR(p_fg_U);
    p_cg_U = create_buf1D(cfg.n_cell * cfg.n_cell, cfg.ker_precision);    VALIDATE_PTR(p_cg_U);
    p_og_U = create_buf1D(cfg.n_cell * cfg.n_cell, cfg.ker_precision);    VALIDATE_PTR(p_og_U);

    p_ig_W_bias = create_buf1D(cfg.n_cell, 32);   VALIDATE_PTR(p_ig_W_bias);
    p_fg_W_bias = create_buf1D(cfg.n_cell, 32);   VALIDATE_PTR(p_fg_W_bias);
    p_cg_W_bias = create_buf1D(cfg.n_cell, 32);   VALIDATE_PTR(p_cg_W_bias);
    p_og_W_bias = create_buf1D(cfg.n_cell, 32);   VALIDATE_PTR(p_og_W_bias);

    lstm_weights.p_ig_W = (WORD8 *)(p_ig_W->p);
    lstm_weights.p_fg_W = (WORD8 *)(p_fg_W->p);
    lstm_weights.p_cg_W = (WORD8 *)(p_cg_W->p);
    lstm_weights.p_og_W = (WORD8 *)(p_og_W->p);
    lstm_weights.p_ig_U = (WORD8 *)(p_ig_U->p);
    lstm_weights.p_fg_U = (WORD8 *)(p_fg_U->p);
    lstm_weights.p_cg_U = (WORD8 *)(p_cg_U->p);
    lstm_weights.p_og_U = (WORD8 *)(p_og_U->p);

    lstm_biases.p_ig_W_bias = (WORD32 *)(p_ig_W_bias->p);
    lstm_biases.p_fg_W_bias = (WORD32 *)(p_fg_W_bias->p);
    lstm_biases.p_cg_W_bias = (WORD32 *)(p_cg_W_bias->p);
    lstm_biases.p_og_W_bias = (WORD32 *)(p_og_W_bias->p);

    lstm_biases.p_ig_U_bias = NULL;
    lstm_biases.p_fg_U_bias = NULL;
    lstm_biases.p_cg_U_bias = NULL;
    lstm_biases.p_og_U_bias = NULL;

    scratch_size = xa_nn_lstm_getsize(cfg.n_batch, cfg.n_itr, cfg.n_cell, cfg.cell_precision);
    p_scratch = create_buf1D(scratch_size, 8); VALIDATE_PTR(p_scratch);
  }

  if(strcmp(cfg.kernel_name,"lstm") == 0)
  {
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, 4 * (cfg.n_itr * cfg.n_batch * cfg.n_cell) * (cfg.n_cell + cfg.inp_size), "MAC/cyc", 1);
  }
  else if(strcmp(cfg.kernel_name,"gru") == 0)
  {  
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, 3 * (cfg.n_itr * cfg.n_batch * cfg.hidden_size) * (cfg.hidden_size + cfg.inp_size), "MAC/cyc", 1);
  }
  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    if(strcmp(cfg.kernel_name,"gru") == 0)
    {
      load_gru_input_data(cfg.write_file, fptr_inp, p_inp, p_hidden, 
          p_ug_W, p_rg_W, p_ms_W, p_ug_U, p_rg_U, p_ms_U, 
          p_ug_W_bias, p_rg_W_bias, p_ms_W_bias,
          p_ug_U_bias, p_rg_U_bias, p_ms_U_bias);
    }
    else
    {
      load_rnn_input_data(cfg.write_file, fptr_inp, p_inp, p_hidden, p_cell,
          p_ig_W, p_fg_W, p_cg_W, p_og_W, p_ig_U, p_fg_U, p_cg_U, p_og_U,
          p_ig_W_bias, p_fg_W_bias, p_cg_W_bias, p_og_W_bias);
    }
    // Call the kernel_name specified on command line
    PROCESS_RNN;

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
      pass_count += compare_buf1D(ptr_ref, p_out, cfg.verify, cfg.io_precision, 1);
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
  free_buf1D(p_out);
  free_buf1D(p_hidden);
  free_buf1D(p_cell);
  free_buf1D(p_inp);
  if(strcmp(cfg.kernel_name,"gru") == 0)
  {
    free_buf1D(p_ug_W);
    free_buf1D(p_rg_W);
    free_buf1D(p_ms_W);
    free_buf1D(p_ug_W);
    free_buf1D(p_ug_U);
    free_buf1D(p_rg_U);
    free_buf1D(p_ms_U);
    free_buf1D(p_ug_W_bias);
    free_buf1D(p_rg_W_bias);
    free_buf1D(p_ms_W_bias);
    free_buf1D(p_ug_U_bias);
    free_buf1D(p_rg_U_bias);
    free_buf1D(p_ms_U_bias);
  }
  else
  {
    free_buf1D(p_ig_W);
    free_buf1D(p_fg_W);
    free_buf1D(p_cg_W);
    free_buf1D(p_og_W);
    free_buf1D(p_ig_U);
    free_buf1D(p_fg_U);
    free_buf1D(p_cg_U);
    free_buf1D(p_og_U);
    free_buf1D(p_ig_W_bias);
    free_buf1D(p_fg_W_bias);
    free_buf1D(p_cg_W_bias);
    free_buf1D(p_og_W_bias);
  }
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



