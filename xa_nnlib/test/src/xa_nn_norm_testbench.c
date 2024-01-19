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

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_KERNEL_NAME_LENGTH 20

#define XA_MAX_CMD_LINE_LENGTH 200
#define XA_MAX_ARGS 100
#define PARAMFILE "paramfilesimple_norm.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);

char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{
  int help;
  int num_elms;
  int io_height;
  int io_width;
  int io_channels;
  int out_shift;
  int out_activation_min;
  int out_activation_max;
  int inp_data_format;
  int out_data_format;
  int inp_precision;
  int out_precision;
  char kernel_name[MAX_KERNEL_NAME_LENGTH];
  int frames;
  int write_file;
  char read_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char read_ref_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_inp_file_name[XA_MAX_CMD_LINE_LENGTH];
  char write_out_file_name[XA_MAX_CMD_LINE_LENGTH];
  int verify;
  // quant8 specific parameters
  int zero_point;
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  {

    p_cfg->help     = 0;
    p_cfg->num_elms = 256;
    p_cfg->io_height = 40;
    p_cfg->io_width = 32;
    p_cfg->io_channels = 32;
    p_cfg->out_shift = -16;
    p_cfg->out_activation_min = -128;
    p_cfg->out_activation_max = 127;
    p_cfg->inp_data_format = 0;
    p_cfg->out_data_format = 0;
    p_cfg->inp_precision = 16;
    p_cfg->out_precision = 16;
    strcpy(p_cfg->kernel_name, "l2_norm");
    p_cfg->frames   = 2;
    p_cfg->write_file = 0;
    p_cfg->read_inp_file_name[0] = '\0';
    p_cfg->read_ref_file_name[0] = '\0';
    p_cfg->write_inp_file_name[0]='\0';
    p_cfg->write_out_file_name[0] = '\0';
    p_cfg->verify = 1;
    p_cfg->zero_point = 0;
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
    printf("\t-num_elms: Number of elements; Default=256\n");
    printf("\t-io_height: Height for 3D input/output; Default=40\n");
    printf("\t-io_width: Width for 3D input/output; Default=32\n");
    printf("\t-io_channels: Number of channels for 3D input/output; Default=32\n");
    printf("\t-out_shift: Output shift; Default=-16\n");
    printf("\t-out_activation_min: Output minimum limit; Default=-128 for 8-bit output\n");
    printf("\t-out_activation_min: Output maximum limit; Default=127 for 8-bit output\n");
    printf("\t-inp_data_format: Input data format, 0 : NHWC; Default=0\n");
    printf("\t-out_data_format: Output data format, 0 : NHWC; Default=0\n");
    printf("\t-inp_precision: 8, 16, -1(single prec float), -4(asym8s); Default=16\n");
    printf("\t-out_precision: 8, 16, -1(single prec float), -4(asym8s); Default=16\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: l2_norm; Default=""l2_norm""\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - inp) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - inp) \n");
    printf("\t-write_out_file_name: Full filename for writing output \n");
    printf("\t-verify: Verify output against provided reference; 0: Disable, 1: Bitexact match; Default=1\n");
    printf("\t-zero_point: l2_norm_asym8s input parameter; Default=0\n");
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
    ARGTYPE_ONETIME_CONFIG("-num_elms",p_cfg->num_elms);
    ARGTYPE_ONETIME_CONFIG("-io_height",p_cfg->io_height);
    ARGTYPE_ONETIME_CONFIG("-io_width",p_cfg->io_width);
    ARGTYPE_ONETIME_CONFIG("-io_channels",p_cfg->io_channels);
    ARGTYPE_ONETIME_CONFIG("-out_shift",p_cfg->out_shift);
    ARGTYPE_ONETIME_CONFIG("-out_activation_min",p_cfg->out_activation_min);
    ARGTYPE_ONETIME_CONFIG("-out_activation_max",p_cfg->out_activation_max);
    ARGTYPE_ONETIME_CONFIG("-inp_data_format",p_cfg->inp_data_format);
    ARGTYPE_ONETIME_CONFIG("-out_data_format",p_cfg->out_data_format);
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
    ARGTYPE_ONETIME_CONFIG("-zero_point",p_cfg->zero_point);

    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}



#if HIFI_VFPU
#define L2_NORM_KERNEL_F_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, \
        cfg.num_elms); \
    XTPWR_PROFILER_STOP(0);\
  }
#else
#define L2_NORM_KERNEL_F_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    printf("unsupported normalization operation\n"); return -1;}
#endif

#define L2_NORM_KERNEL_ASYM8S_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_asym8s_asym8s ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, \
        cfg.zero_point,\
        cfg.num_elms); \
    XTPWR_PROFILER_STOP(0);\
  }

#define BATCH_NORM_3D_KERNEL_8_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) { \
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_8_8 ( \
        (WORD8 *)p_out->p, (WORD8 *) p_inp->p, \
        (WORD16 *)p_alpha->p, (WORD32 *)p_beta->p, \
        cfg.io_height, cfg.io_width, cfg.io_channels, cfg.out_shift, \
        cfg.out_activation_min, cfg.out_activation_max, \
        cfg.inp_data_format, cfg.out_data_format); \
    XTPWR_PROFILER_STOP(0); \
  }

#define PROCESS_NORM \
    L2_NORM_KERNEL_F_FN(l2_norm, -1, -1) \
    else L2_NORM_KERNEL_ASYM8S_FN(l2_norm, -4, -4) \
    else BATCH_NORM_3D_KERNEL_8_FN(batch_norm_3D, 8, 8) \
    else {  printf("unsupported normalization operation\n"); return -1;}

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH];
  int inp_size, out_size;
  int num_ops=0;

  test_config_t cfg;

  buf1D_t *p_inp;
  buf1D_t *p_out;
  buf1D_t *p_alpha;
  buf1D_t *p_beta;
  buf1D_t *p_ref = NULL;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref = NULL;

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

  if(!strcmp(cfg.kernel_name, "batch_norm_3D"))
  {
    if(cfg.io_height >= 0 && cfg.io_width >= 0 && cfg.io_channels >= 0)
      inp_size = cfg.io_height * cfg.io_width * cfg.io_channels;
    else
      inp_size = 0;
    out_size = inp_size;
  }
  else
  {
    inp_size = cfg.num_elms;
    out_size = cfg.num_elms;
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
  if(!strcmp(cfg.kernel_name,"batch_norm_3D"))
  {
    sprintf(profiler_params, "io_height=%d, io_width = %d, io_channels = %d, inp_data_format = %d, out_data_format %d",
            cfg.io_height, cfg.io_width, cfg.io_channels, cfg.inp_data_format, cfg.out_data_format);
  }
  else
  {
    sprintf(profiler_params, "num_elms=%d", cfg.num_elms);
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

  if(!strcmp(cfg.kernel_name,"batch_norm_3D"))
  {
    p_alpha = create_buf1D(cfg.io_channels, 16);                                  VALIDATE_PTR(p_alpha);
    p_beta = create_buf1D(cfg.io_channels, 32);                                   VALIDATE_PTR(p_beta);
    /*p_alpha & p_beta arrays are not read through bin files, hence initialized here*/
    memset(p_alpha->p, -20, cfg.io_channels* 2);
    memset(p_beta->p,  -20, cfg.io_channels* 4);
  }

  if(!strcmp(cfg.kernel_name,"l2_norm"))
    num_ops = 2*cfg.num_elms;   // First calculated square root of energy and then divide input by it
  else if(!strcmp(cfg.kernel_name,"batch_norm_3D"))
    num_ops = inp_size;

  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    if(!strcmp(cfg.kernel_name,"batch_norm_3D"))
      load_batch_norm_3D_input_data(cfg.write_file, fptr_inp, p_inp, p_alpha, p_beta);
    else
      load_norm_input_data(cfg.write_file, fptr_inp, p_inp);

    // Call the cnn kernel_name specified on command line
    PROCESS_NORM;

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
