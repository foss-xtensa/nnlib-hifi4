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
#define PARAMFILE "paramfilesimple_pool.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);

enum DATA_FORMAT{
    NHWC=0,
    CHW, // This is same as WHD 
    };

#define CHW_TO_HWC(inp, out, height, width, channels){\
        int c, h, w;\
        for(c = 0; c<channels; c++)\
        {\
            for(h=0; h<height; h++)\
            {\
                for(w=0; w<width; w++)\
                {\
                    out[h*(width*channels) + (w*channels) + c] = inp[c*(height*width) + (h*width) + w];\
                }\
            }\
        }\
}


char pb_input_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_output_file_path[XA_MAX_CMD_LINE_LENGTH] = "";
char pb_ref_file_path[XA_MAX_CMD_LINE_LENGTH] = "";

typedef struct _test_config_t
{

  int help;
  int inp_data_format; // 0 for nhwc; 1 for chw/dhw
  int input_height;
  int input_width;
  int input_channels;
  int kernel_height;
  int kernel_width;
  int x_stride;
  int y_stride;
  int x_padding;
  int y_padding;
  int out_height;
  int out_width;
  int acc_shift;
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
}test_config_t;

int default_config(test_config_t *p_cfg)
{
  if(p_cfg)
  { 

    p_cfg->help     = 0;
    p_cfg->inp_data_format = 1;
    p_cfg->input_height = 16;
    p_cfg->input_width = 16;
    p_cfg->input_channels = 4;
    p_cfg->kernel_height = 3;
    p_cfg->kernel_width = 3;
    p_cfg->x_stride = 2;
    p_cfg->y_stride = 2;
    p_cfg->x_padding = 2;
    p_cfg->y_padding = 2;
    p_cfg->out_height = 16;
    p_cfg->out_width = 16;
    p_cfg->acc_shift = -7;
    p_cfg->out_data_format = 1;
    p_cfg->inp_precision = 16;
    p_cfg->out_precision = 16;
    strcpy(p_cfg->kernel_name, "avgpool");
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
    printf("\t-inp_data_format: data format of input and output, 0 for nhwc, 1 for dhw/chw; Default=1\n");
    printf("\t-input_height: input height; Default=16\n");
    printf("\t-input_width: input width; Default=16\n");
    printf("\t-input_channels: input channels; Default=4\n");
    printf("\t-kernel_height: kernel height; Default=3\n");
    printf("\t-kernel_width: kernel width; Default=3\n");
    printf("\t-x_stride: stride in width dimension; Default=2\n");
    printf("\t-y_stride: stride in height dimension; Default=2\n");
    printf("\t-x_padding: left padding in width dimension; Default=2\n");
    printf("\t-y_padding: top padding in width dimension; Default=2\n");
    printf("\t-out_height: output height; Default=16\n");
    printf("\t-out_width: output width; Default=16\n");
    printf("\t-acc_shift: accumulator left shift; Default=-7\n");
    printf("\t-out_data_format: data format; Default=1 (WHD)\n");
    printf("\t-inp_precision: 8, 16, -1(single prec float); Default=16\n");
    printf("\t-out_precision: 8, 16, -1(single prec float); Default=16\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: avgpool, maxpool; Default=""avgpool""\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - inp) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - inp) \n");
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
      printf("Invalid argument: %s\n",argv[argidx]);
      show_usage();
      exit(1);
    }
    ARGTYPE_INDICATE("--help", p_cfg->help);
    ARGTYPE_INDICATE("-help", p_cfg->help);
    ARGTYPE_INDICATE("-h", p_cfg->help);
    ARGTYPE_ONETIME_CONFIG("-inp_data_format",p_cfg->inp_data_format);
    ARGTYPE_ONETIME_CONFIG("-input_height",p_cfg->input_height);
    ARGTYPE_ONETIME_CONFIG("-input_width",p_cfg->input_width);
    ARGTYPE_ONETIME_CONFIG("-input_channels",p_cfg->input_channels);
    ARGTYPE_ONETIME_CONFIG("-kernel_height",p_cfg->kernel_height);
    ARGTYPE_ONETIME_CONFIG("-kernel_width",p_cfg->kernel_width);
    ARGTYPE_ONETIME_CONFIG("-x_stride",p_cfg->x_stride);
    ARGTYPE_ONETIME_CONFIG("-y_stride",p_cfg->y_stride);
    ARGTYPE_ONETIME_CONFIG("-x_padding",p_cfg->x_padding);
    ARGTYPE_ONETIME_CONFIG("-y_padding",p_cfg->y_padding);
    ARGTYPE_ONETIME_CONFIG("-out_height",p_cfg->out_height);
    ARGTYPE_ONETIME_CONFIG("-out_width",p_cfg->out_width);
    ARGTYPE_ONETIME_CONFIG("-acc_shift",p_cfg->acc_shift);
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
    
    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}



#define AVGPOOL_KERNEL_F_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define AVGPOOL_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC ( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define MAXPOOL_KERNEL_F_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_f32 ( \
        (FLOAT32 *)p_out->p, (FLOAT32 *) p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define MAXPOOL_KERNEL_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_##IPREC ( \
        (WORD##OPREC *)p_out->p, (WORD##IPREC *) p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#define POOL_KERNEL_ASYM8_FN(KERNEL, IPREC, OPREC) \
  if(!strcmp(cfg.kernel_name,#KERNEL) && (IPREC == p_inp->precision)) {\
    XTPWR_PROFILER_START(0);\
    err = xa_nn_##KERNEL##_asym8( \
        (UWORD8 *)p_out->p, (UWORD8 *)p_inp->p, \
        cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, \
        cfg.x_stride, cfg.y_stride, cfg.x_padding, cfg.y_padding, cfg.out_height, cfg.out_width, \
        cfg.inp_data_format, cfg.out_data_format, p_scratch);\
    XTPWR_PROFILER_STOP(0);\
  }

#if HIFI_VFPU
#define PROCESS_POOL \
    AVGPOOL_KERNEL_FN(avgpool, 16, 16) \
    else AVGPOOL_KERNEL_FN(avgpool, 8, 8) \
    else AVGPOOL_KERNEL_F_FN(avgpool, -1, -1) \
    else MAXPOOL_KERNEL_FN(maxpool, 8, 8) \
    else MAXPOOL_KERNEL_FN(maxpool, 16, 16) \
    else MAXPOOL_KERNEL_F_FN(maxpool, -1, -1) \
    else POOL_KERNEL_ASYM8_FN(maxpool, -3, -3) \
    else POOL_KERNEL_ASYM8_FN(avgpool, -3, -3) \
    else {  printf("unsupported pooling operation\n"); return -1;}
#else
#define PROCESS_POOL \
    AVGPOOL_KERNEL_FN(avgpool, 16, 16) \
    else AVGPOOL_KERNEL_FN(avgpool, 8, 8) \
    else MAXPOOL_KERNEL_FN(maxpool, 8, 8) \
    else MAXPOOL_KERNEL_FN(maxpool, 16, 16) \
    else POOL_KERNEL_ASYM8_FN(maxpool, -3, -3) \
    else POOL_KERNEL_ASYM8_FN(avgpool, -3, -3) \
    else {  printf("unsupported pooling operation\n"); return -1;}
#endif

int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 
  void *p_scratch;
  int inp_size, out_size;
  int num_ops=0;

  test_config_t cfg;

  buf1D_t *p_inp;
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

  inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
  out_size = cfg.out_height * cfg.out_width * cfg.input_channels;

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
  else
  {
    sprintf(profiler_params, "_%d", 
        cfg.inp_precision);
    strcat(profiler_name, profiler_params);
  }

  if(cfg.inp_data_format == NHWC)
  {
    sprintf(profiler_params, "_nhwc");
    strcat(profiler_name, profiler_params);
  }
  
  // Set profiler parameters
  sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, kernel_width=%d, out_height=%d, out_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_height, cfg.out_width);

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

  if(!strcmp(cfg.kernel_name,"avgpool"))
    num_ops = out_size * (1 + cfg.kernel_height * cfg.kernel_width);
  else if(!strcmp(cfg.kernel_name,"maxpool"))
    num_ops = out_size * cfg.kernel_height * cfg.kernel_width;

  XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, num_ops, "OPs/cyc", 1);

  // Init
  WORD32 scratch_size = 0;

  // Get persistent size and allocate 
  if(!strcmp(cfg.kernel_name,"avgpool"))
  {
      scratch_size = xa_nn_avgpool_getsize(cfg.input_channels
              ,cfg.inp_precision
              ,cfg.out_precision
              ,cfg.input_height
              ,cfg.input_width
              ,cfg.kernel_height
              ,cfg.kernel_width
              ,cfg.x_stride
              ,cfg.y_stride
              ,cfg.x_padding
              ,cfg.y_padding
              ,cfg.out_height
              ,cfg.out_width
              ,cfg.inp_data_format
              ,cfg.out_data_format);
  }
  else if(!strcmp(cfg.kernel_name,"maxpool"))
  {
      scratch_size = xa_nn_maxpool_getsize(cfg.input_channels
              ,cfg.inp_precision
              ,cfg.out_precision
              ,cfg.input_height
              ,cfg.input_width
              ,cfg.kernel_height
              ,cfg.kernel_width
              ,cfg.x_stride
              ,cfg.y_stride
              ,cfg.x_padding
              ,cfg.y_padding
              ,cfg.out_height
              ,cfg.out_width
              ,cfg.inp_data_format
              ,cfg.out_data_format);
  }

  PRINT_VAR(scratch_size)

  p_scratch = (xa_nnlib_handle_t)malloc(scratch_size); PRINT_PTR(p_scratch)

  fprintf(stdout, "\nScratch size: %d bytes\n", scratch_size);

  // Frame processing loop
  for(frame = 0; frame < cfg.frames; frame++)
  {
    // If write_file enabled, generate random data for input, else read from file
    load_pool_input_data(cfg.write_file, fptr_inp, p_inp);
    
    // Call the cnn kernel_name specified on command line
    PROCESS_POOL;

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

  free(p_scratch);

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


