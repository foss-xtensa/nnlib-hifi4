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
#include "nnlib/xa_nnlib_cnn_api.h"
#include "xt_manage_buffers.h"
#include "cmdline_parser.h"
#include "file_io.h"
#include "xa_nnlib_standards.h"

#define PROF_ALLOCATE
#include "xt_profiler.h"

#define MAX_KERNEL_NAME_LENGTH 40

#define XA_MAX_CMD_LINE_LENGTH 200
#define XA_MAX_ARGS 100
#define PARAMFILE "paramfilesimple_cnn.txt"

#define VALIDATE_PTR(ptr) if(NULL == ptr) { printf("%s: allocation failed\n", #ptr); return -1;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);

#define FILL_IO_SHAPE_CUBE(shape, nheight, nwidth, ndepth, type)  \
{                                                                 \
  shape.shape_type = type;                                        \
  shape.dim.cube.height = nheight;                                \
  shape.dim.cube.width = nwidth;                                  \
  shape.dim.cube.depth = ndepth;                                  \
  if (type == SHAPE_CUBE_DWH_T)                                   \
  {                                                               \
    shape.dim.cube.depth_offset = 1;                              \
    shape.dim.cube.height_offset = ndepth*nwidth;                 \
    shape.dim.cube.width_offset = ndepth;                         \
  }                                                               \
  else if(type == SHAPE_CUBE_WHD_T)                               \
  {                                                               \
    shape.dim.cube.depth_offset = nwidth*nheight;                 \
    shape.dim.cube.height_offset = nwidth;                        \
    shape.dim.cube.width_offset = 1;                              \
  }                                                               \
  shape.n_shapes = 1;                                             \
  shape.shape_offset = -1;                                        \
}

#define FILL_KERNEL_SHAPE_CUBE_DIMS(shape, nheight, nwidth, ndepth, type, nshapes)  \
{                                                                                   \
  shape.shape_type = type;                                                          \
  shape.dim.cube.height = nheight;                                                  \
  shape.dim.cube.width = nwidth;                                                    \
  shape.dim.cube.depth = ndepth;                                                    \
  shape.n_shapes = nshapes;                                                         \
}

#define FILL_KERNEL_SHAPE_CUBE_OFFSETS(shape, nhoffset, nwoffset, ndoffset)         \
{                                                                                   \
  shape.dim.cube.depth_offset = (ndoffset);                                         \
  shape.dim.cube.height_offset = (nhoffset);                                        \
  shape.dim.cube.width_offset = (nwoffset);                                         \
  shape.shape_offset = (nhoffset)*(nwoffset)*(ndoffset);                            \
}

#define FILL_SHAPE_MATRIX(shape, rows_shape, cols_shape)  \
{                                                         \
  shape.shape_type = SHAPE_MATRIX_T;                      \
  shape.dim.matrix.rows = rows_shape;                     \
  shape.dim.matrix.cols = cols_shape;                     \
  shape.dim.matrix.row_offset = cols_shape;               \
  shape.n_shapes = 1;                                     \
  shape.shape_offset = -1;                                \
}

#define FILL_SHAPE_VECTOR(shape, length_shape)  \
{                                               \
  shape.shape_type = SHAPE_VECTOR_T;            \
  shape.dim.vector.length = length_shape;       \
  shape.n_shapes = 1;                           \
  shape.shape_offset = -1;                      \
}

void print_buf3d(void *buf, int height, int width, int depth, int bitwidth)
{
    int i, j, k;

    for (i=0; i<depth; i++)
    {
        for (j=0; j<height; j++)
        {
            for(k=0; k<width; k++)
            {
                if (bitwidth == 16 ) fprintf(stderr, "%8d ", ((short *)buf)[i*height*width + j*width + k]);
                if (bitwidth ==  8 ) fprintf(stderr, "%8d ", ((char*)buf)[i*height*width + j*width + k]);
                if (bitwidth == -1 ) fprintf(stderr, "%8f ", ((float *)buf)[i*height*width + j*width + k]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
}

static inline void error_code_parse(int error_code)
{
  switch (error_code) 
  {
    case XA_NNLIB_FATAL_MEM_ALLOC:
      printf("\nError in memory allocation, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ALGO:
      printf("\nInvalid Algorithm name, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PRECISION:
      printf("\nInvalid Preceision, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHIFT:
      printf("\nInvalid bias shift, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_ACC_SHIFT:
      printf("\nInvalid acc shift, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_STRIDE:
      printf("\nInvalid stride, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PADDING:
      printf("\nInvalid padding, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_BIAS_SHAPE:
      printf("\nInvalid bias shape, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_PARAM_COMBINATION:
      printf("\nInvalid param combination, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_KERNEL_SHAPE:
      printf("\nInvalid kernel shape, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_OUTPUT_SHAPE:
      printf("\nInvalid output shape, Exiting\n");
      break;
    case XA_NNLIB_CNN_CONFIG_FATAL_INVALID_INPUT_SHAPE:
      printf("\nInvalid input shape, Exiting\n");
      break;
    case XA_NNLIB_CNN_EXECUTE_FATAL_INVALID_INPUT_SHAPE:
      printf("\nInput shape mismatch during execution, Exiting\n");
      break;
    default:
      printf("\nUnknown error condition, Exiting\n");
      break;
  }
}

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
  int out_data_format;
  int inp_precision;
  int kernel_precision;
  int out_precision;
  int bias_precision;
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
    p_cfg->input_height = 16;
    p_cfg->input_width = 16;
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
    p_cfg->out_data_format = 0;
    p_cfg->inp_precision = 16;
    p_cfg->kernel_precision = 8;
    p_cfg->out_precision = 16;
    p_cfg->bias_precision = 16;
    strcpy(p_cfg->kernel_name, "conv2d_std");
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
    printf("\t-input_height: input height; Default=16\n");
    printf("\t-input_width: input width; Default=16\n");
    printf("\t-input_channels: input channels; Default=4\n");
    printf("\t-kernel_height: kernel height; Default=3\n");
    printf("\t-kernel_width: kernel width; Default=3\n");
    printf("\t-out_channels: output channels; Default=4\n");
    printf("\t-channels_multiplier: channel multiplier; Default=1\n");
    printf("\t-x_stride: stride in width dimension; Default=2\n");
    printf("\t-y_stride: stride in height dimension; Default=2\n");
    printf("\t-x_padding: left padding in width dimension; Default=2\n");
    printf("\t-y_padding: top padding in height dimension; Default=2\n");
    printf("\t-out_height: output height; Default=16\n");
    printf("\t-out_width: output width; Default=16\n");
    printf("\t-bias_shift: bias left shift; Default=7\n");
    printf("\t-acc_shift: accumulator left shift; Default=-7\n");
    printf("\t-out_data_format: Output data format, 0 (DWH), 1 (WHD); Default=0 (DWH)\n");
    printf("\t-inp_precision: 8, 16, -1(single prec float); Default=16\n");
    printf("\t-kernel_precision: 8, 16, -1(single prec float); Default=8\n");
    printf("\t-out_precision: 8, 16, -1(single prec float); Default=16\n");
    printf("\t-bias_precision: 8, 16, -1(single prec float); Default=16\n");
    printf("\t-frames: Positive number; Default=2\n");
    printf("\t-kernel_name: conv2d_std, conv2d_depth, conv1d_std; Default="" : conv2d_std\n");
    printf("\t-write_file: set to 1 to write input and output vectors to file; Default=0\n");
    printf("\t-read_inp_file_name: Full filename for reading inputs (order - inp, kernel, bias, (kernel_point, bias_point for conv2d_depth kernel)) \n");
    printf("\t-read_ref_file_name: Full filename for reading reference output \n");
    printf("\t-write_inp_file_name: Full filename for writing inputs (order - inp, kernel, bias, (kernel_point, bias_point for conv2d_depth kernel)) \n");
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
    ARGTYPE_ONETIME_CONFIG("-kernel_height",p_cfg->kernel_height);
    ARGTYPE_ONETIME_CONFIG("-kernel_width",p_cfg->kernel_width);
    ARGTYPE_ONETIME_CONFIG("-out_channels",p_cfg->out_channels);
    ARGTYPE_ONETIME_CONFIG("-channels_multiplier",p_cfg->channels_multiplier);
    ARGTYPE_ONETIME_CONFIG("-x_stride",p_cfg->x_stride);
    ARGTYPE_ONETIME_CONFIG("-y_stride",p_cfg->y_stride);
    ARGTYPE_ONETIME_CONFIG("-x_padding",p_cfg->x_padding);
    ARGTYPE_ONETIME_CONFIG("-y_padding",p_cfg->y_padding);
    ARGTYPE_ONETIME_CONFIG("-out_height",p_cfg->out_height);
    ARGTYPE_ONETIME_CONFIG("-out_width",p_cfg->out_width);
    ARGTYPE_ONETIME_CONFIG("-bias_shift",p_cfg->bias_shift);
    ARGTYPE_ONETIME_CONFIG("-acc_shift",p_cfg->acc_shift);
    ARGTYPE_ONETIME_CONFIG("-out_data_format",p_cfg->out_data_format);
    ARGTYPE_ONETIME_CONFIG("-inp_precision",p_cfg->inp_precision);
    ARGTYPE_ONETIME_CONFIG("-kernel_precision",p_cfg->kernel_precision);
    ARGTYPE_ONETIME_CONFIG("-out_precision",p_cfg->out_precision);
    ARGTYPE_ONETIME_CONFIG("-bias_precision",p_cfg->bias_precision);
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

int map_test_cfg_to_cnn_cfg(test_config_t *p_cfg, xa_nnlib_cnn_init_config_t *cnn_cfg)
{

    if (p_cfg && cnn_cfg)
    {
        memset(cnn_cfg, 0, sizeof(xa_nnlib_cnn_init_config_t));

        if      (!strcmp(p_cfg->kernel_name,"conv1d_std"))   cnn_cfg->algo = XA_NNLIB_CNN_CONV1D_STD;  
        else if (!strcmp(p_cfg->kernel_name,"conv2d_std"))   cnn_cfg->algo = XA_NNLIB_CNN_CONV2D_STD;  
        else if (!strcmp(p_cfg->kernel_name,"conv2d_depth")) cnn_cfg->algo = XA_NNLIB_CNN_CONV2D_DS;  

        cnn_cfg->x_stride = p_cfg->x_stride;
        cnn_cfg->y_stride = p_cfg->y_stride;
        cnn_cfg->x_padding = p_cfg->x_padding;
        cnn_cfg->y_padding = p_cfg->y_padding;
        cnn_cfg->channels_multiplier = p_cfg->channels_multiplier;
        
        cnn_cfg->bias_shift = p_cfg->bias_shift;
        cnn_cfg->acc_shift  = p_cfg->acc_shift;
 
        cnn_cfg->output_height   = p_cfg->out_height;
        cnn_cfg->output_width    = p_cfg->out_width;
        cnn_cfg->output_channels = p_cfg->out_channels;
        cnn_cfg->output_format   = p_cfg->out_data_format;
        

        if(!strcmp(p_cfg->kernel_name,"conv1d_std"))
        {   
            int input_channelsXwidth_pad;
            if(p_cfg->inp_precision == -1)
              input_channelsXwidth_pad = (p_cfg->input_channels * p_cfg->input_width + 2 - 1) & ~(2 - 1);
            else
              input_channelsXwidth_pad = (p_cfg->input_channels * p_cfg->input_width + 4 - 1) & ~(4 - 1);
          
            cnn_cfg->output_width    = 1;

            FILL_KERNEL_SHAPE_CUBE_DIMS(cnn_cfg->kernel_std_shape, p_cfg->kernel_height, p_cfg->kernel_width, p_cfg->input_channels, SHAPE_CUBE_DWH_T, p_cfg->out_channels)
            FILL_KERNEL_SHAPE_CUBE_OFFSETS(cnn_cfg->kernel_std_shape, (input_channelsXwidth_pad), (p_cfg->input_channels), (1))
            FILL_SHAPE_VECTOR(cnn_cfg->bias_std_shape, p_cfg->out_channels)
            FILL_IO_SHAPE_CUBE(cnn_cfg->input_shape, p_cfg->input_height, p_cfg->input_width, p_cfg->input_channels, SHAPE_CUBE_DWH_T)
        }
        else if(!strcmp(p_cfg->kernel_name,"conv2d_std"))
        {
            int kernel_channels_pad;
            if(p_cfg->inp_precision == -1)
              kernel_channels_pad = ((p_cfg->input_channels + 1) & (~1));
            else
              kernel_channels_pad = ((p_cfg->input_channels + 3) & (~3));

            FILL_KERNEL_SHAPE_CUBE_DIMS(cnn_cfg->kernel_std_shape, p_cfg->kernel_height, p_cfg->kernel_width, p_cfg->input_channels, SHAPE_CUBE_DWH_T, p_cfg->out_channels)
            FILL_KERNEL_SHAPE_CUBE_OFFSETS(cnn_cfg->kernel_std_shape, (kernel_channels_pad*p_cfg->kernel_width), (kernel_channels_pad), (1))
            FILL_SHAPE_VECTOR(cnn_cfg->bias_std_shape, p_cfg->out_channels)
            FILL_IO_SHAPE_CUBE(cnn_cfg->input_shape, p_cfg->input_height, p_cfg->input_width, p_cfg->input_channels, SHAPE_CUBE_DWH_T)
        }
        else if(!strcmp(p_cfg->kernel_name,"conv2d_depth"))
        {
            int kernel_width_pad = ((p_cfg->kernel_width + 3) & (~3)); 
            FILL_KERNEL_SHAPE_CUBE_DIMS(cnn_cfg->kernel_ds_depth_shape, p_cfg->kernel_height, p_cfg->kernel_width, p_cfg->input_channels*p_cfg->channels_multiplier, SHAPE_CUBE_WHD_T, 1)
            FILL_KERNEL_SHAPE_CUBE_OFFSETS(cnn_cfg->kernel_ds_depth_shape, (kernel_width_pad), (1), (kernel_width_pad*p_cfg->kernel_height))
            FILL_SHAPE_VECTOR(cnn_cfg->bias_ds_depth_shape, p_cfg->input_channels*p_cfg->channels_multiplier) 
            FILL_SHAPE_MATRIX(cnn_cfg->kernel_ds_point_shape, p_cfg->out_channels, p_cfg->input_channels*p_cfg->channels_multiplier)
            FILL_SHAPE_VECTOR(cnn_cfg->bias_ds_point_shape, p_cfg->out_channels)
            FILL_IO_SHAPE_CUBE(cnn_cfg->input_shape, p_cfg->input_height, p_cfg->input_width, p_cfg->input_channels, SHAPE_CUBE_WHD_T)
        }

        if (p_cfg->kernel_precision ==  8 && p_cfg->inp_precision == 16 && p_cfg->out_precision == 16 && p_cfg->bias_precision == 16) 
        {
            cnn_cfg->precision = XA_NNLIB_CNN_8bx16b;
        }
        else if (p_cfg->kernel_precision == 16 && p_cfg->inp_precision == 16 && p_cfg->out_precision == 16 && p_cfg->bias_precision == 16) 
        {
            cnn_cfg->precision = XA_NNLIB_CNN_16bx16b;
        }
        else if (p_cfg->kernel_precision ==  8 && p_cfg->inp_precision ==  8 && p_cfg->out_precision ==  8 && p_cfg->bias_precision ==  8) 
        {
            cnn_cfg->precision = XA_NNLIB_CNN_8bx8b;
        }
        else if (p_cfg->kernel_precision == -1 && p_cfg->inp_precision == -1 && p_cfg->out_precision == -1 && p_cfg->bias_precision == -1) 
        {
            cnn_cfg->precision = XA_NNLIB_CNN_f32xf32;
        }
    }
 
    return 0;
}




int xa_nn_main_process(int argc, char *argv[])
{

  int frame;
  int err = 0;
  int pass_count=0;
  char profiler_name[MAX_PROFILER_NAME_LENGTH]; 
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 
  xa_nnlib_handle_t cnn_handle;
  int inp_size=0, kernel_size, out_size;
  int kernel_size_pad, input_channels_pad, kernel_width_pad;
  int kernel_channels;
  int input_channelsXwidth_pad;
  int kernel_point_size, dw_out_size;
  int bias_size, bias_point_size;
  void *p_scratch;

  test_config_t cfg;
  xa_nnlib_cnn_init_config_t cnn_cfg;

  buf1D_t *p_inp;
  buf2D_t *p_kernel;
  buf1D_t *p_kernel_point;
  buf1D_t *p_bias;
  buf1D_t *p_bias_point;
  buf1D_t *p_dw_out;
  buf1D_t *p_out;
  buf1D_t *p_ref = NULL;

  FILE *fptr_inp;
  FILE *fptr_out;
  FILE *fptr_ref = NULL;

  /* Library name version etc print */
  fprintf(stderr, "\n--------------------------------------------------------\n");
  fprintf(stderr, "%s library version %s\n",
          xa_nnlib_get_lib_name_string(),
          xa_nnlib_get_lib_version_string());
  fprintf(stderr, "API version: %s\n", xa_nnlib_get_lib_api_version_string());
  fprintf(stderr, "Cadence Design Systems, Inc. http://www.cadence.com\n");

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
  map_test_cfg_to_cnn_cfg(&cfg, &cnn_cfg);
 

  if(!strcmp(cfg.kernel_name,"conv2d_std"))
  {
    inp_size = cfg.input_height * cfg.input_width * cfg.input_channels;
    kernel_size = cfg.kernel_height * cfg.kernel_width * cfg.input_channels;
    if(cfg.inp_precision == -1)
    {
      input_channels_pad = (cfg.input_channels + 2 - 1) & ~(2 - 1);
    }
    else
    {
      input_channels_pad = (cfg.input_channels + 4 - 1) & ~(4 - 1);
    }
    kernel_size_pad = cfg.kernel_height * cfg.kernel_width * input_channels_pad;
    out_size = cfg.out_height * cfg.out_width * cfg.out_channels;
    bias_size = cfg.out_channels;
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    inp_size          = cfg.input_channels      * cfg.input_height        * cfg.input_width;
    kernel_size       = cfg.channels_multiplier * cfg.input_channels      * cfg.kernel_height  * cfg.kernel_width;

    kernel_channels = cfg.input_channels * cfg.channels_multiplier;
    kernel_width_pad = cfg.kernel_width;
    kernel_size_pad = cfg.kernel_height * kernel_width_pad * kernel_channels;
    dw_out_size       = cfg.channels_multiplier * cfg.input_channels      * cfg.out_height     * cfg.out_width;
    kernel_point_size = cfg.out_channels        * cfg.channels_multiplier * cfg.input_channels * 1 * 1;
    out_size          = cfg.out_channels        * cfg.out_height          * cfg.out_width;
    bias_size = cfg.channels_multiplier * cfg.input_channels;
    bias_point_size = cfg.out_channels;
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
    out_size = cfg.out_height * cfg.out_channels;
    bias_size = cfg.out_channels;
  }

  // Set profiler name 
  sprintf(profiler_params, "cnn_");
  strcpy(profiler_name, profiler_params);
  if(cfg.kernel_name[0])
  {
    strcat(profiler_name,cfg.kernel_name);
  }
  if((cfg.kernel_precision == -1) || (cfg.inp_precision == -1))
  {
    sprintf(profiler_params, "_f32xf32");
    strcat(profiler_name, profiler_params);
    
    // If VFPU is not supported, return
    if(!HIFI_VFPU)
    {
      printf("%s: NOT TESTED\n", profiler_name);
      return 0;
    }
  }
  else
  {
    sprintf(profiler_params, "_%dx%d", cfg.kernel_precision, cfg.inp_precision);
    strcat(profiler_name, profiler_params);
  }
  
  // Set profiler parameters
  if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, out_channels=%d, out_height=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.out_channels, cfg.out_height);
  }
  else
  {
    sprintf(profiler_params, "input_height=%d, input_width=%d, input_channels=%d, kernel_height=%d, kernel_width=%d, out_channels=%d, out_height=%d, out_width=%d", 
      cfg.input_height, cfg.input_width, cfg.input_channels, cfg.kernel_height, cfg.kernel_width, cfg.out_channels, cfg.out_height, cfg.out_width);
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
  p_inp = create_buf1D(inp_size, cfg.inp_precision);    VALIDATE_PTR(p_inp);
  p_out = create_buf1D(out_size, cfg.out_precision);    VALIDATE_PTR(p_out);
  if(!strcmp(cfg.kernel_name,"conv2d_std"))
  {
    p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height * cfg.kernel_width, cfg.input_channels, input_channels_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, out_size * kernel_size, "MACs/cyc", 1);
  }
  else if(!strcmp(cfg.kernel_name,"conv1d_std"))
  {
    p_kernel = create_buf2D(cfg.out_channels * cfg.kernel_height, cfg.input_width * cfg.input_channels, input_channelsXwidth_pad, cfg.kernel_precision, 0);    VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                            VALIDATE_PTR(p_bias);

    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, out_size * kernel_size, "MACs/cyc", 1);
  }
  else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    p_kernel = create_buf2D(kernel_channels * cfg.kernel_height, cfg.kernel_width, kernel_width_pad, cfg.kernel_precision, 0);            VALIDATE_PTR(p_kernel);
    p_bias = create_buf1D(bias_size, cfg.bias_precision);                                    VALIDATE_PTR(p_bias);
    p_kernel_point = create_buf1D(kernel_point_size, cfg.kernel_precision); VALIDATE_PTR(p_kernel_point);
    p_dw_out = create_buf1D(dw_out_size, cfg.out_precision);                                 VALIDATE_PTR(p_dw_out);
    p_bias_point = create_buf1D(bias_point_size, cfg.bias_precision);                        VALIDATE_PTR(p_bias_point);

    int total_MACS = (
       (cfg.channels_multiplier * cfg.input_channels * cfg.out_height * cfg.out_width * cfg.kernel_height * cfg.kernel_width) + /* MACs in depthwise */
       (cfg.out_channels * cfg.channels_multiplier * cfg.input_channels * cfg.out_height * cfg.out_width * 1 * 1)               /* MACs in pointwise */
       );
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, total_MACS, "MACs/cyc", 1);
  }

  /* Initialization Loop */
  {
    int persistent_size;
    int scratch_size;

    /* Get persistent and scratch sizes and allocate them */
    persistent_size = xa_nnlib_cnn_get_persistent_fast(&cnn_cfg);  
    if(persistent_size < 0)
    {
      error_code_parse(persistent_size);
      return persistent_size;
    }
    scratch_size = xa_nnlib_cnn_get_scratch_fast(&cnn_cfg);   
    if(scratch_size < 0)
    {
      error_code_parse(scratch_size);
      return scratch_size;
    }

    cnn_handle = (xa_nnlib_handle_t)malloc(persistent_size); PRINT_PTR(cnn_handle)
    p_scratch  = malloc(scratch_size);    PRINT_PTR(p_scratch)

    fprintf(stdout, "\nPersistent(fast) size: %8d bytes\n", persistent_size);
    fprintf(stdout, "Scratch(fast) size:    %8d bytes\n", scratch_size);
    if(cfg.inp_precision < 0 && cfg.out_precision < 0)
    {
      // For float32 inp_precision and out_precision are -1
      fprintf(stdout, "Input size:            %8d bytes\n", inp_size*4);
      fprintf(stdout, "Output size:           %8d bytes\n\n", out_size*4);
    }
    else
    {
      fprintf(stdout, "Input size:            %8d bytes\n", inp_size*(cfg.inp_precision>>3));
      fprintf(stdout, "Output size:           %8d bytes\n\n", out_size*(cfg.out_precision>>3));
    }
    /* Initialize CNN Layer with configurations */
    err =xa_nnlib_cnn_init(cnn_handle, &cnn_cfg);

    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    }
  }
  
  /* Set up kernel and bias pointers */
  {
    void *kernel[2];
    void *bias[2];
    
    if(!strcmp(cfg.kernel_name,"conv2d_std") || !strcmp(cfg.kernel_name,"conv1d_std"))
    {
      kernel[0] = p_kernel->p;
      bias[0]   = p_bias->p;
    }
    else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
    {
      kernel[0] = p_kernel->p;
      bias[0]   = p_bias->p;
      kernel[1] = p_kernel_point->p;
      bias[1]   = p_bias_point->p;
    }
    err=xa_nnlib_cnn_set_config(cnn_handle, XA_NNLIB_CNN_KERNEL, &kernel[0]);
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    } 
    err=xa_nnlib_cnn_set_config(cnn_handle, XA_NNLIB_CNN_BIAS, &bias[0]);
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    } 
  }  

  /* Execution Loop */
  {
    xa_nnlib_shape_t output_shape;

    for(frame = 0; frame < cfg.frames; frame++)
    {
      // If write_file enabled, generate random data for input, else read from file
      if(!strcmp(cfg.kernel_name,"conv2d_std"))
        load_conv2d_std_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, cfg.input_channels, input_channels_pad, 0);
      else if(!strcmp(cfg.kernel_name,"conv2d_depth"))
        load_conv2d_ds_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, p_kernel_point, p_bias_point, 0);
      else if(!strcmp(cfg.kernel_name,"conv1d_std"))
        load_conv1d_std_input_data(cfg.write_file, fptr_inp, p_inp, p_kernel, p_bias, cfg.input_channels, cfg.input_width, input_channelsXwidth_pad, 0);

      XTPWR_PROFILER_START(0);
      err = xa_nnlib_cnn_process(cnn_handle, 
                                 p_scratch, 
                                 p_inp->p, 
                                 p_out->p, 
                                 &cnn_cfg.input_shape, 
                                 &output_shape);
      XTPWR_PROFILER_STOP(0);

      if(XA_NNLIB_NO_ERROR != err)
      {
        error_code_parse(err);
        return err;
      }

      XTPWR_PROFILER_UPDATE(0);
      XTPWR_PROFILER_PRINT(0);

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
  }

  XTPWR_PROFILER_CLOSE(0, (pass_count == cfg.frames), cfg.verify);

  fclose(fptr_inp);
  fclose(fptr_out);

  // Free all buffers
  free_buf1D(p_inp);
  free_buf2D(p_kernel);
  free_buf1D(p_bias);
  free_buf1D(p_out);
  if(!strcmp(cfg.kernel_name,"conv2d_depth"))
  {
    free_buf1D(p_kernel_point);
    free_buf1D(p_bias_point);
    free_buf1D(p_dw_out);
  }

  if(cfg.verify)
  {
    fclose(fptr_ref);
    free_buf1D(p_ref);
  }

  free(cnn_handle);
  if (p_scratch) free(p_scratch);

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


