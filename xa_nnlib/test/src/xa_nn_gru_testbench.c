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
#define FILE_IO
#define PROF_ALLOCATE
#define INT16_MAX_ERR 0
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "xa_type_def.h"
#include "xa_nnlib_gru_api.h"
#include "cmdline_parser.h"
#include "xt_profiler.h"
#ifdef hifi4
#define XA_PAD_BYTES    8
#endif

#ifdef hifi5
#define XA_PAD_BYTES    16
#endif

#define XA_MAX_FILE_PATH_LENGTH 200
#define XA_MAX_FILE_NAME_LENGTH  80
#define XA_MAX_FULL_FILE_NAME_LENGTH (XA_MAX_FILE_PATH_LENGTH + XA_MAX_FILE_NAME_LENGTH)
#define XA_MAX_ARGS 30
#define PARAMFILE "paramfilesimple_gru.txt"

char pb_input_file_path[XA_MAX_FILE_PATH_LENGTH] = "";
char pb_output_file_path[XA_MAX_FILE_PATH_LENGTH] = "";
char pb_ref_file_path[XA_MAX_FILE_PATH_LENGTH] = "";
char pb_prev_h_file_path[XA_MAX_FILE_PATH_LENGTH] = "";

#define CHECK_PTR(ptr, context) \
  if(NULL == ptr) {printf("%s: Failed\n", context); return -1;}

#define CHECK_PTR_RETURN_NULL(ptr, context) \
  if(NULL == ptr) {printf("%s: Failed\n", context); return NULL;}

#define PRINT_VAR(var)  // printf("%d: %s = %d\n", __LINE__, #var, (int) var); fflush(stdout); fflush(stderr);
#define PRINT_PTR(ptr)  // printf("%d: %s = %p\n", __LINE__, #ptr, (void *) ptr); fflush(stdout); fflush(stderr);
#define PRINT_STR(str)  // printf("%d: %s\n", __LINE__, str); fflush(stdout); fflush(stderr);

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

#define READ_FILE(fname, fpath, file, buffer, bytes, in_size, out_size, pad, context)               \
{                                                                                                   \
  FILE *fptr;                                                                                       \
  strcpy(fname, fpath);                                                                             \
  strcat(fname, file);                                                                              \
  fptr = fopen(fname,"rb");                                                                         \
    if(NULL == fptr) {printf("%s: Failed\n", context); free(weights_and_biases); return NULL;}      \
  int i, read_b;                                                                                    \
  for(i=0; i<out_size; i++){                                                                        \
    read_b = fread(buffer+i*(in_size+pad), bytes, (size_t)in_size, fptr);                           \
    if(read_b != in_size)                                                                           \
    {                                                                                               \
        printf("File %s has insufficent data\n", fname);                                            \
        free(weights_and_biases);                                                                   \
        return NULL;                                                                                \
    }                                                                                               \
  }                                                                                                 \
  fclose(fptr);                                                                                     \
}

const char *coef_files[12] = 
{
  "/w_z.bin",
  "/u_z.bin",
  "/w_r.bin",
  "/u_r.bin",
  "/w_h.bin",
  "/u_h.bin",
  "/b_z.bin",
  "/b_r.bin",
  "/b_h.bin",
  "/bs_z.bin",
  "/bs_r.bin",
  "/bs_h.bin"
};

static inline void error_code_parse(int error_code)
{
  switch (error_code)
  {
    case XA_NNLIB_FATAL_MEM_ALLOC:
      printf("\nError in memory allocation, Exiting\n");
      break;
    case XA_NNLIB_FATAL_MEM_ALIGN:
      printf("\nError in pointer memory alignment, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IN_FEATS:
      printf("\nInvalid input features, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_OUT_FEATS:
      printf("\nInvalid output features, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_PRECISION:
      printf("\nInvalid precision, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_COEFF_QFORMAT:
      printf("\nInvalid coefficient QFormat, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_IO_QFORMAT:
      printf("\nInvalid input/output QFormat, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_MEMBANK_PADDING:
      printf("\nInvalid memory padding, Exiting\n");
      break;
    case XA_NNLIB_GRU_CONFIG_FATAL_INVALID_SPLIT_BIAS:
      printf("\nInvalid split bias option, Exiting\n");
      break;
    case XA_NNLIB_FATAL_INVALID_SHAPE:
      printf("\nInvalid shape, Exiting\n");
      break;
    case XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_OUTPUT_BUFFER_SPACE:
      printf("\nInsufficient output buffer space, Exiting\n");
      break;
    case XA_NNLIB_GRU_EXECUTE_FATAL_INSUFFICIENT_DATA:
      printf("\nInsufficient data, Exiting\n");
      break;
    default:
      printf("\nUnknown error condition, Exiting\n");
      break;
  }
}

void show_usage(void)
{
  printf("xt-run <binary> [Options]\n");
  printf("--in_feats:        \t Input length (Default=256)                   \t  Range: 4-2048 NOTE:-Input length must be multiple of 4\n");
  printf("--out_feats:       \t Output length (Default=256)                  \t  Range: 4-2048 NOTE:-Output length must be multiple of 4\n");
  printf("--split_bias:      \t Split Bias (Default=0)                       \t  Must be 0 or 1\n");
  printf("--membank_padding: \t Memory bank padding (Default=1)              \t  Must be 0 or 1\n");
  printf("--mat_prec:        \t Coefficient precision (Default=16)           \t  Must be 8 or 16\n");
  printf("--vec_prec:        \t Input precision (Default=16)                 \t  Must be 16\n");
  printf("--verify:          \t Verify output against ref output (Default=1) \t  Supported values: 0:-Disable  1:-Enable\n");
  printf("--input_file:      \t File containing input shape\n");
  printf("--filter_path:     \t Path where file containing filter are stored\n");
  printf("--output_file:     \t File to which output will be written\n");
  printf("--prev_h_file:     \t File containing context data\n");
  printf("--ref_file:        \t File which has ref output\n");
  printf("-h/-help/--help:   \t Prints help\n");
}
     
void *setup_weights_and_biases(xa_nnlib_gru_weights_t *weights,
                 xa_nnlib_gru_biases_t *biases,
                 int in_feats, int out_feats, int pad_flag,
                 char *filter_path,
                 xa_nnlib_gru_precision_t precision,
                 int split_bias)
{
  if(precision == XA_NNLIB_GRU_16bx16b)
  {
      coeff_t *weights_and_biases, *ptr;
      size_t size;
      char coef_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
      int pad = (XA_PAD_BYTES*pad_flag) >> 1;  //Width of mem bank for HiFi4/5

      size  = 3 * (in_feats + pad) * out_feats;
      size += 3 * (out_feats + pad) * out_feats;

      int num_bias = 3;
      if(split_bias == 1){
        num_bias = 6;
      }     
      size += num_bias * out_feats ;

      CHECK_PTR_RETURN_NULL(weights, "Allocation for weights");
      CHECK_PTR_RETURN_NULL(biases, "Allocation for biases");

      weights_and_biases = ptr = malloc(size * sizeof(coeff_t));
      CHECK_PTR_RETURN_NULL(ptr, "Allocation for weights_and_biases");
  
      weights->weights16.w_z = ptr; ptr += (in_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_w_z, out_feats, in_feats)
  
      weights->weights16.u_z = ptr; ptr += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_u_z, out_feats, out_feats)
  
      weights->weights16.w_r = ptr; ptr += (in_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_w_r, out_feats, in_feats)
  
      weights->weights16.u_r = ptr; ptr += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_u_r, out_feats, out_feats)
  
      weights->weights16.w_h = ptr; ptr += (in_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_w_h, out_feats, in_feats)
  
      weights->weights16.u_h = ptr; ptr += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights16.shape_u_h, out_feats, out_feats)
  
      biases->b_z  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_z, out_feats)
  
      biases->b_r  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_r, out_feats)
  
      biases->b_h  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_h, out_feats)

      if(split_bias == 1){
        biases->bs_z  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_z, out_feats)
    
        biases->bs_r  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_r, out_feats)
    
        biases->bs_h  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_h, out_feats)
      }
#ifdef FILE_IO
      // Read from file
      READ_FILE(coef_file_name, filter_path, coef_files[0], weights->weights16.w_z, 2, in_feats  , out_feats, pad, "Allocation for w_z")
      READ_FILE(coef_file_name, filter_path, coef_files[1], weights->weights16.u_z, 2, out_feats , out_feats, pad, "Allocation for u_z")
      READ_FILE(coef_file_name, filter_path, coef_files[2], weights->weights16.w_r, 2, in_feats  , out_feats, pad, "Allocation for w_r")
      READ_FILE(coef_file_name, filter_path, coef_files[3], weights->weights16.u_r, 2, out_feats , out_feats, pad, "Allocation for u_r")
      READ_FILE(coef_file_name, filter_path, coef_files[4], weights->weights16.w_h, 2, in_feats  , out_feats, pad, "Allocation for w_h")
      READ_FILE(coef_file_name, filter_path, coef_files[5], weights->weights16.u_h, 2, out_feats , out_feats, pad, "Allocation for u_h")
      READ_FILE(coef_file_name, filter_path, coef_files[6], biases->b_z , 2,  out_feats, 1, 0     , "Allocation for b_z")
      READ_FILE(coef_file_name, filter_path, coef_files[7], biases->b_r , 2,  out_feats, 1, 0     , "Allocation for b_r")
      READ_FILE(coef_file_name, filter_path, coef_files[8], biases->b_h , 2,  out_feats, 1, 0     , "Allocation for b_h")
      if(split_bias == 1){
        READ_FILE(coef_file_name, filter_path, coef_files[9], biases->bs_z , 2,  out_feats, 1, 0     , "Allocation for bs_z")
        READ_FILE(coef_file_name, filter_path, coef_files[10], biases->bs_r , 2,  out_feats, 1, 0     , "Allocation for bs_r")
        READ_FILE(coef_file_name, filter_path, coef_files[11], biases->bs_h , 2,  out_feats, 1, 0     , "Allocation for bs_h")
      }
#else
      // Generate random data for weights
      coeff_t *random = weights_and_biases;

      for(i=0;i<size;i++)
      {
        random[i] = rand();
      }
#endif

      return weights_and_biases;
      // If not, allocate memory (single allocation)
  }
  else if(precision == XA_NNLIB_GRU_8bx16b)
  {
      coeff8_t *weights_and_biases, *ptr8;
      coeff_t *ptr;
      size_t size, size_b;
      char coef_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
      int pad = XA_PAD_BYTES*pad_flag;  //Width of mem bank for HiFi4/5

      size   = 3 * (in_feats + pad) * out_feats;
      size  += 3 * (out_feats + pad) * out_feats;
      
      int num_bias = 3;
      if(split_bias == 1){
        num_bias = 6;
      }
      size_b = num_bias * out_feats ;

      CHECK_PTR_RETURN_NULL(weights, "Allocation for weights");
      CHECK_PTR_RETURN_NULL(biases, "Allocation for biases");
  
      ptr = malloc(size * sizeof(coeff8_t)+ size_b * sizeof(coeff_t));
      weights_and_biases = (coeff8_t *)ptr;
      CHECK_PTR_RETURN_NULL(ptr, "Allocation for weights_and_biases");
  
      biases->b_z  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_z, out_feats)
  
      biases->b_r  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_r, out_feats)
  
      biases->b_h  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_h, out_feats)

      if(split_bias == 1){
        biases->bs_z  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_z, out_feats)
    
        biases->bs_r  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_r, out_feats)
    
        biases->bs_h  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_h, out_feats)
      }
      ptr8 = (coeff8_t*)ptr;

      weights->weights8.w_z = ptr8; ptr8 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_w_z, out_feats, in_feats)
  
      weights->weights8.u_z = ptr8; ptr8 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_u_z, out_feats, out_feats)
  
      weights->weights8.w_r = ptr8; ptr8 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_w_r, out_feats, in_feats)
  
      weights->weights8.u_r = ptr8; ptr8 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_u_r, out_feats, out_feats)
  
      weights->weights8.w_h = ptr8; ptr8 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_w_h, out_feats, in_feats)
  
      weights->weights8.u_h = ptr8; ptr8 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weights8.shape_u_h, out_feats, out_feats)

#ifdef FILE_IO
      // Read from file
  
      READ_FILE(coef_file_name, filter_path, coef_files[0], weights->weights8.w_z, 1, in_feats,  out_feats, pad, "Allocation for w_z")
      READ_FILE(coef_file_name, filter_path, coef_files[1], weights->weights8.u_z, 1, out_feats, out_feats, pad, "Allocation for u_z")
      READ_FILE(coef_file_name, filter_path, coef_files[2], weights->weights8.w_r, 1, in_feats,  out_feats, pad, "Allocation for w_r")
      READ_FILE(coef_file_name, filter_path, coef_files[3], weights->weights8.u_r, 1, out_feats, out_feats, pad, "Allocation for u_r")
      READ_FILE(coef_file_name, filter_path, coef_files[4], weights->weights8.w_h, 1, in_feats,  out_feats, pad, "Allocation for w_h")
      READ_FILE(coef_file_name, filter_path, coef_files[5], weights->weights8.u_h, 1, out_feats, out_feats, pad, "Allocation for u_h")
      READ_FILE(coef_file_name, filter_path, coef_files[6], biases->b_z , 2,  out_feats, 1, 0     , "Allocation for b_z")
      READ_FILE(coef_file_name, filter_path, coef_files[7], biases->b_r , 2,  out_feats, 1, 0     , "Allocation for b_r")
      READ_FILE(coef_file_name, filter_path, coef_files[8], biases->b_h , 2,  out_feats, 1, 0     , "Allocation for b_h")
      if(split_bias == 1){
        READ_FILE(coef_file_name, filter_path, coef_files[9], biases->bs_z , 2,  out_feats, 1, 0     , "Allocation for bs_z")
        READ_FILE(coef_file_name, filter_path, coef_files[10], biases->bs_r , 2,  out_feats, 1, 0     , "Allocation for bs_r")
        READ_FILE(coef_file_name, filter_path, coef_files[11], biases->bs_h , 2,  out_feats, 1, 0     , "Allocation for bs_h")    
      }  
#else
      // Generate random data for weights
      coeff_t *random = weights_and_biases;

      for(i=0;i<size;i++)
      {
        random[i] = rand();
      }
#endif

      return weights_and_biases;
      // If not, allocate memory (single allocation)
  }
  else if(precision == XA_NNLIB_GRU_flt32xflt32)
  {
      coefff32_t *weights_and_biases, *ptr32;
      coefff32_t *ptr;
      size_t size, size_b;
      char coef_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
      int pad = (XA_PAD_BYTES*pad_flag) >> 2;  //Width of mem bank for HiFi4/5

      size   = 3 * (in_feats + pad) * out_feats;
      size  += 3 * (out_feats + pad) * out_feats;
      
      int num_bias = 3;
      if(split_bias == 1){
        num_bias = 6;
      }
      size_b = num_bias * out_feats ;

      CHECK_PTR_RETURN_NULL(weights, "Allocation for weights");
      CHECK_PTR_RETURN_NULL(biases, "Allocation for biases");
  
      ptr = malloc(size * sizeof(coefff32_t)+ size_b * sizeof(coefff32_t));
      weights_and_biases = (coefff32_t *)ptr;
      CHECK_PTR_RETURN_NULL(ptr, "Allocation for weights_and_biases");
  
      biases->b_z  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_z, out_feats)
  
      biases->b_r  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_r, out_feats)
  
      biases->b_h  = ptr; ptr += out_feats;
      FILL_SHAPE_VECTOR(biases->shape_b_h, out_feats)

      if(split_bias == 1){
        biases->bs_z  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_z, out_feats)
    
        biases->bs_r  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_r, out_feats)
    
        biases->bs_h  = ptr; ptr += out_feats;
        FILL_SHAPE_VECTOR(biases->shape_bs_h, out_feats)
      }
      ptr32 = (coefff32_t*)ptr;

      weights->weightsf32.w_z = ptr32; ptr32 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_w_z, out_feats, in_feats)
  
      weights->weightsf32.u_z = ptr32; ptr32 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_u_z, out_feats, out_feats)
  
      weights->weightsf32.w_r = ptr32; ptr32 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_w_r, out_feats, in_feats)
  
      weights->weightsf32.u_r = ptr32; ptr32 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_u_r, out_feats, out_feats)
  
      weights->weightsf32.w_h = ptr32; ptr32 += (in_feats + pad)  * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_w_h, out_feats, in_feats)
  
      weights->weightsf32.u_h = ptr32; ptr32 += (out_feats + pad) * out_feats;
      FILL_SHAPE_MATRIX(weights->weightsf32.shape_u_h, out_feats, out_feats)

#ifdef FILE_IO
      // Read from file
  
      READ_FILE(coef_file_name, filter_path, coef_files[0], weights->weightsf32.w_z, 4, in_feats,  out_feats, pad, "Allocation for w_z")
      READ_FILE(coef_file_name, filter_path, coef_files[1], weights->weightsf32.u_z, 4, out_feats, out_feats, pad, "Allocation for u_z")
      READ_FILE(coef_file_name, filter_path, coef_files[2], weights->weightsf32.w_r, 4, in_feats,  out_feats, pad, "Allocation for w_r")
      READ_FILE(coef_file_name, filter_path, coef_files[3], weights->weightsf32.u_r, 4, out_feats, out_feats, pad, "Allocation for u_r")
      READ_FILE(coef_file_name, filter_path, coef_files[4], weights->weightsf32.w_h, 4, in_feats,  out_feats, pad, "Allocation for w_h")
      READ_FILE(coef_file_name, filter_path, coef_files[5], weights->weightsf32.u_h, 4, out_feats, out_feats, pad, "Allocation for u_h")
      READ_FILE(coef_file_name, filter_path, coef_files[6], biases->b_z , 4,  out_feats, 1, 0     , "Allocation for b_z")
      READ_FILE(coef_file_name, filter_path, coef_files[7], biases->b_r , 4,  out_feats, 1, 0     , "Allocation for b_r")
      READ_FILE(coef_file_name, filter_path, coef_files[8], biases->b_h , 4,  out_feats, 1, 0     , "Allocation for b_h")
      if(split_bias == 1){
        READ_FILE(coef_file_name, filter_path, coef_files[9], biases->bs_z , 4,  out_feats, 1, 0     , "Allocation for bs_z")
        READ_FILE(coef_file_name, filter_path, coef_files[10], biases->bs_r , 4,  out_feats, 1, 0     , "Allocation for bs_r")
        READ_FILE(coef_file_name, filter_path, coef_files[11], biases->bs_h , 4,  out_feats, 1, 0     , "Allocation for bs_h")    
      }  
#else
      // Generate random data for weights
      coefff32_t *random = weights_and_biases;

      for(i=0;i<size;i++)
      {
        random[i] = rand();
      }
#endif
  return weights_and_biases;
  }
  return NULL;
}

#ifdef VERIFY
#define ABS(A) (((A) < 0) ? -(A):(A))

static float machine_eps(float value, int sum_length)
{
    float epsilon = 1.19e-07;
    float eps, eps_sum;
    int eps_exp;
    frexp(value, &eps_exp);


    if(eps_exp > 0){
        eps = epsilon * eps_exp;
        eps_sum = ((sum_length+1)/2)*eps + eps;
    }
    else if(eps_exp < 0){
        eps = epsilon /(eps_exp * (-1));
        eps_sum = ((sum_length+1)/2)*eps + eps;
    }
    else
    {
        eps = epsilon;
        eps_sum = ((sum_length+1)/2)*eps + eps;
    }
    return eps_sum;
}

static int verify_epsf32(void *p_ref, void *p_out, int len, int sum_length)
{
  int i;
  float *p_in1 = (float *)p_ref;
  float *p_in2 = (float *)p_out;
  float ref_lo, ref_hi;
  float eps;

  for(i = 0; i < len; i++)
  {
    eps = machine_eps(p_in1[i], sum_length);
    ref_lo = p_in1[i] - eps;
    ref_hi = p_in1[i] + eps;
    if(p_in2[i] < ref_lo || p_in2[i] > ref_hi) {return -1;}
  }
  return 0;
}

int comparef32(float *p_dut, float *p_ref, int len, int sum_length)
{
  if(verify_epsf32(p_ref, p_dut, len, sum_length))
  {
      return -1;
  }
  else
  {
      return 0;
  }  
}

int compare(vect_t *p_dut, vect_t *p_ref, int len)
{

  int j;
  vect_t err;
  vect_t max_err;
#ifdef MODEL_FLT64
  err = max_err = 0.0;
#elif MODEL_INT16
  err = max_err = 0;

#endif
  for(j=0;j<len;j++)
  {
    err = ABS(p_ref[j] - p_dut[j]) ;
    if( err > max_err)
    {
      max_err = err; 
    }
  }
#ifdef MODEL_FLT64
  // printf("Max error found wrt the reference = %.20lf\n", max_err);
  if(max_err > 10e-6) return -1;
#elif MODEL_INT16
  printf("Max error found wrt the reference = %d\n", max_err);
  if(max_err > INT16_MAX_ERR) return -1;
  err = max_err = 0;

#endif
  return 0;
}
#endif

int default_config(xa_nnlib_gru_init_config_t *config, 
                    int *verify_flag,
                    char *input_file_name, 
                    char *filter_path, 
                    char *output_file_name, 
                    char *prev_h_file_name,
                    char *ref_file_name) 
{
  if(config)
  {
    config->in_feats = 256;
    config->out_feats = 256;
    config->pad = 1;
    config->split_bias = 0;
    config->mat_prec = 16;
    config->vec_prec = 16;
    config->precision = XA_NNLIB_GRU_16bx16b;
    config->coeff_Qformat = 15;
    config->io_Qformat = 12;
    *verify_flag=1;
    input_file_name[0] = '\0';
    filter_path[0] = '\0';
    output_file_name[0] = '\0';
    prev_h_file_name[0] = '\0';
    ref_file_name[0] = '\0';
    return 0;
  }
  else
  {
    return -1;
  }
}

void parse_arguments(int argc, char** argv, 
                      xa_nnlib_gru_init_config_t *config, 
                      int *show_help,
                      int *verify_flag,
                      char *input_file_name, 
                      char *filter_path, 
                      char *output_file_name, 
                      char *prev_h_file_name,
                      char *ref_file_name)
{
  int argidx;
  for (argidx=1;argidx<argc;argidx++)
  {
    if(strncmp((argv[argidx]), "-", 1) != 0)
    {
      printf("Invalid argument: %s\n",argv[argidx]);
      show_usage();
      exit(1);
    }
    ARGTYPE_INDICATE("-h",*show_help);
    ARGTYPE_INDICATE("-help",*show_help);
    ARGTYPE_INDICATE("--help",*show_help);
    ARGTYPE_ONETIME_CONFIG("--in_feats",config->in_feats);
    ARGTYPE_ONETIME_CONFIG("--out_feats",config->out_feats);
    ARGTYPE_ONETIME_CONFIG("--membank_padding",config->pad);
    ARGTYPE_ONETIME_CONFIG("--split_bias",config->split_bias);
    ARGTYPE_ONETIME_CONFIG("--mat_prec",config->mat_prec);
    ARGTYPE_ONETIME_CONFIG("--vec_prec",config->vec_prec);
    ARGTYPE_ONETIME_CONFIG("--verify",*verify_flag);
    ARGTYPE_STRING("--input_file", input_file_name, XA_MAX_FULL_FILE_NAME_LENGTH);
    ARGTYPE_STRING("--filter_path", filter_path, XA_MAX_FILE_PATH_LENGTH);
    ARGTYPE_STRING("--output_file", output_file_name, XA_MAX_FULL_FILE_NAME_LENGTH);
    ARGTYPE_STRING("--prev_h_file", prev_h_file_name, XA_MAX_FULL_FILE_NAME_LENGTH);
    ARGTYPE_STRING("--ref_file", ref_file_name, XA_MAX_FULL_FILE_NAME_LENGTH);
    
    // If arg doesnt match with any of the above supported options, report option as invalid
    printf("Invalid argument: %s\n",argv[argidx]);
    show_usage();
    exit(1);
  }
}

 /****************************************************************************/
 /*                                   MAIN                                   */
 /****************************************************************************/

#define N_FRAMES 5

int xa_nn_main_process(int argc, char *argv[])
{
  int i;
  int err=0;
  xa_nnlib_gru_init_config_t config;
  char profiler_name[MAX_PROFILER_NAME_LENGTH];
  char profiler_params[MAX_PROFILER_PARAMS_LENGTH]; 
  void *p_weights_biases;
  xa_nnlib_handle_t gru_handle;
  void *p_scratch;
  FILE *input_file;
  FILE *output_file;
  vect_t *p_input;
  vect_t *p_output;
  xa_nnlib_shape_t input_shape;
  xa_nnlib_shape_t output_shape;
  char input_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
  char filter_path[XA_MAX_FILE_PATH_LENGTH];
  char output_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
  char prev_h_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
  int show_help = 0;
  int verify_pass = 1;
#ifdef VERIFY
  FILE *output_ref_file;
  vect_t *output_ref;
  char ref_file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
  int verify_flag;
#endif

  /* Set default configurations */
  if(default_config(&config,
                    &verify_flag,
                    input_file_name, 
                    filter_path, 
                    output_file_name, 
                    prev_h_file_name,
                    ref_file_name))
  {
    return -1;
  }

  /* Library name version etc print */
  fprintf(stderr, "\n--------------------------------------------------------\n");
  fprintf(stderr, "%s library version %s\n",
          xa_nnlib_get_lib_name_string(),
          xa_nnlib_get_lib_version_string());
  fprintf(stderr, "API version: %s\n", xa_nnlib_get_lib_api_version_string());
  fprintf(stderr, "Cadence Design Systems, Inc. http://www.cadence.com\n");
  fprintf(stderr, "\n");

  /* Parse command line options */
  if(argc>1)
  {
    parse_arguments(argc, argv, 
                    &config,
                    &show_help,
                    &verify_flag,
                    input_file_name, 
                    filter_path, 
                    output_file_name, 
                    prev_h_file_name,
                    ref_file_name);
    if(show_help)
    {
      show_usage();
      return 0;
    }
  }

    /* Set precision as per command line */
    if((config.mat_prec == 16)&&(config.vec_prec == 16))
        config.precision = XA_NNLIB_GRU_16bx16b;
    else if((config.mat_prec == 8)&&(config.vec_prec == 16))
        config.precision = XA_NNLIB_GRU_8bx16b;
    else if((config.mat_prec == -1)&&(config.vec_prec == -1))
        config.precision = XA_NNLIB_GRU_flt32xflt32;        
    else
        return err;
        //#error "Unsupported precision\n"

  /* Set coeff_Qformat=7 for mat_prec=8, otherwise coeff_Qformat=15 is default */
  if(config.mat_prec == 8)
    config.coeff_Qformat = 7;

  if(config.precision == XA_NNLIB_GRU_flt32xflt32)
  {
  fprintf(stdout, "Use Case:\nGRU_f32xf32: In Feats: %d, Out Feats: %d",
          config.in_feats, config.out_feats);
  }
  else
  {
  fprintf(stdout, "Use Case:\nGRU_%dx%d: In Feats: %d, Out Feats: %d, Qformats- Weights and Biases: Q%d, Input and Output: Q%d\n",
          config.mat_prec, config.vec_prec, config.in_feats, config.out_feats, config.coeff_Qformat, config.io_Qformat);
  }
  PRINT_STR("Init Loop ");
  {
    int persistent_size;
    int scratch_size;

    /* Get persistent and scratch sizes and allocate them */
    persistent_size = xa_nnlib_gru_get_persistent_fast(&config);  PRINT_VAR(persistent_size)
    if(persistent_size < 0)
    {
      error_code_parse(persistent_size);
      return persistent_size;
    }
    scratch_size = xa_nnlib_gru_get_scratch_fast(&config);   PRINT_VAR(scratch_size)
    if(scratch_size < 0)
    {
      error_code_parse(scratch_size);
      return scratch_size;
    }
   
    gru_handle = (xa_nnlib_handle_t)malloc(persistent_size); PRINT_PTR(gru_handle)
    p_scratch  = malloc(scratch_size);    PRINT_PTR(p_scratch)

    fprintf(stdout, "\nPersistent(fast) size: %8d bytes\n", persistent_size);
    fprintf(stdout, "Scratch(fast) size:    %8d bytes\n", scratch_size);
    /* Initialize GRU Layer with configurations */
    err =xa_nnlib_gru_init(gru_handle, &config);
    
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    }
  }

  /* Set weights and biases for GRU */
  PRINT_STR("Setup Filter and Biases ");
  {
#ifndef CONSTANT_WEIGHTS
    xa_nnlib_gru_weights_t weights;
    xa_nnlib_gru_biases_t biases;

    p_weights_biases = setup_weights_and_biases(
        &weights, 
        &biases, 
        config.in_feats,
        config.out_feats,
        config.pad,
        filter_path,
        config.precision,
        config.split_bias);

    CHECK_PTR(p_weights_biases, "Allocation for p_weights_biases");
#else
#error "Unsupported in this version\n"
#endif

    err=xa_nnlib_gru_set_config(gru_handle, XA_NNLIB_GRU_WEIGHT, &weights);
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    }
    err=xa_nnlib_gru_set_config(gru_handle, XA_NNLIB_GRU_BIAS,   &biases);
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    }
  }


  err=xa_nnlib_gru_get_config(gru_handle, XA_NNLIB_GRU_INPUT_SHAPE, &input_shape);PRINT_VAR(input_shape.dim.vector.length);
  if(XA_NNLIB_NO_ERROR != err)
  {
    error_code_parse(err);
    return err;
  }

  err=xa_nnlib_gru_get_config(gru_handle, XA_NNLIB_GRU_OUTPUT_SHAPE, &output_shape);PRINT_VAR(output_shape.dim.vector.length);
  if(XA_NNLIB_NO_ERROR != err)
  {
    error_code_parse(err);
    return err;
  }
  //Restore context for gru state. This restores the 
  // reference context so that we can match output
  PRINT_STR("GRU restore context")
  {
    char file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
    FILE *prev_h_file;
    void *prev_h;
    strcpy(file_name, pb_prev_h_file_path);
    strcat(file_name, prev_h_file_name);
    prev_h_file=fopen(file_name, "rb");
    CHECK_PTR(prev_h_file, "Opening the context file");

    if(config.precision == XA_NNLIB_GRU_flt32xflt32)
    {
      prev_h = malloc(output_shape.dim.vector.length * sizeof(FLOAT32));
    }
    else
    {
      prev_h = malloc(output_shape.dim.vector.length * sizeof(vect_t));
    }
    CHECK_PTR(prev_h, "temporary Allocate memory for prev context");

    if(config.precision == XA_NNLIB_GRU_flt32xflt32)
    {
      fread((FLOAT32 *)prev_h,sizeof(FLOAT32),output_shape.dim.vector.length,prev_h_file);
    }
    else
    {
      fread((vect_t *)prev_h,sizeof(vect_t),output_shape.dim.vector.length,prev_h_file);
    }
    

    err=xa_nnlib_gru_set_config(gru_handle, XA_NNLIB_GRU_RESTORE_CONTEXT, prev_h);
    if(XA_NNLIB_NO_ERROR != err)
    {
      error_code_parse(err);
      return err;
    }

    fclose(prev_h_file);
    free(prev_h);
  }

  PRINT_STR("GRU Process loop")
  {
    char file_name[XA_MAX_FULL_FILE_NAME_LENGTH];
    strcpy(file_name, pb_input_file_path);
    strcat(file_name, input_file_name);
    Int32 input_buffer_size, output_buffer_size;
    input_file  = fopen(file_name,"rb");
    CHECK_PTR(input_file, "Allocation for input_file");

    strcpy(file_name, pb_output_file_path);
    strcat(file_name, output_file_name);
    output_file = fopen(file_name,"wb");
    CHECK_PTR(output_file, "Allocation for output_file");

    /* Allocate input and output buffer */
    if(config.precision == XA_NNLIB_GRU_flt32xflt32)
    {
      input_buffer_size = input_shape.dim.vector.length * sizeof(FLOAT32);
    }
    else
    {
      input_buffer_size = input_shape.dim.vector.length * sizeof(vect_t);
    }
    p_input   = malloc(input_buffer_size); PRINT_VAR(input_buffer_size);
    CHECK_PTR(p_input, "Allocation for p_input");

    if(config.precision == XA_NNLIB_GRU_flt32xflt32)
    {
      output_buffer_size = output_shape.dim.vector.length * sizeof(FLOAT32);
    }
    else
    {
      output_buffer_size = output_shape.dim.vector.length * sizeof(vect_t);
    }
    p_output = malloc(output_buffer_size); PRINT_VAR(output_buffer_size);
    CHECK_PTR(p_output, "Allocation for p_output");

    fprintf(stdout, "Input size:            %8d bytes\n", input_buffer_size);
    fprintf(stdout, "Output size:           %8d bytes\n\n", output_buffer_size);
#ifdef VERIFY
    if(verify_flag)
    {
      strcpy(file_name, pb_ref_file_path);
      strcat(file_name, ref_file_name);
      output_ref_file = fopen(file_name,"rb");
      CHECK_PTR(output_ref_file, "Allocation for output_ref_file");

      output_ref = malloc(output_shape.dim.vector.length * sizeof(vect_t));
      CHECK_PTR(output_ref, "Allocation for output_ref");
    }
  
#endif // VERIFY  

    // Set profiler name 
    if(config.precision == XA_NNLIB_GRU_flt32xflt32)
    {
      sprintf(profiler_name, "gru_f32xf32");
    }
    else
    {
      sprintf(profiler_name, "gru_%dx%d", 
          config.mat_prec, config.vec_prec);
    }
 
    // Set profiler parameters
    sprintf(profiler_params, "in_feats=%d, out_feats=%d", config.in_feats, config.out_feats);
        
    XTPWR_PROFILER_OPEN(0, profiler_name, profiler_params, config.out_feats, NULL, 0);

    /* Execution Loop */
    PRINT_STR("GRU Process loop starts")
    for(i = 0;i < N_FRAMES; i++)
    {
      xa_nnlib_shape_t output_length;
      xa_nnlib_shape_t input_length;  
      output_length.dim.vector.length = output_shape.dim.vector.length; 
      output_length.shape_type = output_shape.shape_type; 
      // Read input frame
      if(config.precision == XA_NNLIB_GRU_flt32xflt32)
      {
        input_length.dim.vector.length  = fread(p_input, sizeof(FLOAT32), input_shape.dim.vector.length, input_file);
      }
      else
      {
        input_length.dim.vector.length  = fread(p_input, sizeof(vect_t), input_shape.dim.vector.length, input_file);
      }
      
      input_length.shape_type = input_shape.shape_type;
      
      if (input_length.dim.vector.length < input_shape.dim.vector.length) 
      { 
        printf("File end / partial frame \n");
        break;
      }
      
      XTPWR_PROFILER_START(0);
      // Process
      err = xa_nnlib_gru_process(
                gru_handle, 
                p_scratch, 
                p_input, 
                p_output, 
                &input_length, 
                &output_length);
      XTPWR_PROFILER_STOP(0);

      if(XA_NNLIB_NO_ERROR != err)
      {
        error_code_parse(err);
        return err;
      }
     
      PRINT_VAR(input_length.dim.vector.length);
      PRINT_VAR(output_length.dim.vector.length);  
      
      // Write output frame
      if(config.precision == XA_NNLIB_GRU_flt32xflt32)
      {
        fwrite(p_output, sizeof(FLOAT32), output_length.dim.vector.length, output_file);
      }
      else
      {
        fwrite(p_output, sizeof(vect_t), output_length.dim.vector.length, output_file);
      }

#ifdef VERIFY
      {
        if(verify_flag)
        {
          if(config.precision == XA_NNLIB_GRU_flt32xflt32)
          {
            fread(output_ref,sizeof(FLOAT32),output_shape.dim.vector.length,output_ref_file);
          }
          else
          {
            fread(output_ref,sizeof(vect_t),output_shape.dim.vector.length,output_ref_file);
          }
          if(config.precision == XA_NNLIB_GRU_flt32xflt32)
          {
            if(XA_NNLIB_NO_ERROR != comparef32((FLOAT32 *)p_output, (FLOAT32 *)output_ref, output_length.dim.vector.length, 2))
            {
              verify_pass = 0;
            }
          }
          else
          {
            if(XA_NNLIB_NO_ERROR != compare(p_output, output_ref, output_length.dim.vector.length))
            {
              verify_pass = 0;
            }            
          }
        }
      }
      /*---------------------------Verification Part End-----------------------------*/
#endif

    XTPWR_PROFILER_UPDATE(0);
    XTPWR_PROFILER_PRINT(0);
    }

    PRINT_STR("GRU Process loop ended")
#ifdef VERIFY
    XTPWR_PROFILER_CLOSE(0, verify_pass, verify_flag);
#else
    XTPWR_PROFILER_CLOSE(0, verify_pass, 0);
#endif

#ifdef VERIFY
    if(verify_flag)
    {
      fclose(output_ref_file);
      free(output_ref);
    }
#endif
    fclose(output_file);
    fclose(input_file);

    free(p_output);
    free(p_input);
  }
  
  free(p_scratch);
  free(gru_handle);
  
#ifndef CONSTANT_WEIGHTS  
  free(p_weights_biases);
#endif

  return 0;
}

int main (int argc, char *argv[])
{
    FILE *param_file_id;
    int err_code = 0;

    WORD8 curr_cmd[XA_MAX_ARGS * XA_MAX_FULL_FILE_NAME_LENGTH];
    WORD32 fargc, curpos;
    WORD32 processcmd = 0;

    char fargv[XA_MAX_ARGS][XA_MAX_FULL_FILE_NAME_LENGTH];

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
        while(fgets((char *)curr_cmd, XA_MAX_ARGS * XA_MAX_FULL_FILE_NAME_LENGTH, param_file_id))
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
                    fgets((char *)curr_cmd + curpos, XA_MAX_FULL_FILE_NAME_LENGTH, param_file_id);
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
            
            if(strcmp(fargv[0], "@Context_path") == 0)
            {
                if(fargc > 1) strcpy((char *)pb_prev_h_file_path, fargv[1]);
                else strcpy((char *)pb_prev_h_file_path, "");
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
