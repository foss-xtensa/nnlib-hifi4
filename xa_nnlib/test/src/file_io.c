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
#include "xa_type_def.h"
#include "xt_manage_buffers.h"

#define GET_SIZE_FROM_PRECISION(precision, size) {        \
  switch(precision)                                       \
  {                                                       \
    case -1: size = sizeof(float);                break;  \
    case -2: size = sizeof(short);                break;  \
    case ASYM8_TYPE: size = sizeof(char);         break;  \
    case ASYM8S_TYPE: size = sizeof(char);        break;  \
    case SYM8S_TYPE: size = sizeof(char);         break;  \
    case SYM16S_TYPE: size = sizeof(short int);   break;  \
    case ASYM16S_TYPE: size = sizeof(short int);  break;  \
    case ASYM32S_TYPE: size = sizeof(int);        break;  \
    case 1: size = sizeof(char);                  break;  \
    case -12:                                             \
    case -13:                                             \
    case 8: size = sizeof(char);                  break;  \
    case 16: size = sizeof(short int);            break;  \
    case 32: size = sizeof(int);                  break;  \
    case 64: size = sizeof(double);               break;  \
  }                                                       \
}

int read_buf1D_from_file(FILE *fptr_read_data, buf1D_t *ptr_buf1D) 
{                                  
  int size;                                                                            
  GET_SIZE_FROM_PRECISION(ptr_buf1D->precision, size)                                  
  if(ptr_buf1D->length != fread(ptr_buf1D->p, size, ptr_buf1D->length, fptr_read_data))
  {                                                                                    
    printf("Error reading input/reference vector from file\n");                   
    return -1;                                                                         
  }                                                                                    
  return 0;
}

int read_buf2D_from_file(FILE *fptr_read_data, buf2D_t *ptr_buf2D, int pad_val) 
{    
  int size, row;                                         
  char *ptr_mat = (char *)ptr_buf2D->p;                  
  GET_SIZE_FROM_PRECISION(ptr_buf2D->precision, size)    
  for(row=0; row<ptr_buf2D->rows; row++)                 
  {                                 
    if(ptr_buf2D->precision == -13)
    {
      if(ptr_buf2D->cols != fread((ptr_mat + ((row * ptr_buf2D->row_offset * size) / 2)), size, ptr_buf2D->cols, fptr_read_data))
      {                                                                                        
        printf("Error reading input/reference matrix from file\n");                       
        return -1;                                                                             
      }
    } 
    else
    {                    
      if(ptr_buf2D->cols != fread((ptr_mat + (row * ptr_buf2D->row_offset * size)), size, ptr_buf2D->cols, fptr_read_data))
      {                                                                                        
        printf("Error reading input/reference matrix from file\n");                       
        return -1;                                                                             
      }
    }
    if(ptr_buf2D->precision == ASYM8_TYPE)
    {
      int pad_size = ptr_buf2D->row_offset - ptr_buf2D->cols;
      pad_size = (pad_size < 0) ? 0: pad_size;
      memset((ptr_mat + (row * ptr_buf2D->row_offset * size) + ptr_buf2D->cols * size), (UWORD8)pad_val, pad_size);
    }
    else if(ptr_buf2D->precision == -13)
    {
      int pad_size = ptr_buf2D->row_offset - ptr_buf2D->cols;
      pad_size = (pad_size < 0) ? 0: pad_size;
      memset((ptr_mat + ((row * ptr_buf2D->row_offset * size) / 2) + ptr_buf2D->cols * size), (UWORD8)pad_val, pad_size);
    }    
    else
    {
      int pad_size = ptr_buf2D->row_offset - ptr_buf2D->cols;
      pad_size = (pad_size < 0) ? 0: pad_size;
      memset((ptr_mat + (row * ptr_buf2D->row_offset * size) + ptr_buf2D->cols * size), 0, size * pad_size);
    }
  }
  return 0;  
}

int write_buf1D_to_file(FILE *fptr_write_data, buf1D_t *ptr_buf1D) 
{                                    
  int size;                                                                              
  GET_SIZE_FROM_PRECISION(ptr_buf1D->precision, size)                                    
  if(ptr_buf1D->length != fwrite(ptr_buf1D->p, size, ptr_buf1D->length, fptr_write_data))
  {                                                                                      
    printf("Error writing vector to file\n");                       
    return -1;                                                                            
  }                                                                                      
  return 0;
}

int write_buf2D_to_file(FILE *fptr_write_data, buf2D_t *ptr_buf2D) 
{                                                    
  int size, row;                                                                                             
  char *ptr_mat = (char *)ptr_buf2D->p;                                                                      
  GET_SIZE_FROM_PRECISION(ptr_buf2D->precision, size)                                                        
  for(row=0; row<ptr_buf2D->rows; row++)                                                                     
  {                                                                                                          
    if(ptr_buf2D->cols != fwrite((ptr_mat + (row * ptr_buf2D->row_offset * size)), size, ptr_buf2D->cols, fptr_write_data))
    {                                                                                      
      printf("Error writing matrix to file\n");                       
      return -1;                                                                           
    }                                                                                      
  }
  return 0;  
}

int load_matXvec_input_data(int write_file, FILE *fptr_inp, buf2D_t *p_mat1, buf1D_t *p_vec1, 
    buf2D_t *p_mat2, buf1D_t *p_vec2, buf1D_t *p_bias) 
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf2D(p_mat1);                                                      
    set_rand_inp_buf1D(p_vec1);                                                      
    set_rand_inp_buf2D(p_mat2);                                                      
    set_rand_inp_buf1D(p_vec2);                                                      
    set_rand_inp_buf1D(p_bias);                                                      
                                                                                     
    /* Write input data into file */                                                 
    write_buf2D_to_file(fptr_inp, p_mat1);                  
    write_buf1D_to_file(fptr_inp, p_vec1);                  
    write_buf2D_to_file(fptr_inp, p_mat2);                  
    write_buf1D_to_file(fptr_inp, p_vec2);                  
    write_buf1D_to_file(fptr_inp, p_bias);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf2D_from_file(fptr_inp, p_mat1, 0);                  
    read_buf1D_from_file(fptr_inp, p_vec1);                  
    read_buf2D_from_file(fptr_inp, p_mat2, 0);                  
    read_buf1D_from_file(fptr_inp, p_vec2);                  
    read_buf1D_from_file(fptr_inp, p_bias);                  
  }                                                                                  
  return 0;
}

int load_activation_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf1D_t *p_inp_alpha, char *kernel_name) 
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
                                                                                     
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  

    if(!strcmp(kernel_name,"prelu"))
    {
      /* Set random input_alpha data */                                                      
      set_rand_inp_buf1D(p_inp_alpha);                                                      
                                                                                     
      /* Write input_alpha data into file */                                                 
      write_buf1D_to_file(fptr_inp, p_inp_alpha);
    }
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  

    if(!strcmp(kernel_name,"prelu"))
    {
      /* Read input_alpha data from file */                           
      read_buf1D_from_file(fptr_inp, p_inp_alpha);
    }
  }                                                                                  
  return 0;
}

int load_conv2d_std_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel, 
    buf1D_t *p_bias, int input_channels, int input_channels_pad, int kernel_pad_val) 
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    set_rand_inp_buf2D(p_kernel);                                                      
    set_rand_inp_buf1D(p_bias);                                                      
    
    /* If input_channels has been padded, zero-out the padded channels in kernel */     
    if(input_channels != input_channels_pad)
    {
      int i;
      char *p = (char *) p_kernel->p + input_channels*p_kernel->bytes_per_element;
      int padded_bytes = (input_channels_pad-input_channels)*p_kernel->bytes_per_element;
      for (i = 0; i < p_kernel->rows * p_kernel->row_offset; i+=input_channels_pad)
      {
        memset((p + i*p_kernel->bytes_per_element), 0, padded_bytes);
      }
    }

    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
    write_buf2D_to_file(fptr_inp, p_kernel);                  
    write_buf1D_to_file(fptr_inp, p_bias);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
    read_buf2D_from_file(fptr_inp, p_kernel, kernel_pad_val);                  
    read_buf1D_from_file(fptr_inp, p_bias);                  
  }                                                                                  
  return 0;
}

int load_conv1d_std_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel, 
    buf1D_t *p_bias, int input_channels, int input_width, int input_channelsXwidth_pad,
    int kernel_pad_val) 
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    set_rand_inp_buf2D(p_kernel);                                                      
    set_rand_inp_buf1D(p_bias);                                                      

    /* If input channels X width plane has been padded, zero-out the padded planes in kernel */
    if(input_channels * input_width != input_channelsXwidth_pad)
    {
      int i;
      char *p = (char *) p_kernel->p + input_channels*input_width*p_kernel->bytes_per_element;
      int padded_bytes = (input_channelsXwidth_pad-input_channels*input_width)*p_kernel->bytes_per_element;
      for (i = 0; i < p_kernel->rows * p_kernel->row_offset; i+=input_channelsXwidth_pad)
      {
        memset((p + i*p_kernel->bytes_per_element), 0, padded_bytes);
      }
    }

    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
    write_buf2D_to_file(fptr_inp, p_kernel);                  
    write_buf1D_to_file(fptr_inp, p_bias);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
    read_buf2D_from_file(fptr_inp, p_kernel, kernel_pad_val);                  
    read_buf1D_from_file(fptr_inp, p_bias);                  
  }                                                                                  
  return 0;
}
int load_conv2d_ds_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf2D_t *p_kernel, 
    buf1D_t *p_bias, buf1D_t *p_kernel_point, buf1D_t *p_bias_point, int kernel_pad_val)
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    set_rand_inp_buf2D(p_kernel);                                                      
    set_rand_inp_buf1D(p_bias);                                                      
    set_rand_inp_buf1D(p_kernel_point);                                                      
    set_rand_inp_buf1D(p_bias_point);                                                      
                                                                                     
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
    write_buf2D_to_file(fptr_inp, p_kernel);                  
    write_buf1D_to_file(fptr_inp, p_bias);                  
    write_buf1D_to_file(fptr_inp, p_kernel_point);                  
    write_buf1D_to_file(fptr_inp, p_bias_point);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
    read_buf2D_from_file(fptr_inp, p_kernel, kernel_pad_val);                  
    read_buf1D_from_file(fptr_inp, p_bias);                  
    read_buf1D_from_file(fptr_inp, p_kernel_point);                  
    read_buf1D_from_file(fptr_inp, p_bias_point);                  
  }                                                                                  
  return 0;
}

int load_dilated_conv2d_depth_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp,
    buf2D_t *p_kernel, buf1D_t *p_bias, int kernel_pad_val)
{
  if(write_file)
  {
    /* Set random input data */
    set_rand_inp_buf1D(p_inp);
    set_rand_inp_buf2D(p_kernel);
    set_rand_inp_buf1D(p_bias);

    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);
    write_buf2D_to_file(fptr_inp, p_kernel);
    write_buf1D_to_file(fptr_inp, p_bias);
  }
  else
  {
    /* Read input data from file */
    read_buf1D_from_file(fptr_inp, p_inp);
    read_buf2D_from_file(fptr_inp, p_kernel, kernel_pad_val);
    read_buf1D_from_file(fptr_inp, p_bias);
  }
  return 0;
}

int load_conv2d_pt_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp,
    buf1D_t *p_kernel_point, buf1D_t *p_bias_point)
{  
  if(write_file)                                                                     
  {
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    set_rand_inp_buf1D(p_kernel_point);                                                      
    set_rand_inp_buf1D(p_bias_point);                                                      
                                                                                     
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
    write_buf1D_to_file(fptr_inp, p_kernel_point);                  
    write_buf1D_to_file(fptr_inp, p_bias_point);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
    read_buf1D_from_file(fptr_inp, p_kernel_point);                  
    read_buf1D_from_file(fptr_inp, p_bias_point);                  
  }                                                                                  
  return 0;
}

int load_pool_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp) 
{  
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
  }                                                                                  
  return 0;
}

int load_norm_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp) 
{
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
  }                                                                                  
  return 0;
}

int load_batch_norm_3D_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp, buf1D_t *p_alpha, buf1D_t *p_beta) 
{
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
  }                                                                                  
  return 0;
}

int load_reorg_input_data(int write_file, FILE *fptr_inp, buf1D_t *p_inp) 
{
  if(write_file)                                                                     
  {                                                                                  
    /* Set random input data */                                                      
    set_rand_inp_buf1D(p_inp);                                                      
    
    /* Write input data into file */                                                 
    write_buf1D_to_file(fptr_inp, p_inp);                  
  }                                                           
  else                                                        
  {                                                           
    /* Read input data from file */                           
    read_buf1D_from_file(fptr_inp, p_inp);                  
  }                                                                                  
  return 0;
}

int write_output_data(FILE *fptr_out, buf1D_t *p_out) 
{  
  write_buf1D_to_file(fptr_out, p_out);                  
  return 0;
}

FILE* file_open(char *file_path, char *file_name, char *mode, int max_file_name_length)
{
  FILE *fptr;
  char *ptr_file = malloc(max_file_name_length);

  if (max_file_name_length >= (strlen(file_path) + strlen(file_name)))
  {
    strcpy(ptr_file, file_path);
    strcat(ptr_file, file_name);
  }
  else
  {
    printf("parse_cmdline: Filename too big %s%s, should be <%d chars\n", file_path, file_name, max_file_name_length);
    exit(1);
  }

  if(NULL == (fptr = fopen(ptr_file, mode)))
  {
    printf("Error opening file %s with mode %s\n", ptr_file, mode);
    exit(1);
  }
  return fptr;
}


int load_basic_func_data(int write_file, FILE *fptr_inp1, FILE *fptr_inp2, buf1D_t *p_inp1, buf1D_t *p_inp2) 
{
  if(write_file)
  {
    /* Set random input data */
	if(p_inp1)
		set_rand_inp_buf1D(p_inp1);
	if(p_inp2)
		set_rand_inp_buf1D(p_inp2);

    /* Write input data into file */
	if(p_inp1 && fptr_inp1)
		write_buf1D_to_file(fptr_inp1, p_inp1);
	if(p_inp2 && fptr_inp2)
      		write_buf1D_to_file(fptr_inp2, p_inp2);
  }
  else
  {
    /* Read input data from file */
	if(p_inp1 && fptr_inp1)
		 read_buf1D_from_file(fptr_inp1, p_inp1);
	if(p_inp2 && fptr_inp2)
  		 read_buf1D_from_file(fptr_inp2, p_inp2);
  }
  return 0;
}

