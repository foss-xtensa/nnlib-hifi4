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
#include <xtensa/config/core-isa.h>
#include "xt_manage_buffers.h"
#include <math.h>
#define COLS_PER_ROW 32  
#define DBG_PRINT printf
//#define DBG_PRINT(...) //printf

int flag = 1;
int  RAND(void)
{
  int signed_random, last_bit;

  if(flag)
  {
    srand(1);
    flag=0;
  } 

  signed_random = 2*(rand() - ((long long)RAND_MAX + 1)/2);
  last_bit = rand() & 1;

  if(signed_random > 0)
  {
     return signed_random - last_bit; 
  }
  else
  {
     return signed_random + last_bit; 
  }
}

buf1D_t *create_buf1D(int len, int precision)
{
  int size_in_bytes;
  buf1D_t *pbuf = malloc(sizeof(buf1D_t));
  if(NULL == pbuf)
  { 
    DBG_PRINT("unable to allocate structure pbuf size(%d )\n", (int) sizeof(buf2D_t));
    return NULL;
  }
  pbuf->length = len;
  pbuf->precision = precision;
  switch(precision)
  {
    case -1: pbuf->bytes_per_element = 4; break;
    case -2: pbuf->bytes_per_element = 2; break;
    case ASYM8_TYPE: pbuf->bytes_per_element = 1; break;
    case ASYM8S_TYPE: pbuf->bytes_per_element = 1; break;
    case SYM8S_TYPE: pbuf->bytes_per_element = 1; break;
    case ASYM16S_TYPE: pbuf->bytes_per_element = 2; break;
    case SYM16S_TYPE: pbuf->bytes_per_element = 2; break;
    case ASYM32S_TYPE: pbuf->bytes_per_element = 4; break;
    case  1: pbuf->bytes_per_element = 1; break;
    case  8: pbuf->bytes_per_element = 1; break;
    case 16: pbuf->bytes_per_element = 2; break;
    case 32: pbuf->bytes_per_element = 4; break;
    case 64: pbuf->bytes_per_element = 8; break;
    default: DBG_PRINT("precision %d is undefined\n", precision);
             return NULL;
  }
  size_in_bytes = (pbuf->bytes_per_element * len);
  if (0 > size_in_bytes)
  {
    DBG_PRINT("[Error] Cannot malloc negative value %d in %s\n", size_in_bytes, __func__);
    return NULL;
  }
  pbuf->p = malloc(size_in_bytes);
  if(NULL == pbuf->p)
  {
    DBG_PRINT("[Error] Unable to allocate buffer for pbuf in %s of size %d\n", __func__, size_in_bytes);
    free_buf1D(pbuf);
    return NULL;
  }
  return pbuf;
}
buf2D_t *create_buf2D(int rows, int cols, int row_offset, int precision, int pad)
{
  int size_in_bytes;

  buf2D_t *pbuf = malloc(sizeof(buf2D_t));
  if(NULL == pbuf) 
  {
    DBG_PRINT("unable to allocate structure pbuf size(%d )\n", (int)sizeof(buf2D_t));
    return NULL;
  }

  pbuf->precision = precision;
  pbuf->cols = cols;
  pbuf->rows = rows;
  pbuf->row_offset = row_offset;


  // Compute bytes per elemnt
  switch(precision)
  {
    case -1:pbuf->bytes_per_element = 4; break;
    case -2:pbuf->bytes_per_element = 2; break;
    case ASYM8_TYPE: pbuf->bytes_per_element = 1;break;
    case ASYM8S_TYPE: pbuf->bytes_per_element = 1; break;
    case SYM8S_TYPE: pbuf->bytes_per_element = 1; break;
    case ASYM16S_TYPE: pbuf->bytes_per_element = 2; break;
    case SYM16S_TYPE: pbuf->bytes_per_element = 2; break;
    case 1: pbuf->bytes_per_element = 1; break;
    case -13:
    case -12:
    case 8: pbuf->bytes_per_element = 1;break;
    case 16:pbuf->bytes_per_element = 2; break;
    case 32:pbuf->bytes_per_element = 4; break;
    case 64:pbuf->bytes_per_element = 8; break;
    default: DBG_PRINT("precision %d is undefined\n", precision);
             return NULL;
  }
#define ADJUST_FOR_MEM_BANK //TBD: move to algo

#ifdef ADJUST_FOR_MEM_BANK
  // We need to ensure the adjacent rows fall in different banks!
  {
    int width_in_bytes = (pbuf->bytes_per_element * cols);
    int bank_offset = width_in_bytes & XCHAL_DATA_WIDTH; 
    if(  (1 == pad ) && 
        (rows > 1)  &&
        (0== bank_offset))
    {
      //rows fall into same bank, need adjustment;
      int pad_elements  = (XCHAL_DATA_WIDTH / pbuf->bytes_per_element);
      pbuf->row_offset = cols + pad_elements;
      DBG_PRINT("Offset adjusted to %d\n", pbuf->row_offset);
    }
  }
#endif

  size_in_bytes = (pbuf->bytes_per_element * pbuf->row_offset * pbuf->rows);
  DBG_PRINT("allocating %d bytes\n", size_in_bytes);
  pbuf->p = malloc(size_in_bytes);
  if(NULL == pbuf->p)
  {
    DBG_PRINT("[Error] Unable to allocate buffer for pbuf in %s of size %d\n", __func__, size_in_bytes);
    free_buf2D(pbuf);
    return NULL;
  }
  return pbuf;
}

void free_buf1D(buf1D_t *ptr)
{
  free(ptr->p);
  free(ptr);
}
void free_buf2D(buf2D_t *ptr)
{
  free(ptr->p);
  free(ptr);
}
int interleave_buf1D(buf1D_t *pbuf, buf1D_t *pbuf_interleaved, int length)
{
    int numbytes=0;
    //Implemented only for numbytes=sizeof(int), rest of the cases are TBD
    int * ptr2 = NULL;
    int * ptr1 = pbuf->p;
    int * ptr3 = pbuf_interleaved->p;
    int j=0;
    int i=0;

    switch(pbuf->precision)
    {
      case -1: numbytes = sizeof(float);       break;
      case ASYM8_TYPE: numbytes = sizeof(char);        break;
      case ASYM8S_TYPE: numbytes = sizeof(char); break;
      case SYM8S_TYPE: numbytes = sizeof(char); break;
      case 8: numbytes = sizeof(char);        break;
      case 16: numbytes = sizeof(short int);   break;
      case 32: numbytes = sizeof(int);         break;
      case 64: numbytes = sizeof(double);     break;    
    }

    ptr2 = pbuf->p+numbytes * length * 2;

    for ( i=0;i< 2*length;i+=2)
    {
      ptr3[j] = ptr1[i]; j++;
      ptr3[j] = ptr1[i+1]; j++;
      
      ptr3[j] = ptr2[i]; j++;
      ptr3[j] = ptr2[i+1]; j++;

    }
  return 0;
}
int deinterleave_buf1D(buf1D_t *pbuf, buf1D_t *pbuf_deinterleaved, int length)
{
    int numbytes = 0;
    //Implemented only for numbytes=sizeof(int), rest of the cases are TBD
    int * ptr1 = pbuf_deinterleaved->p;
    int * ptr2 = NULL;
    int * ptr3 = pbuf->p;
    int j=0;
    int i=0;
    
    switch(pbuf->precision)
    {
      case -1: numbytes = sizeof(float);       break;
      case ASYM8_TYPE: numbytes = sizeof(char);        break;
      case ASYM8S_TYPE: numbytes = sizeof(char); break;
      case SYM8S_TYPE: numbytes = sizeof(char); break;
      case 8: numbytes = sizeof(char);        break;
      case 16: numbytes = sizeof(short int);   break;
      case 32: numbytes = sizeof(int);         break;
      case 64: numbytes = sizeof(double);     break;    
    }
    
    ptr2 = pbuf_deinterleaved->p + numbytes*length*2;

    for (i=0;i< 2*length ;i+=2)
    {
        ptr1[i]= ptr3[j];j++;
        ptr1[i+1]= ptr3[j];j++;

        ptr2[i]= ptr3[j];j++;
        ptr2[i+1]= ptr3[j];j++;
    }
  return 0;
}
int interleave_buf1D_real(buf1D_t *pbuf, buf1D_t *pbuf_interleaved, int length)
{
    int numbytes=0;
    //Implemented only for numbytes=sizeof(int), rest of the cases are TBD
    int * ptr2 = NULL;
    int * ptr1 = pbuf->p;
    int * ptr3 = pbuf_interleaved->p;
    int j=0;
    int i=0;

    switch(pbuf->precision)
    {
      case -1: numbytes = sizeof(float);       break;
      case ASYM8_TYPE: numbytes = sizeof(char);        break;
      case ASYM8S_TYPE: numbytes = sizeof(char); break;
      case SYM8S_TYPE: numbytes = sizeof(char); break;
      case 8: numbytes = sizeof(char);        break;
      case 16: numbytes = sizeof(short int);   break;
      case 32: numbytes = sizeof(int);         break;
      case 64: numbytes = sizeof(double);     break;    
    }

    ptr2 = pbuf->p+numbytes * length;

    for ( i=0;i< length;i++)
    {
      ptr3[j] = ptr1[i]; j++;
      ptr3[j] = ptr2[i]; j++;
    }
  return 0;
}
int deinterleave_buf1D_real(buf1D_t *pbuf, buf1D_t *pbuf_deinterleaved, int length)
{
    int numbytes = 0;
    //Implemented only for numbytes=sizeof(int), rest of the cases are TBD
    int * ptr1 = pbuf_deinterleaved->p;
    int * ptr2 = NULL;
    int * ptr3 = pbuf->p;
    int j=0;
    int i=0;
    
    switch(pbuf->precision)
    {
      case -1: numbytes = sizeof(float);       break;
      case ASYM8_TYPE: numbytes = sizeof(char);        break;
      case ASYM8S_TYPE: numbytes = sizeof(char); break;
      case SYM8S_TYPE: numbytes = sizeof(char); break;
      case  8: numbytes = sizeof(char);        break;
      case 16: numbytes = sizeof(short int);   break;
      case 32: numbytes = sizeof(int);         break;
      case 64: numbytes = sizeof(double);      break;    
    }
    
    ptr2 = pbuf_deinterleaved->p + numbytes*length;

    for (i=0;i< length ;i++)
    {
        ptr1[i]= ptr3[j];j++;
        ptr2[i]= ptr3[j];j++;
    }

  return 0;
}

int set_rand_inp_buf1D(buf1D_t *ptr_buf1D)
{
  int i;
  // Generate random data
  switch(ptr_buf1D->precision)
  {
    case -1:
      {
        float *p = (float *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = ((float)RAND())/((float)((long long)RAND_MAX+1));
        }
      }
      break;
    case -2:
      {
        short *p = (short *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;      
    case ASYM8_TYPE: 
    case ASYM8S_TYPE: 
    case SYM8S_TYPE: 
      {
        char *p = (char *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 8: 
      {
        char *p = (char *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 1: 
      {
        char *p = (char *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = (RAND() & 1);
        }
      }
      break;
    case ASYM16S_TYPE: 
    case SYM16S_TYPE: 
    case 16:
      {
        short *p = (short *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 32:
      {
        int *p = (int *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 64: 
      {
        long long *p = (long long *) ptr_buf1D->p;
        for (i = 0; i < ptr_buf1D->length; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    default: 
      printf("Error in setting random input for vector - Unknown precision %d\n",ptr_buf1D->precision);
      return -1;
  }
  return 0;
}

int set_rand_inp_buf2D(buf2D_t *ptr_buf2D)
{
  // Generate random data
  int i;
  switch(ptr_buf2D->precision)
  {
    case -1:
      {
        float *p = (float *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = ((float)RAND())/((float)RAND_MAX);
          if(p[i]>1 || p[i]<-1)
            printf("rand() not in [-1, 1) %f\n",p[i]);
        }
      }
      break;
    case -2:
      {
        short *p = (short *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }      
    case ASYM8_TYPE:
    case ASYM8S_TYPE: 
    case SYM8S_TYPE: 
      {
        char *p = (char *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case -12:  
    case -13:
      {
        char *p = (char *) ptr_buf2D->p;
        for (i = 0; i < ((ptr_buf2D->rows * ptr_buf2D->row_offset) / 2); i++)
        {
          p[i] = RAND();
        }
      }
      break;      
    case 8: 
      {
        char *p = (char *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case ASYM16S_TYPE: 
    case SYM16S_TYPE: 
    case 16:
      {
        short *p = (short *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 32:
      {
        int *p = (int *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    case 64: 
      {
        long long *p = (long long *) ptr_buf2D->p;
        for (i = 0; i < ptr_buf2D->rows * ptr_buf2D->row_offset; i++)
        {
          p[i] = RAND();
        }
      }
      break;
    default: 
      printf("Error in setting random input for matrix - Unknown precision %d\n",ptr_buf2D->precision);
      return -1;
  }
  return 0;
}


#define WRITE_ROW(n)\
  fprintf(f,"%s","\t"); \
  for(cols = 0; cols < n;cols++){fprintf(f,"0x%02x,",(unsigned char )(*p++));} \
  fprintf(f,"%s","\n");

#define END_STRING \
  fprintf(f,"%s\n"," }; ");

static void write_array(FILE * f, unsigned char *p, int l, BOOL e, char *str)
{
  if(e)
  {

    int i,cols;
    fprintf(f,"unsigned char %s[] = { \n", (NULL == str) ? "bin" : str);

    if(l>=COLS_PER_ROW)
    {
      for (i = 0; i < l/COLS_PER_ROW; i++)
      {
        WRITE_ROW(COLS_PER_ROW)
      }
    }
    WRITE_ROW(l%COLS_PER_ROW)
      END_STRING
  }
  else
  {
    fwrite (p , sizeof(char), l, f);
  }
}

void write_buf1D(buf1D_t *pbuf, FILE *file,int extensionIndicator, char * var_name)
{
  if( NULL != file)
  {
    int length=0;
    switch(pbuf->precision)
    {
      case -1: length = sizeof(float) * pbuf->length;   break;
      case ASYM8_TYPE:  length = sizeof(char)  * pbuf->length;   break;
      case ASYM8S_TYPE:  length = sizeof(char)  * pbuf->length;   break;
      case SYM8S_TYPE:  length = sizeof(char)  * pbuf->length;   break;
      case ASYM16S_TYPE:  length = sizeof(short int)  * pbuf->length;   break;
      case SYM16S_TYPE:  length = sizeof(short int)  * pbuf->length;   break;
	    case 1:  length = sizeof(char)  * pbuf->length; break;
      case 8:  length = sizeof(char)  * pbuf->length;   break;
      case 16: length = sizeof(short) * pbuf->length;   break;
      case 32: length = sizeof(int) * pbuf->length;   break;
      case 64: length = sizeof(long long) * pbuf->length;   break;
      default: DBG_PRINT("precision %d is undefined\n", pbuf->precision);

    }
    write_array(file, (unsigned char *) pbuf->p, length, extensionIndicator, var_name );
  }
}


void write_buf2D(buf2D_t *pbuf, FILE *file,int extensionIndicator, char * var_name)
{
  if( NULL != file)
  {
    int length=0;
    switch(pbuf->precision)
    {
      case -1: length = sizeof(float) * pbuf->rows * pbuf->row_offset;   break;
      case ASYM8_TYPE: length = sizeof(char)  * pbuf->rows * pbuf->row_offset;   break;
      case ASYM8S_TYPE:  length = sizeof(char)  * pbuf->rows * pbuf->row_offset;   break;
      case SYM8S_TYPE:  length = sizeof(char)  * pbuf->rows * pbuf->row_offset;   break;
      case ASYM16S_TYPE:  length = sizeof(short int)  * pbuf->row_offset;   break;
      case SYM16S_TYPE:  length = sizeof(short int)  * pbuf->row_offset;   break;
	    case 1:  length = sizeof(char)  * pbuf->rows * pbuf->row_offset;   break;
      case 8:  length = sizeof(char)  * pbuf->rows * pbuf->row_offset;   break;
      case 16: length = sizeof(short) * pbuf->rows * pbuf->row_offset;   break;
      case 32: length = sizeof(int) * pbuf->rows * pbuf->row_offset;   break;
      case 64: length = sizeof(long long) * pbuf->rows * pbuf->row_offset;   break;
      default: DBG_PRINT("precision %d is undefined\n", pbuf->precision);

    }
    write_array(file, (unsigned char *) pbuf->p, length, extensionIndicator, var_name );
  }
}

static int verify_bitexact(void *p_ref, void *p_out, int len)
{
  int i;
  char *p_in1 = (char *)p_ref;
  char *p_in2 = (char *)p_out;

  for(i = 0; i < len; i++)
  {
    if(p_in1[i] != p_in2[i]) {
      return -1;
    }
  }
  return 0;
}

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
/*
 * Compare 1D buffers.
 * Return 1 is match else 0
 */
int compare_buf1D(buf1D_t *pbuf_ref, buf1D_t *pbuf_out, int method, int precision, int sum_length)
{
  if(method == 1 && (precision == -1))/*For f32 cases only*/
   {
       int length = pbuf_ref->length;
       if(verify_epsf32(pbuf_ref->p, pbuf_out->p, length, sum_length))
       {
           return 0;
       }
       else
       {
           return 1;
       }

   } 
  if(method == 1 && (precision != -1)) /* Bitexact match */
   {
       int size_in_bytes = (pbuf_ref->bytes_per_element * pbuf_ref->length);
       if(verify_bitexact(pbuf_ref->p, pbuf_out->p, size_in_bytes))
       {
           return 0;
       }
       else
       {
           return 1;
       }
   }
      
   return 1;
}


int compare_buf2D(buf2D_t *pbuf_ref, buf2D_t *pbuf_out, int method)
{
   if((pbuf_ref->rows != pbuf_out->rows) || (pbuf_ref->cols != pbuf_out->cols))
   {
      DBG_PRINT("compare failed: reference and output dimensions doesnt match\n");
      return 0;
   }

   if(method == 1) /* Bitexact match */
   {
       int size_in_bytes = (pbuf_ref->bytes_per_element * pbuf_ref->rows * pbuf_ref->cols);
       if(verify_bitexact(pbuf_ref->p, pbuf_out->p, size_in_bytes))
       {
           return 0;
       }
       else
       {
           return 1;
       }
   }
      
   return 0;
}

