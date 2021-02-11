/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
#ifndef __STANDARDS_H__
#define __STANDARDS_H__

#include <xtensa/config/core-isa.h>

#if defined(__cplusplus)
extern "C"
{
#endif

#if ( (XCHAL_HAVE_HIFI4_VFPU) )
#define HIFI_VFPU 1
#elif ( (XCHAL_HAVE_HIFI3Z_VFPU) )
#define HIFI_VFPU 1
#elif ( (XCHAL_HAVE_HIFI3_VFPU) )
#define HIFI_VFPU 1
#else
#define HIFI_VFPU 0
#endif

typedef double flt64;
typedef char  Int4;
typedef char  Int8;
typedef short Int16;
typedef int Int32;
typedef int Int24;
typedef long long int Int64;
typedef int Bool;
typedef float Flt32;


#ifdef MODEL_FLT64
typedef double vect_t;
typedef double coeff_t;
typedef double accu_t;

#elif MODEL_INT16
typedef short vect_t;
typedef short coeff_t;
typedef signed char coeff8_t;
typedef long long accu_t;
typedef float coefff32_t;
#endif

typedef struct xa_nnlib_opaque { Int32 _; } *xa_nnlib_handle_t;

typedef enum _xa_nnlib_prec_t
{
  PREC_BOOL   =  1,
  PREC_8      =  8,
  PREC_16     = 16,
  PREC_32     = 32,
  PREC_F32    = -1,
  PREC_F16    = -2,
  PREC_ASYM8U = -3,
  PREC_ASYM8S = -4,
  PREC_SYM8S  = -5,
  PREC_ASYM16U = -6,
  PREC_ASYM16S = -7,
  PREC_SYM16S  = -8,
} xa_nnlib_prec_t;

#define PREC_ASYM8 PREC_ASYM8U

typedef enum _xa_nnlib_shape_type_t
{
  SHAPE_UNKNOWN_T  = 0,
  SHAPE_VECTOR_T   = 1,
  SHAPE_MATRIX_T   = 2,
  SHAPE_CUBE_DWH_T = 3,
  SHAPE_CUBE_WHD_T = 4
} xa_nnlib_shape_type_t;

typedef struct _xa_nnlib_shape_t{
  xa_nnlib_shape_type_t shape_type;
  Int32 n_shapes;
  Int32 shape_offset;   // Offest between current shape and next shape
  union
  {
    struct
    {
      Int32 height;
      Int32 height_offset;
      Int32 width;
      Int32 width_offset;
      Int32 depth;
      Int32 depth_offset;
    } cube;

    struct
    {
      Int32 length;
    } vector;
    struct
    {
      Int32 rows;
      Int32 row_offset;    // Offset between current row and next row
      Int32 cols;
    } matrix;
  } dim;

} xa_nnlib_shape_t;

/*****************************************************************************/
/* Constant hash defines                                                     */
/*****************************************************************************/
#define XA_NNLIB_NO_ERROR    0
/* error handling 'AND' definition */
#define XA_FATAL_ERROR    0x80000000

enum xa_error_severity {
  xa_severity_nonfatal = 0,
  xa_severity_fatal    = (int)0xffffffff
};

enum xa_error_class {
  xa_class_nnlib   = 0,
  xa_class_config  = 1,
  xa_class_execute = 2
};

#define XA_NNLIB_GENERIC    0

#define XA_ERROR_CODE(severity, class, codec, index)    ((severity << 31) | (class << 12) | (codec << 7) | index)
#define XA_ERROR_SEVERITY(code)    (((code) & XA_FATAL_ERROR) != 0)
#define XA_ERROR_CLASS(code)    (((code) >> 12) & 0x0f)
#define XA_ERROR_CODEC(code)    (((code) >>  7) & 0x1f)
#define XA_ERROR_SUBCODE(code)    (((code) >>  0) & 0x3f)

/* Our convention is that only nnlib-class errors can be generic ones. */

/*****************************************************************************/
/* Class 0: NNLib Errors                                                     */
/*****************************************************************************/
/* Non Fatal Errors */
/* (none) */
/* Fatal Errors */
enum xa_error_fatal_nnlib_generic {
  XA_NNLIB_FATAL_MEM_ALLOC        = XA_ERROR_CODE(xa_severity_fatal, xa_class_nnlib, XA_NNLIB_GENERIC, 0),
  XA_NNLIB_FATAL_MEM_ALIGN        = XA_ERROR_CODE(xa_severity_fatal, xa_class_nnlib, XA_NNLIB_GENERIC, 1),
  XA_NNLIB_FATAL_INVALID_SHAPE    = XA_ERROR_CODE(xa_severity_fatal, xa_class_nnlib, XA_NNLIB_GENERIC, 3)
};

/*****************************************************************************/
/* NNLib Startup Functions                                                   */
/*****************************************************************************/
const Int8 * xa_nnlib_get_lib_name_string(void);
const Int8 * xa_nnlib_get_lib_version_string(void);
const Int8 * xa_nnlib_get_lib_api_version_string(void);

#if defined(__cplusplus)
}
#endif

#endif /* __STANDARDS_H__ */
