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
#ifndef __XA_NNLIB_LEGACY_COMPAT_H__
#define __XA_NNLIB_LEGACY_COMPAT_H__
#ifndef ENABLE_SCRATCH_SIZE_API_ONLY
#include <xtensa/config/core-isa.h>
#include "xtensa/tie/xt_hifi2.h"
#endif /* #ifndef ENABLE_SCRATCH_SIZE_API_ONLY */
#include "xa_nnlib_api.h"
#include "xa_nnlib_standards.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_hifi_isa_compat.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_internal.h"
#endif /* __XA_NNLIB_LEGACY_COMPAT_H__ */

#define RI9_HWVERSION 281090
#if (XCHAL_HAVE_HIFI3 && !(XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5))
#define XA_HAVE_HIFI3_CORE 1
#else
#define XA_HAVE_HIFI3_CORE 0
#endif
