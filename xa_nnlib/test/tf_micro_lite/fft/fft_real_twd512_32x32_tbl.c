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
/* ------------------------------------------------------------------------ */
/* Copyright (c) 2017 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ("Cadence    */
/* Libraries") are the copyrighted works of Cadence Design Systems Inc.	    */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* DSP Library                                                              */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2015-2017 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
#include "NatureDSP_Signal_fft.h"
#include "fft_twiddles32x32.h"
#include "hifi_common.h"

ALIGN(8) static const int32_t __fft_real_tw[] =
{
    (int32_t)0x7FFFFFFF, (int32_t)0x00000000, (int32_t)0x7FFD885A, (int32_t)0xFE6DE2E0, (int32_t)0x7FF62182, (int32_t)0xFCDBD541, (int32_t)0x7FE9CBC0, (int32_t)0xFB49E6A3,
    (int32_t)0x7FD8878E, (int32_t)0xF9B82684, (int32_t)0x7FC25596, (int32_t)0xF826A462, (int32_t)0x7FA736B4, (int32_t)0xF6956FB7, (int32_t)0x7F872BF3, (int32_t)0xF50497FB,
    (int32_t)0x7F62368F, (int32_t)0xF3742CA2, (int32_t)0x7F3857F6, (int32_t)0xF1E43D1C, (int32_t)0x7F0991C4, (int32_t)0xF054D8D5, (int32_t)0x7ED5E5C6, (int32_t)0xEEC60F31,
    (int32_t)0x7E9D55FC, (int32_t)0xED37EF91, (int32_t)0x7E5FE493, (int32_t)0xEBAA894F, (int32_t)0x7E1D93EA, (int32_t)0xEA1DEBBB, (int32_t)0x7DD6668F, (int32_t)0xE8922622,
    (int32_t)0x7D8A5F40, (int32_t)0xE70747C4, (int32_t)0x7D3980EC, (int32_t)0xE57D5FDA, (int32_t)0x7CE3CEB2, (int32_t)0xE3F47D96, (int32_t)0x7C894BDE, (int32_t)0xE26CB01B,
    (int32_t)0x7C29FBEE, (int32_t)0xE0E60685, (int32_t)0x7BC5E290, (int32_t)0xDF608FE4, (int32_t)0x7B5D039E, (int32_t)0xDDDC5B3B, (int32_t)0x7AEF6323, (int32_t)0xDC597781,
    (int32_t)0x7A7D055B, (int32_t)0xDAD7F3A2, (int32_t)0x7A05EEAD, (int32_t)0xD957DE7A, (int32_t)0x798A23B1, (int32_t)0xD7D946D8, (int32_t)0x7909A92D, (int32_t)0xD65C3B7B,
    (int32_t)0x78848414, (int32_t)0xD4E0CB15, (int32_t)0x77FAB989, (int32_t)0xD3670446, (int32_t)0x776C4EDB, (int32_t)0xD1EEF59E, (int32_t)0x76D94989, (int32_t)0xD078AD9E,
    (int32_t)0x7641AF3D, (int32_t)0xCF043AB3, (int32_t)0x75A585CF, (int32_t)0xCD91AB39, (int32_t)0x7504D345, (int32_t)0xCC210D79, (int32_t)0x745F9DD1, (int32_t)0xCAB26FA9,
    (int32_t)0x73B5EBD1, (int32_t)0xC945DFEC, (int32_t)0x7307C3D0, (int32_t)0xC7DB6C50, (int32_t)0x72552C85, (int32_t)0xC67322CE, (int32_t)0x719E2CD2, (int32_t)0xC50D1149,
    (int32_t)0x70E2CBC6, (int32_t)0xC3A94590, (int32_t)0x7023109A, (int32_t)0xC247CD5A, (int32_t)0x6F5F02B2, (int32_t)0xC0E8B648, (int32_t)0x6E96A99D, (int32_t)0xBF8C0DE3,
    (int32_t)0x6DCA0D14, (int32_t)0xBE31E19B, (int32_t)0x6CF934FC, (int32_t)0xBCDA3ECB, (int32_t)0x6C242960, (int32_t)0xBB8532B0, (int32_t)0x6B4AF279, (int32_t)0xBA32CA71,
    (int32_t)0x6A6D98A4, (int32_t)0xB8E31319, (int32_t)0x698C246C, (int32_t)0xB796199B, (int32_t)0x68A69E81, (int32_t)0xB64BEACD, (int32_t)0x67BD0FBD, (int32_t)0xB5049368,
    (int32_t)0x66CF8120, (int32_t)0xB3C0200C, (int32_t)0x65DDFBD3, (int32_t)0xB27E9D3C, (int32_t)0x64E88926, (int32_t)0xB140175B, (int32_t)0x63EF3290, (int32_t)0xB0049AB3,
    (int32_t)0x62F201AC, (int32_t)0xAECC336C, (int32_t)0x61F1003F, (int32_t)0xAD96ED92, (int32_t)0x60EC3830, (int32_t)0xAC64D510, (int32_t)0x5FE3B38D, (int32_t)0xAB35F5B5,
    (int32_t)0x5ED77C8A, (int32_t)0xAA0A5B2E, (int32_t)0x5DC79D7C, (int32_t)0xA8E21106, (int32_t)0x5CB420E0, (int32_t)0xA7BD22AC, (int32_t)0x5B9D1154, (int32_t)0xA69B9B68,
    (int32_t)0x5A82799A, (int32_t)0xA57D8666, (int32_t)0x59646498, (int32_t)0xA462EEAC, (int32_t)0x5842DD54, (int32_t)0xA34BDF20, (int32_t)0x571DEEFA, (int32_t)0xA2386284,
    (int32_t)0x55F5A4D2, (int32_t)0xA1288376, (int32_t)0x54CA0A4B, (int32_t)0xA01C4C73, (int32_t)0x539B2AF0, (int32_t)0x9F13C7D0, (int32_t)0x5269126E, (int32_t)0x9E0EFFC1,
    (int32_t)0x5133CC94, (int32_t)0x9D0DFE54, (int32_t)0x4FFB654D, (int32_t)0x9C10CD70, (int32_t)0x4EBFE8A5, (int32_t)0x9B1776DA, (int32_t)0x4D8162C4, (int32_t)0x9A22042D,
    (int32_t)0x4C3FDFF4, (int32_t)0x99307EE0, (int32_t)0x4AFB6C98, (int32_t)0x9842F043, (int32_t)0x49B41533, (int32_t)0x9759617F, (int32_t)0x4869E665, (int32_t)0x9673DB94,
    (int32_t)0x471CECE7, (int32_t)0x9592675C, (int32_t)0x45CD358F, (int32_t)0x94B50D87, (int32_t)0x447ACD50, (int32_t)0x93DBD6A0, (int32_t)0x4325C135, (int32_t)0x9306CB04,
    (int32_t)0x41CE1E65, (int32_t)0x9235F2EC, (int32_t)0x4073F21D, (int32_t)0x91695663, (int32_t)0x3F1749B8, (int32_t)0x90A0FD4E, (int32_t)0x3DB832A6, (int32_t)0x8FDCEF66,
    (int32_t)0x3C56BA70, (int32_t)0x8F1D343A, (int32_t)0x3AF2EEB7, (int32_t)0x8E61D32E, (int32_t)0x398CDD32, (int32_t)0x8DAAD37B, (int32_t)0x382493B0, (int32_t)0x8CF83C30,
    (int32_t)0x36BA2014, (int32_t)0x8C4A142F, (int32_t)0x354D9057, (int32_t)0x8BA0622F, (int32_t)0x33DEF287, (int32_t)0x8AFB2CBB, (int32_t)0x326E54C7, (int32_t)0x8A5A7A31,
    (int32_t)0x30FBC54D, (int32_t)0x89BE50C3, (int32_t)0x2F875262, (int32_t)0x8926B677, (int32_t)0x2E110A62, (int32_t)0x8893B125, (int32_t)0x2C98FBBA, (int32_t)0x88054677,
    (int32_t)0x2B1F34EB, (int32_t)0x877B7BEC, (int32_t)0x29A3C485, (int32_t)0x86F656D3, (int32_t)0x2826B928, (int32_t)0x8675DC4F, (int32_t)0x26A82186, (int32_t)0x85FA1153,
    (int32_t)0x25280C5E, (int32_t)0x8582FAA5, (int32_t)0x23A6887F, (int32_t)0x85109CDD, (int32_t)0x2223A4C5, (int32_t)0x84A2FC62, (int32_t)0x209F701C, (int32_t)0x843A1D70,
    (int32_t)0x1F19F97B, (int32_t)0x83D60412, (int32_t)0x1D934FE5, (int32_t)0x8376B422, (int32_t)0x1C0B826A, (int32_t)0x831C314E, (int32_t)0x1A82A026, (int32_t)0x82C67F14,
    (int32_t)0x18F8B83C, (int32_t)0x8275A0C0, (int32_t)0x176DD9DE, (int32_t)0x82299971, (int32_t)0x15E21445, (int32_t)0x81E26C16, (int32_t)0x145576B1, (int32_t)0x81A01B6D,
    (int32_t)0x12C8106F, (int32_t)0x8162AA04, (int32_t)0x1139F0CF, (int32_t)0x812A1A3A, (int32_t)0x0FAB272B, (int32_t)0x80F66E3C, (int32_t)0x0E1BC2E4, (int32_t)0x80C7A80A,
    (int32_t)0x0C8BD35E, (int32_t)0x809DC971, (int32_t)0x0AFB6805, (int32_t)0x8078D40D, (int32_t)0x096A9049, (int32_t)0x8058C94C, (int32_t)0x07D95B9E, (int32_t)0x803DAA6A,
    (int32_t)0x0647D97C, (int32_t)0x80277872, (int32_t)0x04B6195D, (int32_t)0x80163440, (int32_t)0x03242ABF, (int32_t)0x8009DE7E, (int32_t)0x01921D20, (int32_t)0x800277A6,
};


static const fft_real32x32_descr_t __rfft_descr =
{
    (const fft_handle_t)&__cfft_descr256_32x32,
    __fft_real_tw
};
/*
static const fft_real32x32_descr_t __rifft_descr =
{
    (const fft_handle_t)&__cifft_descr256_32x32,
    __fft_real_tw
};
*/
const fft_handle_t rfft32_512 = (const fft_handle_t)&__rfft_descr;
//const fft_handle_t rifft32_512 = (const fft_handle_t)&__rifft_descr;
