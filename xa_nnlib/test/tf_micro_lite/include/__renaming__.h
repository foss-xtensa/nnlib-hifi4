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
#define  tab_invQ30                     NatureDSP_Signal_000
#define  twiddleSplit24x24              NatureDSP_Signal_001
#define  twiddleSplit                   NatureDSP_Signal_002
#define  fft_getTables                  NatureDSP_Signal_003
#define  fft_getTables_24x24            NatureDSP_Signal_004
#define  fft_cplx_getTables             NatureDSP_Signal_005
#define  fft_getTables_hifi3            NatureDSP_Signal_006
#define  fft_getTables_24x24_hifi3      NatureDSP_Signal_007
#define  recip_table                    NatureDSP_Signal_008
#define  sine_table32                   NatureDSP_Signal_009
#define  sqrt_table                     NatureDSP_Signal_010
#define  log2_table                     NatureDSP_Signal_011
#define  atan_table                     NatureDSP_Signal_012
#define  atan_table16                   NatureDSP_Signal_013
#define  pow2_table                     NatureDSP_Signal_014
#define dct_split_32_32x16              NatureDSP_Signal_015
#define fft16_32x16                     NatureDSP_Signal_016
#define dct_split_32_24x24              NatureDSP_Signal_017
#define fft16_24x24                     NatureDSP_Signal_018
#define isplitPart_x2                   NatureDSP_Signal_019
#define splitPart_x2                    NatureDSP_Signal_020
#define ifft_core_s3                    NatureDSP_Signal_021
#define fft_core_s3                     NatureDSP_Signal_022
#define isplitPart_x2_24x24             NatureDSP_Signal_027
#define splitPart_x2_24x24              NatureDSP_Signal_028
#define ifft_cplx_24x24_lbut2           NatureDSP_Signal_029
#define ifft_cplx_24x24_lbut4           NatureDSP_Signal_030
#define ifft_cplx_24x24_core_s0         NatureDSP_Signal_031
#define ifft_cplx_24x24_core_s1         NatureDSP_Signal_032
#define ifft_cplx_24x24_core_s3         NatureDSP_Signal_033
#define fft_cplx_24x24_lbut2            NatureDSP_Signal_034
#define fft_cplx_24x24_lbut4            NatureDSP_Signal_035
#define fft_cplx_24x24_core_s0          NatureDSP_Signal_036
#define fft_cplx_24x24_core_s1          NatureDSP_Signal_037
#define fft_cplx_24x24_core_s3          NatureDSP_Signal_038
#define firdec2_proc                    NatureDSP_Signal_039
#define firdec3_proc                    NatureDSP_Signal_040
#define firdec4_proc                    NatureDSP_Signal_041
#define firdecX_proc                    NatureDSP_Signal_042
#define firinterp2_proc                 NatureDSP_Signal_047
#define firinterp3_proc                 NatureDSP_Signal_048
#define firinterp4_proc                 NatureDSP_Signal_049
#define firinterpX_proc                 NatureDSP_Signal_050
#define latr1_proc                      NatureDSP_Signal_055
#define latr2_proc                      NatureDSP_Signal_056
#define latr3_proc                      NatureDSP_Signal_057
#define latr4_proc                      NatureDSP_Signal_058
#define latr5_proc                      NatureDSP_Signal_059
#define latr6_proc                      NatureDSP_Signal_060
#define latr7_proc                      NatureDSP_Signal_061
#define latr8_proc                      NatureDSP_Signal_062
#define latrX_proc                      NatureDSP_Signal_063
#define ae_cosi                         NatureDSP_Signal_064
#define ae_twd16                        NatureDSP_Signal_065
#define ae_w                            NatureDSP_Signal_066
#define digrevtbl                       NatureDSP_Signal_067
#define inc4096                         NatureDSP_Signal_068
#define inc2048                         NatureDSP_Signal_069
#define inc1024                         NatureDSP_Signal_070
#define inc512                          NatureDSP_Signal_071
#define inc256                          NatureDSP_Signal_072
#define inc128                          NatureDSP_Signal_073
#define inc64                           NatureDSP_Signal_074
#define inc32                           NatureDSP_Signal_075
#define inc16                           NatureDSP_Signal_076
#define twd4096                         NatureDSP_Signal_077
#define twd2048                         NatureDSP_Signal_078
#define twd1024                         NatureDSP_Signal_079
#define twd512                          NatureDSP_Signal_080
#define twd256                          NatureDSP_Signal_081
#define twd128                          NatureDSP_Signal_082
#define twd64                           NatureDSP_Signal_083
#define twd32                           NatureDSP_Signal_084
#define twd16                           NatureDSP_Signal_085
#define fft_inc64                       NatureDSP_Signal_100
#define fft_inc128                      NatureDSP_Signal_101
#define fft_inc256                      NatureDSP_Signal_102
#define fft_inc512                      NatureDSP_Signal_103
#define fft_inc1024                     NatureDSP_Signal_104
#define fft_inc2048                     NatureDSP_Signal_105
#define fft_inc4096                     NatureDSP_Signal_106
#define fft_twd16                       NatureDSP_Signal_107
#define fft_twd32                       NatureDSP_Signal_108
#define fft_twd64                       NatureDSP_Signal_109
#define fft_twd128                      NatureDSP_Signal_110
#define fft_twd256                      NatureDSP_Signal_111
#define fft_twd512                      NatureDSP_Signal_112
#define fft_twd1024                     NatureDSP_Signal_113
#define fft_twd2048                     NatureDSP_Signal_114
#define fft_twd4096                     NatureDSP_Signal_115
#define ifft_twd16                      NatureDSP_Signal_116
#define ifft_twd32                      NatureDSP_Signal_117
#define ifft_twd64                      NatureDSP_Signal_118
#define ifft_twd128                     NatureDSP_Signal_119
#define ifft_twd256                     NatureDSP_Signal_120
#define ifft_twd512                     NatureDSP_Signal_121
#define ifft_twd1024                    NatureDSP_Signal_122
#define ifft_twd2048                    NatureDSP_Signal_127
#define ifft_twd4096                    NatureDSP_Signal_128
#define fft24x24_twd16                  NatureDSP_Signal_129
#define fft24x24_twd32                  NatureDSP_Signal_130
#define fft24x24_twd64                  NatureDSP_Signal_131
#define fft24x24_twd128                 NatureDSP_Signal_132
#define fft24x24_twd256                 NatureDSP_Signal_133
#define fft24x24_twd512                 NatureDSP_Signal_134
#define fft24x24_twd1024                NatureDSP_Signal_135
#define fft24x24_twd2048                NatureDSP_Signal_136
#define fft24x24_twd4096                NatureDSP_Signal_137
#define ifft24x24_twd16                 NatureDSP_Signal_138
#define ifft24x24_twd32                 NatureDSP_Signal_139
#define ifft24x24_twd64                 NatureDSP_Signal_140
#define ifft24x24_twd128                NatureDSP_Signal_150
#define ifft24x24_twd256                NatureDSP_Signal_155
#define ifft24x24_twd512                NatureDSP_Signal_156
#define ifft24x24_twd1024               NatureDSP_Signal_157
#define ifft24x24_twd2048               NatureDSP_Signal_158
#define ifft24x24_twd4096               NatureDSP_Signal_159
#define descr                           NatureDSP_Signal_160
#define seqfft_twd16                    NatureDSP_Signal_161
#define seqfft_twd32                    NatureDSP_Signal_162
#define seqfft_twd64                    NatureDSP_Signal_163
#define seqfft_twd128                   NatureDSP_Signal_164
#define seqfft_twd256                   NatureDSP_Signal_165
#define seqfft_twd512                   NatureDSP_Signal_166
#define seqfft_twd1024                  NatureDSP_Signal_167
#define seqfft_twd2048                  NatureDSP_Signal_168
#define seqfft_twd4096                  NatureDSP_Signal_169
#define seqifft_twd16                   NatureDSP_Signal_170
#define seqifft_twd32                   NatureDSP_Signal_171
#define seqifft_twd64                   NatureDSP_Signal_172
#define seqifft_twd128                  NatureDSP_Signal_173
#define seqifft_twd256                  NatureDSP_Signal_174
#define seqifft_twd512                  NatureDSP_Signal_175
#define seqifft_twd1024                 NatureDSP_Signal_176
#define seqifft_twd2048                 NatureDSP_Signal_177
#define seqifft_twd4096                 NatureDSP_Signal_178
#define firdec32x32ep_DX_proc           NatureDSP_Signal_179
#define firdec32x32ep_D2_proc           NatureDSP_Signal_180
#define firdec32x32ep_D3_proc           NatureDSP_Signal_181
#define firdec32x32ep_D4_proc           NatureDSP_Signal_182
#define firinterp32x32ep_DX_proc	      NatureDSP_Signal_183
#define firinterp32x32ep_D2_proc        NatureDSP_Signal_184
#define firinterp32x32ep_D3_proc        NatureDSP_Signal_185
#define firinterp32x32ep_D4_proc        NatureDSP_Signal_186
#define raw_corr32x32ep                 NatureDSP_Signal_187
#define fft_stage_last_ie			          NatureDSP_Signal_188
#define fft_revorder_ie				          NatureDSP_Signal_189
#define fft_pack32to24_ie			          NatureDSP_Signal_190
#define fft_unpack24to32_ie			        NatureDSP_Signal_191
#define fft_twd_r8			                NatureDSP_Signal_192
#define fir32x32ep_D2_proc              NatureDSP_Signal_193
#define fir32x32ep_D3_proc              NatureDSP_Signal_194
#define fir32x32ep_D4_proc              NatureDSP_Signal_195
#define fir32x32ep_DX_proc              NatureDSP_Signal_196
#define fft_unpack24to32_s1_ie		      NatureDSP_Signal_200
#define fft_cplx_24x24_s1_ie		        NatureDSP_Signal_201
#define pow2poly                        NatureDSP_Signal_202
#define _4log10_2minus1                 NatureDSP_Signal_203
#define log10_2                         NatureDSP_Signal_204
#define log10f_tbl                      NatureDSP_Signal_205
#define alog10fminmax                   NatureDSP_Signal_206
#define invlog10_2_Q29                  NatureDSP_Signal_207
#define alog2fminmax                    NatureDSP_Signal_208
#define atanftbl1                       NatureDSP_Signal_209
#define atanftbl2                       NatureDSP_Signal_210
#define expfminmax                      NatureDSP_Signal_211
#define expftbl_Q30                     NatureDSP_Signal_212
#define invln2_Q30                      NatureDSP_Signal_213
#define fir_decimaf_2x                  NatureDSP_Signal_214
#define fir_decimaf_3x                  NatureDSP_Signal_215
#define fir_decimaf_4x                  NatureDSP_Signal_216
#define fir_decimaf_Dx                  NatureDSP_Signal_217
#define fir_interpf_2x                  NatureDSP_Signal_218
#define fir_interpf_3x                  NatureDSP_Signal_219
#define fir_interpf_4x                  NatureDSP_Signal_220
#define fir_interpf_Dx                  NatureDSP_Signal_221
#define inv2pif_Q53                     NatureDSP_Signal_222
#define inv4pif                         NatureDSP_Signal_223
#define latrf_process1                  NatureDSP_Signal_224
#define latrf_process2                  NatureDSP_Signal_225
#define latrf_process3                  NatureDSP_Signal_226
#define latrf_process4                  NatureDSP_Signal_227
#define latrf_process5                  NatureDSP_Signal_228
#define latrf_process6                  NatureDSP_Signal_229
#define latrf_process7                  NatureDSP_Signal_230
#define latrf_process8                  NatureDSP_Signal_231
#define latrf_processX                  NatureDSP_Signal_232
#define ln2                             NatureDSP_Signal_233
#define log2f_tbl                       NatureDSP_Signal_234
#define minus_qNaN                      NatureDSP_Signal_235
#define minus_sNaN                      NatureDSP_Signal_236
#define qNaN                            NatureDSP_Signal_237
#define sNaN                            NatureDSP_Signal_238
#define minus_qNaNf                     NatureDSP_Signal_239
#define minus_sNaNf                     NatureDSP_Signal_240
#define qNaNf                           NatureDSP_Signal_241
#define sNaNf                           NatureDSP_Signal_242
#define minusInff                       NatureDSP_Signal_243
#define plusInff                        NatureDSP_Signal_244
#define realmaxf                        NatureDSP_Signal_245
#define pi2f                            NatureDSP_Signal_246
#define pi2m1f                          NatureDSP_Signal_247
#define pi4f                            NatureDSP_Signal_248
#define pif                             NatureDSP_Signal_249
#define pi4fc                           NatureDSP_Signal_250
#define polycosf_tbl                    NatureDSP_Signal_251
#define polysinf_tbl                    NatureDSP_Signal_252
#define sinf_maxval                     NatureDSP_Signal_253
#define polytanf_tbl                    NatureDSP_Signal_254
#define tanf_maxval                     NatureDSP_Signal_255
#define raw_corrf                       NatureDSP_Signal_256
#define sine_table16                    NatureDSP_Signal_257
#define sqrt0_5f                        NatureDSP_Signal_258
#define sqrt2f                          NatureDSP_Signal_259
#define lognf_tbl                       NatureDSP_Signal_260
#define _invsqrt2f                      NatureDSP_Signal_261
#define cnst01                          NatureDSP_Signal_262
#define cnst0123                        NatureDSP_Signal_263
#define dct_twd64                       NatureDSP_Signal_264
#define fn                              NatureDSP_Signal_265
#define fxns                            NatureDSP_Signal_266
#define ifft_twd_r8                     NatureDSP_Signal_267
#define inv2pif                         NatureDSP_Signal_268
#define fft24x24_stage_last_s1          NatureDSP_Signal_269
#define isplitPart_x2_24x24_s1          NatureDSP_Signal_270
#define isplitPart_x2_24x24_shift2      NatureDSP_Signal_271
#define splitPart_x2_24x24_s1           NatureDSP_Signal_272

#define  __cfft_descr16_32x32           NatureDSP_Signal_273
#define  __cfft_descr32_32x32           NatureDSP_Signal_274
#define  __cfft_descr64_32x32           NatureDSP_Signal_275
#define  __cfft_descr128_32x32          NatureDSP_Signal_276
#define  __cfft_descr256_32x32          NatureDSP_Signal_277
#define  __cfft_descr512_32x32          NatureDSP_Signal_278
#define  __cfft_descr1024_32x32         NatureDSP_Signal_279
#define  __cfft_descr2048_32x32         NatureDSP_Signal_280
#define  __cfft_descr4096_32x32         NatureDSP_Signal_281
#define  __cfft_descr12_32x32           NatureDSP_Signal_282
#define  __cfft_descr24_32x32           NatureDSP_Signal_283
#define  __cfft_descr36_32x32           NatureDSP_Signal_284
#define  __cfft_descr48_32x32           NatureDSP_Signal_285
#define  __cfft_descr60_32x32           NatureDSP_Signal_286
#define  __cfft_descr72_32x32           NatureDSP_Signal_287
#define  __cfft_descr96_32x32           NatureDSP_Signal_288
#define  __cfft_descr108_32x32          NatureDSP_Signal_289
#define  __cfft_descr120_32x32          NatureDSP_Signal_290
#define  __cfft_descr144_32x32          NatureDSP_Signal_291
#define  __cfft_descr180_32x32          NatureDSP_Signal_292
#define  __cfft_descr192_32x32          NatureDSP_Signal_293
#define  __cfft_descr216_32x32          NatureDSP_Signal_294
#define  __cfft_descr240_32x32          NatureDSP_Signal_295
#define  __cfft_descr288_32x32          NatureDSP_Signal_296
#define  __cfft_descr300_32x32          NatureDSP_Signal_297
#define  __cfft_descr324_32x32          NatureDSP_Signal_298
#define  __cfft_descr360_32x32          NatureDSP_Signal_299
#define  __cfft_descr432_32x32          NatureDSP_Signal_300
#define  __cfft_descr480_32x32          NatureDSP_Signal_301
#define  __cfft_descr540_32x32          NatureDSP_Signal_302
#define  __cfft_descr576_32x32          NatureDSP_Signal_303
#define  __cfft_descr768_32x32          NatureDSP_Signal_304
#define  __cfft_descr960_32x32          NatureDSP_Signal_305
#define  __cifft_descr16_32x32          NatureDSP_Signal_306
#define  __cifft_descr32_32x32          NatureDSP_Signal_307
#define  __cifft_descr64_32x32          NatureDSP_Signal_308
#define  __cifft_descr128_32x32         NatureDSP_Signal_309
#define  __cifft_descr256_32x32         NatureDSP_Signal_310
#define  __cifft_descr512_32x32         NatureDSP_Signal_311
#define  __cifft_descr1024_32x32        NatureDSP_Signal_312
#define  __cifft_descr2048_32x32        NatureDSP_Signal_313
#define  __cifft_descr4096_32x32        NatureDSP_Signal_314
#define  __cifft_descr12_32x32          NatureDSP_Signal_315
#define  __cifft_descr24_32x32          NatureDSP_Signal_316
#define  __cifft_descr36_32x32          NatureDSP_Signal_317
#define  __cifft_descr48_32x32          NatureDSP_Signal_318
#define  __cifft_descr60_32x32          NatureDSP_Signal_319
#define  __cifft_descr72_32x32          NatureDSP_Signal_320
#define  __cifft_descr96_32x32          NatureDSP_Signal_321
#define  __cifft_descr108_32x32         NatureDSP_Signal_322
#define  __cifft_descr120_32x32         NatureDSP_Signal_323
#define  __cifft_descr144_32x32         NatureDSP_Signal_324
#define  __cifft_descr180_32x32         NatureDSP_Signal_325
#define  __cifft_descr192_32x32         NatureDSP_Signal_326
#define  __cifft_descr216_32x32         NatureDSP_Signal_327
#define  __cifft_descr240_32x32         NatureDSP_Signal_328
#define  __cifft_descr288_32x32         NatureDSP_Signal_329
#define  __cifft_descr300_32x32         NatureDSP_Signal_330
#define  __cifft_descr324_32x32         NatureDSP_Signal_331
#define  __cifft_descr360_32x32         NatureDSP_Signal_332
#define  __cifft_descr432_32x32         NatureDSP_Signal_333
#define  __cifft_descr480_32x32         NatureDSP_Signal_334
#define  __cifft_descr540_32x32         NatureDSP_Signal_335
#define  __cifft_descr576_32x32         NatureDSP_Signal_336
#define  __cifft_descr768_32x32         NatureDSP_Signal_337
#define  __cifft_descr960_32x32         NatureDSP_Signal_338

#define  _fft_twiddle_table_128_        NatureDSP_Signal_339
#define  stage_inner_DFT4_16x16_ie      NatureDSP_Signal_340
#define  fft_stageS2_DFT2_last_32x32    NatureDSP_Signal_341
#define  fft_stageS2_DFT2_32x32         NatureDSP_Signal_342
#define  fft_stageS2_DFT2_first_32x32   NatureDSP_Signal_343
#define  fft_stageS2_DFT3_32x32         NatureDSP_Signal_344
#define  fft_stageS2_DFT3_first_32x32   NatureDSP_Signal_345
#define  fft_stageS2_DFT3_last_32x32    NatureDSP_Signal_346
#define  fft_stageS2_DFT4_32x32         NatureDSP_Signal_347
#define  fft_stageS2_DFT4_first_32x32   NatureDSP_Signal_348
#define  fft_stageS2_DFT4_last_32x32    NatureDSP_Signal_349
#define  fft_stageS2_DFT5_32x32         NatureDSP_Signal_350
#define  fft_stageS2_DFT5_first_32x32   NatureDSP_Signal_351
#define  fft_stageS2_DFT5_last_32x32    NatureDSP_Signal_352
#define  ifft_stageS2_DFT2_32x32        NatureDSP_Signal_353
#define  ifft_stageS2_DFT2_first_32x32  NatureDSP_Signal_354
#define  ifft_stageS2_DFT3_32x32        NatureDSP_Signal_355
#define  ifft_stageS2_DFT3_first_32x32  NatureDSP_Signal_356
#define  ifft_stageS2_DFT4_32x32        NatureDSP_Signal_357
#define  ifft_stageS2_DFT4_first_32x32  NatureDSP_Signal_358
#define  ifft_stageS2_DFT5_32x32        NatureDSP_Signal_359
#define  ifft_stageS2_DFT5_first_32x32  NatureDSP_Signal_360
#define  fft_stageS3_DFT2_32x32         NatureDSP_Signal_361
#define  fft_stageS3_DFT2_first_32x32   NatureDSP_Signal_362
#define  fft_stageS3_DFT2_last_32x32    NatureDSP_Signal_363
#define  fft_stageS3_DFT3_32x32         NatureDSP_Signal_364
#define  fft_stageS3_DFT3_first_32x32   NatureDSP_Signal_365
#define  fft_stageS3_DFT3_last_32x32    NatureDSP_Signal_366
#define  fft_stageS3_DFT4_32x32         NatureDSP_Signal_367
#define  fft_stageS3_DFT4_first_32x32   NatureDSP_Signal_368
#define  fft_stageS3_DFT4_last_32x32    NatureDSP_Signal_369
#define  fft_stageS3_DFT5_32x32         NatureDSP_Signal_370
#define  fft_stageS3_DFT5_first_32x32   NatureDSP_Signal_371
#define  fft_stageS3_DFT5_last_32x32    NatureDSP_Signal_372
#define  fft_stageS3_DFT8_last_32x32    NatureDSP_Signal_373
#define  ifft_stageS3_DFT2_32x32        NatureDSP_Signal_374
#define  ifft_stageS3_DFT2_first_32x32  NatureDSP_Signal_375
#define  ifft_stageS3_DFT3_32x32        NatureDSP_Signal_376
#define  ifft_stageS3_DFT3_first_32x32  NatureDSP_Signal_377
#define  ifft_stageS3_DFT4_32x32        NatureDSP_Signal_378
#define  ifft_stageS3_DFT4_first_32x32  NatureDSP_Signal_379
#define  ifft_stageS3_DFT5_32x32        NatureDSP_Signal_380
#define  ifft_stageS3_DFT5_first_32x32  NatureDSP_Signal_381

#define  raw_corr16x16                  NatureDSP_Signal_382
#define  raw_corr32x32                  NatureDSP_Signal_383
#define  raw_lxcorr16x16                NatureDSP_Signal_384
#define  fir_lxcorr32x32                NatureDSP_Signal_385
#define  polyrsqrtq23                   NatureDSP_Signal_386
#define  polyatan24x24q23               NatureDSP_Signal_387
#define  firinterp32x32_D2_proc         NatureDSP_Signal_388
#define  firinterp32x32_D3_proc         NatureDSP_Signal_389
#define  firinterp32x32_D4_proc         NatureDSP_Signal_390
#define  firinterp32x32_DX_proc         NatureDSP_Signal_391
#define  firinterp16x16_D2_proc         NatureDSP_Signal_392
#define  firinterp16x16_D3_proc         NatureDSP_Signal_393
#define  firinterp16x16_D4_proc         NatureDSP_Signal_394
#define  firinterp16x16_DX_proc         NatureDSP_Signal_395
#define  fir32x32_D2_proc               NatureDSP_Signal_396
#define  fir32x32_D3_proc               NatureDSP_Signal_397
#define  fir32x32_D4_proc               NatureDSP_Signal_398
#define  fir32x32_DX_proc               NatureDSP_Signal_399
#define  fir16x16_D2_proc               NatureDSP_Signal_400
#define  fir16x16_D3_proc               NatureDSP_Signal_401
#define  fir16x16_D4_proc               NatureDSP_Signal_402
#define  fir16x16_DX_proc               NatureDSP_Signal_403
#define  _fft_twiddle_table_8_          NatureDSP_Signal_404
#define  _fft_twiddle_table_16_         NatureDSP_Signal_405
#define  _fft_twiddle_table_32_         NatureDSP_Signal_406
#define  _fft_twiddle_table_64_         NatureDSP_Signal_407
#define  _fft_twiddle_table_256_        NatureDSP_Signal_408
#define  _fft_twiddle_table_512_        NatureDSP_Signal_419
#define  _fft_twiddle_table_1024_       NatureDSP_Signal_410
#define  _fft_twiddle_table_2048_       NatureDSP_Signal_411
#define  _fft_twiddle_table_4096_       NatureDSP_Signal_412
#define  fft_twd8                       NatureDSP_Signal_413
#define  ifft_twd8                      NatureDSP_Signal_414
#define  fft_stageS2_DFT8_last_32x32    NatureDSP_Signal_415
