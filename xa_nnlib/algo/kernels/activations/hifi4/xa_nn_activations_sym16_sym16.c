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
#include "xa_nnlib_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros.h"

#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4))
static const uint16_t sigmoid_table_uint16[256] = {
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498,
    40149, 40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255,
    46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865,
    52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174,
    56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544, 58800, 59048, 59288,
    59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441,
    61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886,
    62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
    63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450,
    64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845,
    64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097,
    65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
    65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425,
    65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465,
    65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
    65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518,
    65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
    65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529,
    65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531,
    65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533,
    65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
    65534, 65534, 65535};

#endif

void internal_vec_sigmoid_sym16s_sym16s_spc(WORD16 *p_out,
                                                   const WORD16 *p_vec,
                                                   WORD32 vec_length)
{
  ae_int16x4 *  __restrict__ in_ptr_align  = (ae_int16x4 *)p_vec;
  WORD16 *  __restrict__ out_ptr = (WORD16 *)p_out;

  ae_valign inp_align = AE_LA64_PP(in_ptr_align);
  ae_valign  out_align   = AE_ZALIGN64();

  ae_int16x4 inp0;
#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4))
  ae_int32x2 inp_x_inp_mul0, inp_x_inp_mul1;
  ae_int32x2 uint8_max     = 255;
  ae_int32x2 res_sat_val   = 33553408; /* Value for saturation: 33553408 = (0x7FFF << 10) */
  ae_int16x4 mask_nine_bit = 511;
  ae_int32x2 add_val       = 512;
  ae_int32x2 sub_val       = 33554943; /* Const required for negative inputs: 33554943 = [(1 << (16 + 9)) + (1<<9) -1 */
  ae_int32x2 abs_inp_x_inp_mul0, abs_inp_x_inp_mul1;
  ae_int32x2 ua0, ua1, ub0, ub1;
  ae_int32x2 ua_lsh0, ua_lsh1, ub_minus_ua0, ub_minus_ua1;
  ae_int32x2 res0, res1, res0_plus, res1_plus;
  ae_int32x2 uh_0, uh_1;
  xtbool2 x0, x1;
  ae_int16x4 ut;
  int input_multiplier = 3;
  ae_int16x4 inp_mult = input_multiplier;
#else
  ae_int16x4 sign_bit_mask = AE_MOVDA16(0x7FFF);
  WUR_AE_SAR(4);
#endif  

  int i;
  for (i = 0; i < vec_length >> 2; i++) {
    AE_LA16X4_IP(inp0, inp_align, in_ptr_align);
#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
    ae_int16x4 out = AE_SIGMOID16X4(inp0);
    out = AE_SRAI16(out, 1);
    out = AE_AND16(out, sign_bit_mask);
#else
    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);
    abs_inp_x_inp_mul1 = AE_ABS32S(inp_x_inp_mul1);

    ut = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul1));
    ut = AE_AND16(ut, mask_nine_bit);

    uh_0 = AE_SRAI32(abs_inp_x_inp_mul0, 9);
    uh_1 = AE_SRAI32(abs_inp_x_inp_mul1, 9);

    /*
     *  Following is the alternate code
    ua0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0)], sigmoid_table_uint16[AE_MOVAD32_L(uh_0)]); 
    ua1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1)], sigmoid_table_uint16[AE_MOVAD32_L(uh_1)]); 
    ub0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_0) + 1]); 
    ub1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_1) + 1]); 
    */
    /*From here*/
    int id0,id1,id2,id3;
    id0 = AE_MOVAD32_H(AE_SLAI32(uh_0, 1));
    id1 = AE_MOVAD32_L(AE_SLAI32(uh_0, 1));
    id2 = AE_MOVAD32_H(AE_SLAI32(uh_1, 1));
    id3 = AE_MOVAD32_L(AE_SLAI32(uh_1, 1));

    ae_int16 *psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 zero_16x4 = AE_ZERO16();

    ae_int16x4 sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4); 
    ua0 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));
 
    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ua1  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));

    psigmoid_table_uint16++;

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4);
    ub0  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ub1 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));
    /*Till here */

    ua_lsh0 = AE_SLAI32S(ua0, 9);
    ua_lsh1 = AE_SLAI32S(ua1, 9);

    ub_minus_ua0 = AE_SUB32S(ub0, ua0);
    ub_minus_ua1 = AE_SUB32S(ub1, ua1);

    res0 = ua_lsh0;
    res1 = ua_lsh1;

    AE_MULAP32X16X2_H(res0, ub_minus_ua0, ut);
    AE_MULAP32X16X2_L(res1, ub_minus_ua1, ut);
  
    x0 = AE_LT32(uh_0, uint8_max);
    x1 = AE_LT32(uh_1, uint8_max);
    AE_MOVF32X2(res0, res_sat_val, x0); 
    AE_MOVF32X2(res1, res_sat_val, x1); 

    res0_plus = AE_ADD32S(res0, add_val);
    res1_plus = AE_ADD32S(res1, add_val);

    res0 = AE_SUB32S(sub_val, res0);
    res1 = AE_SUB32S(sub_val, res1);
 
    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());
    x1 = AE_LT32(inp_x_inp_mul1, AE_ZERO32());

    AE_MOVF32X2(res0, res0_plus, x0);
    AE_MOVF32X2(res1, res1_plus, x1);

    res0 = AE_SRAI32(res0, 10);
    res1 = AE_SRAI32(res1, 10);

    ae_int16x4 out = AE_SAT16X4(res0, res1);
#endif
    AE_SA16X4_IP(out, out_align, (ae_int16x4 *)out_ptr); 
  }

  AE_SA64POS_FP(out_align, out_ptr);
  p_vec = (WORD16 *)in_ptr_align;
  p_out = (WORD16 *)out_ptr;

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
  for (i = 0; i < (vec_length & 3); i++) {
    AE_L16_IP(inp0, (ae_int16 *)p_vec, sizeof(WORD16));    
    ae_int16x4 out = AE_SIGMOID16X4(inp0);
    out = AE_SRAI16(out, 1);
    out = AE_AND16(out, sign_bit_mask);    
    AE_S16_0_IP(out, (ae_int16 *)p_out, sizeof(WORD16));  
  }
#else
  /* Following code is directly adapted from TFLM ref code */
  for (i = 0; i < (vec_length & 3); ++i, p_vec++, p_out++) {
    WORD32 input_data = ((*p_vec) * input_multiplier);
    UWORD32 abs_input_data =(UWORD32)AE_MOVAD32_L(AE_ABS32S(AE_MOVDA32(input_data)));
    UWORD32 uh = abs_input_data >> 9;
    UWORD32 result;
    if (uh >= 255) {
      result = 0x7FFF << 10;
    } 
    else {
      UWORD32 ua = sigmoid_table_uint16[uh];
      UWORD32 ub = sigmoid_table_uint16[uh + 1];
      UWORD32 ut32 = abs_input_data & 0x1ff;
      result = (ua << 9) + ut32 * (ub - ua);
    }

    result = (input_data >= 0) ? (result + (1 << 9)) : ((1 << (16 + 9)) - result + (1 << 9) - 1);
    result >>= 10;
    *p_out =(WORD16)result;
  }
#endif
}

/* The scale of input for TFLM reference is 4096*3, which is maintained in the LUT based implementation 
 * in xa_nn_vec_sigmoid_sym16s_sym16s(). 
 * However, the TIE based implementation uses input scale of 4096. The corresponding scaling by 3 in 
 * TFLM prepare function is removed for this implemenation.
 */
WORD32 xa_nn_vec_sigmoid_sym16s_sym16s(WORD16 *p_out,
                      const WORD16 *p_vec,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);  
  XA_NNLIB_ARG_CHK_COND((input_left_shift < 0), -1); 

  if(input_multiplier == 0 && input_left_shift==0){
    internal_vec_sigmoid_sym16s_sym16s_spc(p_out,p_vec, vec_length);
    return 0;
  }

  if (input_multiplier == 0) {  
#if (defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4))    
    input_multiplier = 1 << input_left_shift;
#else 
    input_multiplier = 3 << input_left_shift;
#endif
    input_left_shift = 0;
  }

  WORD32 round = (input_left_shift > 0) ? 1 << (input_left_shift - 1) : 0;

  ae_int16x4 inp_mult = input_multiplier; 
  ae_int16x4 * __restrict__ in_ptr_align  = (ae_int16x4 *)p_vec;
  WORD16 * __restrict__ out_ptr = (WORD16 *)p_out;

  ae_valign inp_align = AE_LA64_PP(in_ptr_align);
  ae_valign  out_align   = AE_ZALIGN64();

  ae_int16x4 inp0;
  ae_int32x2 inp_x_inp_mul0, inp_x_inp_mul1;
#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4))
  ae_int32x2 uint8_max     = 255;
  ae_int32x2 res_sat_val   = 33553408; /* Value for saturation: 33553408 = (0x7FFF << 10) */
  ae_int16x4 mask_nine_bit = 511;
  ae_int32x2 add_val       = 512;
  ae_int32x2 sub_val       = 33554943; /* Const required for negative inputs: 33554943 = [(1 << (16 + 9)) + (1<<9) -1 */
  ae_int32x2 abs_inp_x_inp_mul0, abs_inp_x_inp_mul1;
  ae_int32x2 ua0, ua1, ub0, ub1;
  ae_int32x2 ua_lsh0, ua_lsh1, ub_minus_ua0, ub_minus_ua1;
  ae_int32x2 res0, res1, res0_plus, res1_plus;
  ae_int32x2 uh_0, uh_1;
  xtbool2 x0, x1;
  ae_int16x4 ut;
#else
  ae_int16x4 sign_bit_mask = AE_MOVDA16(0x7FFF);
  ae_int16x4 sigmoid_in;
  WUR_AE_SAR(4);
#endif
  int i;
  for (i = 0; i < vec_length >> 2; i++) {

    AE_LA16X4_IP(inp0, inp_align, in_ptr_align);    

    inp_x_inp_mul0 = round;
    inp_x_inp_mul1 = round;

    AE_MULA16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    inp_x_inp_mul0 = AE_SRAA32(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32(inp_x_inp_mul1, input_left_shift);

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
    sigmoid_in = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1);
    ae_int16x4 out = AE_SIGMOID16X4(sigmoid_in);
    out = AE_SRAI16(out, 1);
    out = AE_AND16(out, sign_bit_mask);
#else
    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);
    abs_inp_x_inp_mul1 = AE_ABS32S(inp_x_inp_mul1);

    ut = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul1));
    ut = AE_AND16(ut, mask_nine_bit);

    uh_0 = AE_SRAI32(abs_inp_x_inp_mul0, 9);
    uh_1 = AE_SRAI32(abs_inp_x_inp_mul1, 9);

#if XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI1
    ua0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0)], sigmoid_table_uint16[AE_MOVAD32_L(uh_0)]); 
    ua1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1)], sigmoid_table_uint16[AE_MOVAD32_L(uh_1)]); 
    ub0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_0) + 1]); 
    ub1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_1) + 1]); 
#else
    int id0,id1,id2,id3;
    id0 = AE_MOVAD32_H(AE_SLAI32(uh_0, 1));
    id1 = AE_MOVAD32_L(AE_SLAI32(uh_0, 1));
    id2 = AE_MOVAD32_H(AE_SLAI32(uh_1, 1));
    id3 = AE_MOVAD32_L(AE_SLAI32(uh_1, 1));

    ae_int16 * __restrict__ psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 zero_16x4 = AE_ZERO16();

    ae_int16x4 sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4); 
    ua0 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));
 
    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ua1  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));

    psigmoid_table_uint16++;

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4);
    ub0  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ub1 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));
#endif
    ua_lsh0 = AE_SLAI32S(ua0, 9);
    ua_lsh1 = AE_SLAI32S(ua1, 9);

    ub_minus_ua0 = AE_SUB32S(ub0, ua0);
    ub_minus_ua1 = AE_SUB32S(ub1, ua1);

    res0 = ua_lsh0;
    res1 = ua_lsh1;

    AE_MULAP32X16X2_H(res0, ub_minus_ua0, ut);
    AE_MULAP32X16X2_L(res1, ub_minus_ua1, ut);
  
    x0 = AE_LT32(uh_0, uint8_max);
    x1 = AE_LT32(uh_1, uint8_max);
    AE_MOVF32X2(res0, res_sat_val, x0); 
    AE_MOVF32X2(res1, res_sat_val, x1); 

    res0_plus = AE_ADD32S(res0, add_val);
    res1_plus = AE_ADD32S(res1, add_val);

    res0 = AE_SUB32S(sub_val, res0);
    res1 = AE_SUB32S(sub_val, res1);
 
    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());
    x1 = AE_LT32(inp_x_inp_mul1, AE_ZERO32());

    AE_MOVF32X2(res0, res0_plus, x0);
    AE_MOVF32X2(res1, res1_plus, x1);

    res0 = AE_SRAI32(res0, 10);
    res1 = AE_SRAI32(res1, 10);

    ae_int16x4 out = AE_SAT16X4(res0, res1);
#endif
    AE_SA16X4_IP(out, out_align, (ae_int16x4 *)out_ptr); 
  }

  AE_SA64POS_FP(out_align, out_ptr);
  p_vec = (WORD16 *)in_ptr_align;
  p_out = (WORD16 *)out_ptr;

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4)
  for (i = 0; i < (vec_length & 3); i++) {
    AE_L16_IP(inp0, (ae_int16 *)p_vec, sizeof(WORD16));    
    inp_x_inp_mul0 = round;
    inp_x_inp_mul1 = round;
    AE_MULA16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    inp_x_inp_mul0 = AE_SRAA32(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32(inp_x_inp_mul1, input_left_shift); 
    sigmoid_in = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1);
    ae_int16x4 out = AE_SIGMOID16X4(sigmoid_in);
    out = AE_SRAI16(out, 1);
    out = AE_AND16(out, sign_bit_mask);    
    AE_S16_0_IP(out, (ae_int16 *)p_out, sizeof(WORD16));  
  }
#else
  /* Following code is directly adapted from TFLM ref code */
  for (i = 0; i < (vec_length & 3); ++i, p_vec++, p_out++) {
    WORD32 input_data = ((*p_vec) * input_multiplier + round) >> input_left_shift;

    UWORD32 abs_input_data = (UWORD32)AE_MOVAD32_L(AE_ABS32S(AE_MOVDA32(input_data)));

    UWORD32 uh = abs_input_data >> 9;
    UWORD32 result;

    if (uh >= 255) {
      result = 0x7FFF << 10;
    } 
    else {
      UWORD32 ua = sigmoid_table_uint16[uh];
      UWORD32 ub = sigmoid_table_uint16[uh + 1];
      UWORD32 ut32 = abs_input_data & 0x1ff;
      result = (ua << 9) + ut32 * (ub - ua);
    }

    result = (input_data >= 0) ? (result + (1 << 9)) : ((1 << (16 + 9)) - result + (1 << 9) - 1);
    result >>= 10;
    *p_out =(WORD16)result;
  }
#endif
  return 0;
}

/* The scale of input for TFLM reference is 4096*3, which is maintained in the LUT based implementation 
 * in xa_nn_vec_tanh_sym16s_sym16s(). 
 * However, the TIE based implementation uses input scale of 4096. The corresponding scaling by 3 in 
 * TFLM prepare function is removed for this implemenation.
 */
WORD32 xa_nn_vec_tanh_sym16s_sym16s(WORD16 *p_out,
                      const WORD16 *p_vec,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{

  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);  

  if (input_multiplier == 0) {  
#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4))    
    input_multiplier = 1 << input_left_shift;
#else 
    input_multiplier = 3 << input_left_shift;
#endif
    input_left_shift = 0;
  }

  WORD32 round = (input_left_shift > 0) ? 1 << (input_left_shift - 1) : 0;

  ae_int16x4 inp_mult = input_multiplier; 
  ae_int16x4 * __restrict__ in_ptr_align  = (ae_int16x4 *)p_vec;
  WORD16 * __restrict__ out_ptr = (WORD16 *)p_out;

  ae_valign inp_align    = AE_LA64_PP(in_ptr_align);
  ae_valign  out_align   = AE_ZALIGN64();

  ae_int16x4 inp0;
  ae_int32x2 inp_x_inp_mul0, inp_x_inp_mul1;
#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4))
  ae_int32x2 uint8_max       = 255;
  ae_int32x2 res_sat_val     = 16776960; /* Saturation value: 16776960 = [0xFFFF << 8] */
  ae_int16x4 mask_eight_bit  = 255;
  /* Constants for adjustmnets for negative and positive inputs (as per TFLM implementation)
   * 8388480 = [(1 << (14 + 9)) - (1 << (9 - 2))]
   * 8388735 = [(1 << (14 + 9)) + (1 << (9 - 2)) - 1]
   */
  ae_int32x2 sub_val0        = 8388480;
  ae_int32x2 sub_val1        = 8388735; 
  ae_int32x2 abs_inp_x_inp_mul0, abs_inp_x_inp_mul1;
  ae_int32x2 ua0, ua1, ub0, ub1;
  ae_int32x2 ua_lsh0, ua_lsh1, ub_minus_ua0, ub_minus_ua1;
  ae_int32x2 res0, res1, res0_minus, res1_minus;
  ae_int32x2 uh_0, uh_1;
  xtbool2 x0, x1;
  ae_int16x4 ut;
#else
  ae_int16x4 tanh_in;
  WUR_AE_SAR(4);
#endif

  int i;
  for (i = 0; i < (vec_length >> 2); i++) {

    AE_LA16X4_IP(inp0, inp_align, in_ptr_align);    

    inp_x_inp_mul0 = round;
    inp_x_inp_mul1 = round;

    AE_MULA16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32(inp_x_inp_mul1, input_left_shift);

#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4))
    tanh_in = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1);
    ae_int16x4 out = AE_TANH16X4(tanh_in);
#else
    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);
    abs_inp_x_inp_mul1 = AE_ABS32S(inp_x_inp_mul1);

    ut = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul1));
    ut = AE_AND16(ut, mask_eight_bit);

    uh_0 = AE_SRAI32(abs_inp_x_inp_mul0, 8);
    uh_1 = AE_SRAI32(abs_inp_x_inp_mul1, 8);
    
#if XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI1
    ua0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0)], sigmoid_table_uint16[AE_MOVAD32_L(uh_0)]); 
    ua1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1)], sigmoid_table_uint16[AE_MOVAD32_L(uh_1)]); 
    ub0 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_0) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_0) + 1]); 
    ub1 = AE_MOVDA32X2(sigmoid_table_uint16[AE_MOVAD32_H(uh_1) + 1], sigmoid_table_uint16[AE_MOVAD32_L(uh_1) + 1]); 
#else
    int id0,id1,id2,id3;
    id0 = AE_MOVAD32_H(AE_SLAI32(uh_0, 1));
    id1 = AE_MOVAD32_L(AE_SLAI32(uh_0, 1));
    id2 = AE_MOVAD32_H(AE_SLAI32(uh_1, 1));
    id3 = AE_MOVAD32_L(AE_SLAI32(uh_1, 1));

    ae_int16 * __restrict__ psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 zero_16x4 = AE_ZERO16();

    ae_int16x4 sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4); 
    ua0 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));
 
    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ua1  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));

    psigmoid_table_uint16++;

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1), zero_16x4);
    ub0  = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0)));

    sel0 = AE_SEL16_7610(AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3), zero_16x4);
    ub1 = AE_MOVINT32X2_FROMINT16X4(AE_SEL16_5146(sel0, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2)));
#endif

    ua_lsh0 = AE_SLAI32S(ua0, 8);
    ua_lsh1 = AE_SLAI32S(ua1, 8);

    ub_minus_ua0 = AE_SUB32S(ub0, ua0);
    ub_minus_ua1 = AE_SUB32S(ub1, ua1);

    res0 = ua_lsh0;
    res1 = ua_lsh1;

    AE_MULAP32X16X2_H(res0, ub_minus_ua0, ut);
    AE_MULAP32X16X2_L(res1, ub_minus_ua1, ut);
    x0 = AE_LT32(uh_0, uint8_max);
    x1 = AE_LT32(uh_1, uint8_max);
    AE_MOVF32X2(res0, res_sat_val, x0); 
    AE_MOVF32X2(res1, res_sat_val, x1); 

    res0_minus = AE_SUB32S(res0, sub_val0);
    res1_minus = AE_SUB32S(res1, sub_val0);
    res0 = AE_SUB32S(sub_val1, res0);
    res1 = AE_SUB32S(sub_val1, res1);

    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());
    x1 = AE_LT32(inp_x_inp_mul1, AE_ZERO32());

    AE_MOVF32X2(res0, res0_minus, x0);
    AE_MOVF32X2(res1, res1_minus, x1);

    res0 = AE_SRAI32(res0, 8);
    res1 = AE_SRAI32(res1, 8);
    ae_int16x4 out = AE_SAT16X4(res0, res1);
#endif
    AE_SA16X4_IP(out, out_align, (ae_int16x4 *)out_ptr);
  }

  AE_SA64POS_FP(out_align, out_ptr);
  p_vec = (WORD16 *)in_ptr_align;
  p_out = (WORD16 *)out_ptr;

#if defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4)
  for (i = 0; i < (vec_length & 3); i++) {
    AE_L16_IP(inp0, (ae_int16 *)p_vec, sizeof(WORD16));    
    inp_x_inp_mul0 = round;
    inp_x_inp_mul1 = round;
    AE_MULA16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    inp_x_inp_mul0 = AE_SRAA32(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32(inp_x_inp_mul1, input_left_shift); 
    tanh_in = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1); 
    ae_int16x4 out = AE_TANH16X4(tanh_in);   
    AE_S16_0_IP(out, (ae_int16 *)p_out, sizeof(WORD16));  
  }
#else
  /* Following code is directly adapted from TFLM ref code */
  for (i = 0; i < (vec_length & 3); ++i, p_vec++, p_out++) {
    WORD32 input_data =
        ((*p_vec) * input_multiplier + round) >> input_left_shift;

    UWORD32 abs_input_data = (UWORD32)AE_MOVAD32_L(AE_ABS32S(AE_MOVDA32(input_data)));
    UWORD32 uh = abs_input_data >> 8;
    WORD32 result;

    if (uh >= 255) {
      result = 0xFFFF << 8;
    } else {
      UWORD32 ua = sigmoid_table_uint16[uh];
      UWORD32 ub = sigmoid_table_uint16[uh + 1];

      UWORD8 ut32 = abs_input_data & 0xFF;

      result = (ua << 8) + ut32 * (ub - ua);
    }

    result = (input_data >= 0)
                 ? (result - (1 << (14 + 9)) + (1 << (9 - 2)))
                 : (-result + (1 << (14 + 9)) + (1 << (9 - 2)) - 1);

    result >>= (9 - 1);

    *p_out = (WORD16)result;
  }
#endif
  return 0;
}
