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
// Generated file (from: conv_quant8_2.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type3(Type::INT32, {});
  OperandType type2(Type::TENSOR_INT32, {1}, 0.25f, 0);
  OperandType type1(Type::TENSOR_QUANT8_ASYMM, {1, 2, 2, 1}, 0.5f, 127);
  OperandType type4(Type::TENSOR_QUANT8_ASYMM, {1, 2, 2, 1}, 1.f, 127);
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, {1, 3, 6, 1}, 0.5f, 127);
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto op3 = model->addOperand(&type2);
  auto pad_valid = model->addOperand(&type3);
  auto act_none = model->addOperand(&type3);
  auto stride1 = model->addOperand(&type3);
  auto stride3 = model->addOperand(&type3);
  auto op4 = model->addOperand(&type4);
  // Phase 2, operations
  static uint8_t op2_init[] = {129, 131, 133, 135};
  model->setOperandValue(op2, op2_init, sizeof(uint8_t) * 4);
  static int32_t op3_init[] = {-4};
  model->setOperandValue(op3, op3_init, sizeof(int32_t) * 1);
  static int32_t pad_valid_init[] = {2};
  model->setOperandValue(pad_valid, pad_valid_init, sizeof(int32_t) * 1);
  static int32_t act_none_init[] = {0};
  model->setOperandValue(act_none, act_none_init, sizeof(int32_t) * 1);
  static int32_t stride1_init[] = {1};
  model->setOperandValue(stride1, stride1_init, sizeof(int32_t) * 1);
  static int32_t stride3_init[] = {3};
  model->setOperandValue(stride3, stride3_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_CONV_2D, {op1, op2, op3, pad_valid, stride3, stride1, act_none}, {op4});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op4});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
