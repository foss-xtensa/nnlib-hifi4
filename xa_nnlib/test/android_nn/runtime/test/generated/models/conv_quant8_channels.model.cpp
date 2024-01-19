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
// Generated file (from: conv_quant8_channels.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type3(Type::INT32, {});
  OperandType type2(Type::TENSOR_INT32, {3}, 0.25, 0);
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, {1, 1, 1, 3}, 0.5f, 0);
  OperandType type4(Type::TENSOR_QUANT8_ASYMM, {1, 1, 1, 3}, 1.0, 0);
  OperandType type1(Type::TENSOR_QUANT8_ASYMM, {3, 1, 1, 3}, 0.5f, 0);
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto op3 = model->addOperand(&type2);
  auto pad0 = model->addOperand(&type3);
  auto act = model->addOperand(&type3);
  auto stride = model->addOperand(&type3);
  auto op4 = model->addOperand(&type4);
  // Phase 2, operations
  static uint8_t op2_init[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  model->setOperandValue(op2, op2_init, sizeof(uint8_t) * 9);
  static int32_t op3_init[] = {0, 0, 0};
  model->setOperandValue(op3, op3_init, sizeof(int32_t) * 3);
  static int32_t pad0_init[] = {0};
  model->setOperandValue(pad0, pad0_init, sizeof(int32_t) * 1);
  static int32_t act_init[] = {0};
  model->setOperandValue(act, act_init, sizeof(int32_t) * 1);
  static int32_t stride_init[] = {1};
  model->setOperandValue(stride, stride_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_CONV_2D, {op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act}, {op4});
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
