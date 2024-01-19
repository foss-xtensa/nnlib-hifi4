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
// Generated file (from: max_pool_quant8_4.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type2(Type::TENSOR_QUANT8_ASYMM, {1, 1, 2, 1}, 0.0625f, 0);
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, {1, 2, 4, 1}, 0.0625f, 0);
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto cons2 = model->addOperand(&type1);
  auto pad_same = model->addOperand(&type1);
  auto act_none = model->addOperand(&type1);
  auto op3 = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t cons2_init[] = {2};
  model->setOperandValue(cons2, cons2_init, sizeof(int32_t) * 1);
  static int32_t pad_same_init[] = {1};
  model->setOperandValue(pad_same, pad_same_init, sizeof(int32_t) * 1);
  static int32_t act_none_init[] = {0};
  model->setOperandValue(act_none, act_none_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_MAX_POOL_2D, {op1, pad_same, cons2, cons2, cons2, cons2, act_none}, {op3});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op3});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
