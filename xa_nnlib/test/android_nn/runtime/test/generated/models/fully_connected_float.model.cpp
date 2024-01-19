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
// Generated file (from: fully_connected_float.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type3(Type::INT32, {});
  OperandType type1(Type::TENSOR_FLOAT32, {1, 1});
  OperandType type2(Type::TENSOR_FLOAT32, {1});
  OperandType type0(Type::TENSOR_FLOAT32, {3, 1});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto b0 = model->addOperand(&type2);
  auto op3 = model->addOperand(&type0);
  auto act = model->addOperand(&type3);
  // Phase 2, operations
  static float op2_init[] = {2.0f};
  model->setOperandValue(op2, op2_init, sizeof(float) * 1);
  static float b0_init[] = {4.0f};
  model->setOperandValue(b0, b0_init, sizeof(float) * 1);
  static int32_t act_init[] = {0};
  model->setOperandValue(act, act_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_FULLY_CONNECTED, {op1, op2, b0, act}, {op3});
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
