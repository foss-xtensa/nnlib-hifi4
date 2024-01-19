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
// Generated file (from: conv_1_h3_w2_SAME.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type0(Type::INT32, {});
  OperandType type3(Type::TENSOR_FLOAT32, {1, 3, 2, 3});
  OperandType type2(Type::TENSOR_FLOAT32, {1, 8, 8, 1});
  OperandType type1(Type::TENSOR_FLOAT32, {1, 8, 8, 3});
  OperandType type4(Type::TENSOR_FLOAT32, {1});
  // Phase 1, operands
  auto b4 = model->addOperand(&type0);
  auto b5 = model->addOperand(&type0);
  auto b6 = model->addOperand(&type0);
  auto b7 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto op3 = model->addOperand(&type2);
  auto op0 = model->addOperand(&type3);
  auto op1 = model->addOperand(&type4);
  // Phase 2, operations
  static int32_t b4_init[] = {1};
  model->setOperandValue(b4, b4_init, sizeof(int32_t) * 1);
  static int32_t b5_init[] = {1};
  model->setOperandValue(b5, b5_init, sizeof(int32_t) * 1);
  static int32_t b6_init[] = {1};
  model->setOperandValue(b6, b6_init, sizeof(int32_t) * 1);
  static int32_t b7_init[] = {0};
  model->setOperandValue(b7, b7_init, sizeof(int32_t) * 1);
  static float op0_init[] = {-0.966213f, -0.467474f, -0.82203f, -0.579455f, 0.0278809f, -0.79946f, -0.684259f, 0.563238f, 0.37289f, 0.738216f, 0.386045f, -0.917775f, 0.184325f, -0.270568f, 0.82236f, 0.0973683f, -0.941308f, -0.144706f};
  model->setOperandValue(op0, op0_init, sizeof(float) * 18);
  static float op1_init[] = {0.0f};
  model->setOperandValue(op1, op1_init, sizeof(float) * 1);
  model->addOperation(ANEURALNETWORKS_CONV_2D, {op2, op0, op1, b4, b5, b6, b7}, {op3});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op2},
    {op3});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
