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
// Generated file (from: conv_float_2_relaxed.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type3(Type::INT32, {});
  OperandType type1(Type::TENSOR_FLOAT32, {1, 3, 3, 1});
  OperandType type0(Type::TENSOR_FLOAT32, {1, 3, 4, 1});
  OperandType type2(Type::TENSOR_FLOAT32, {1});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto op3 = model->addOperand(&type2);
  auto pad_same = model->addOperand(&type3);
  auto act_relu = model->addOperand(&type3);
  auto stride = model->addOperand(&type3);
  auto op4 = model->addOperand(&type0);
  // Phase 2, operations
  static float op2_init[] = {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f};
  model->setOperandValue(op2, op2_init, sizeof(float) * 9);
  static float op3_init[] = {-200.0f};
  model->setOperandValue(op3, op3_init, sizeof(float) * 1);
  static int32_t pad_same_init[] = {1};
  model->setOperandValue(pad_same, pad_same_init, sizeof(int32_t) * 1);
  static int32_t act_relu_init[] = {1};
  model->setOperandValue(act_relu, act_relu_init, sizeof(int32_t) * 1);
  static int32_t stride_init[] = {1};
  model->setOperandValue(stride, stride_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_CONV_2D, {op1, op2, op3, pad_same, stride, stride, act_relu}, {op4});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op4});
  // Phase 4: set relaxed execution
  model->relaxComputationFloat32toFloat16(true);
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
