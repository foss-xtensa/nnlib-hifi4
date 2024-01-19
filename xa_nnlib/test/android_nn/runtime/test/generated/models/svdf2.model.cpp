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
// Generated file (from: svdf2.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type5(Type::INT32, {});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 3});
  OperandType type6(Type::TENSOR_FLOAT32, {2, 4});
  OperandType type4(Type::TENSOR_FLOAT32, {2, 80});
  OperandType type3(Type::TENSOR_FLOAT32, {4});
  OperandType type2(Type::TENSOR_FLOAT32, {8, 10});
  OperandType type1(Type::TENSOR_FLOAT32, {8, 3});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto weights_feature = model->addOperand(&type1);
  auto weights_time = model->addOperand(&type2);
  auto bias = model->addOperand(&type3);
  auto state_in = model->addOperand(&type4);
  auto rank_param = model->addOperand(&type5);
  auto activation_param = model->addOperand(&type5);
  auto state_out = model->addOperand(&type4);
  auto output = model->addOperand(&type6);
  // Phase 2, operations
  static int32_t rank_param_init[] = {2};
  model->setOperandValue(rank_param, rank_param_init, sizeof(int32_t) * 1);
  static int32_t activation_param_init[] = {0};
  model->setOperandValue(activation_param, activation_param_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_SVDF, {input, weights_feature, weights_time, bias, state_in, rank_param, activation_param}, {state_out, output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input, weights_feature, weights_time, bias, state_in},
    {state_out, output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {0};
  return ignore.find(i) != ignore.end();
}
