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
// Generated file (from: max_pool_float_3.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {5, 2, 3, 3});
  OperandType type0(Type::TENSOR_FLOAT32, {5, 50, 70, 3});
  // Phase 1, operands
  auto i0 = model->addOperand(&type0);
  auto stride = model->addOperand(&type1);
  auto filter = model->addOperand(&type1);
  auto padding = model->addOperand(&type1);
  auto relu6_activation = model->addOperand(&type1);
  auto output = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t stride_init[] = {20};
  model->setOperandValue(stride, stride_init, sizeof(int32_t) * 1);
  static int32_t filter_init[] = {20};
  model->setOperandValue(filter, filter_init, sizeof(int32_t) * 1);
  static int32_t padding_init[] = {0};
  model->setOperandValue(padding, padding_init, sizeof(int32_t) * 1);
  static int32_t relu6_activation_init[] = {3};
  model->setOperandValue(relu6_activation, relu6_activation_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_MAX_POOL_2D, {i0, padding, padding, padding, padding, stride, stride, filter, filter, relu6_activation}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {i0},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
