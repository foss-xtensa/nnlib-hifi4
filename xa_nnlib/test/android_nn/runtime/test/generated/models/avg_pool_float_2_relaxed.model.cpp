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
// Generated file (from: avg_pool_float_2_relaxed.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {5, 11, 13, 3});
  OperandType type0(Type::TENSOR_FLOAT32, {5, 52, 60, 3});
  // Phase 1, operands
  auto i0 = model->addOperand(&type0);
  auto stride = model->addOperand(&type1);
  auto filter = model->addOperand(&type1);
  auto padding = model->addOperand(&type1);
  auto activation = model->addOperand(&type1);
  auto output = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t stride_init[] = {5};
  model->setOperandValue(stride, stride_init, sizeof(int32_t) * 1);
  static int32_t filter_init[] = {100};
  model->setOperandValue(filter, filter_init, sizeof(int32_t) * 1);
  static int32_t padding_init[] = {50};
  model->setOperandValue(padding, padding_init, sizeof(int32_t) * 1);
  static int32_t activation_init[] = {0};
  model->setOperandValue(activation, activation_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_AVERAGE_POOL_2D, {i0, padding, padding, padding, padding, stride, stride, filter, filter, activation}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {i0},
    {output});
  // Phase 4: set relaxed execution
  model->relaxComputationFloat32toFloat16(true);
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
