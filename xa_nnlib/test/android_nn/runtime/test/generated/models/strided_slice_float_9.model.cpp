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
// Generated file (from: strided_slice_float_9.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::INT32, {});
  OperandType type3(Type::TENSOR_FLOAT32, {2, 2});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 3});
  OperandType type1(Type::TENSOR_INT32, {2});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto begins = model->addOperand(&type1);
  auto ends = model->addOperand(&type1);
  auto strides = model->addOperand(&type1);
  auto beginMask = model->addOperand(&type2);
  auto endMask = model->addOperand(&type2);
  auto shrinkAxisMask = model->addOperand(&type2);
  auto output = model->addOperand(&type3);
  // Phase 2, operations
  static int32_t begins_init[] = {1, 0};
  model->setOperandValue(begins, begins_init, sizeof(int32_t) * 2);
  static int32_t ends_init[] = {2, 2};
  model->setOperandValue(ends, ends_init, sizeof(int32_t) * 2);
  static int32_t strides_init[] = {1, 1};
  model->setOperandValue(strides, strides_init, sizeof(int32_t) * 2);
  static int32_t beginMask_init[] = {1};
  model->setOperandValue(beginMask, beginMask_init, sizeof(int32_t) * 1);
  static int32_t endMask_init[] = {0};
  model->setOperandValue(endMask, endMask_init, sizeof(int32_t) * 1);
  static int32_t shrinkAxisMask_init[] = {0};
  model->setOperandValue(shrinkAxisMask, shrinkAxisMask_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_STRIDED_SLICE, {input, begins, ends, strides, beginMask, endMask, shrinkAxisMask}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
