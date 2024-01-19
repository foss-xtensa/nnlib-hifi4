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
// Generated file (from: resize_bilinear_2_relaxed.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::INT32, {});
  OperandType type0(Type::TENSOR_FLOAT32, {1, 2, 2, 2});
  OperandType type1(Type::TENSOR_FLOAT32, {1, 3, 3, 2});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto height = model->addOperand(&type2);
  auto width = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t height_init[] = {3};
  model->setOperandValue(height, height_init, sizeof(int32_t) * 1);
  static int32_t width_init[] = {3};
  model->setOperandValue(width, width_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_RESIZE_BILINEAR, {op1, height, width}, {op2});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op2});
  // Phase 4: set relaxed execution
  model->relaxComputationFloat32toFloat16(true);
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
