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
// Generated file (from: space_to_depth_float_3_relaxed.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {1, 2, 2, 8});
  OperandType type0(Type::TENSOR_FLOAT32, {1, 4, 4, 2});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto block_size = model->addOperand(&type1);
  auto output = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t block_size_init[] = {2};
  model->setOperandValue(block_size, block_size_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_SPACE_TO_DEPTH, {input, block_size}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input},
    {output});
  // Phase 4: set relaxed execution
  model->relaxComputationFloat32toFloat16(true);
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
