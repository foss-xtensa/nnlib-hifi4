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
// Generated file (from: lsh_projection_weights_as_inputs.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type3(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {3});
  OperandType type0(Type::TENSOR_FLOAT32, {4, 2});
  OperandType type1(Type::TENSOR_INT32, {3, 2});
  OperandType type4(Type::TENSOR_INT32, {8});
  // Phase 1, operands
  auto hash = model->addOperand(&type0);
  auto lookup = model->addOperand(&type1);
  auto weight = model->addOperand(&type2);
  auto type_param = model->addOperand(&type3);
  auto output = model->addOperand(&type4);
  // Phase 2, operations
  static int32_t type_param_init[] = {2};
  model->setOperandValue(type_param, type_param_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_LSH_PROJECTION, {hash, lookup, weight, type_param}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {hash, lookup, weight},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
