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
// Generated file (from: local_response_norm_float_1.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::FLOAT32, {});
  OperandType type1(Type::INT32, {});
  OperandType type0(Type::TENSOR_FLOAT32, {1, 1, 1, 6});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto radius = model->addOperand(&type1);
  auto bias = model->addOperand(&type2);
  auto alpha = model->addOperand(&type2);
  auto beta = model->addOperand(&type2);
  auto output = model->addOperand(&type0);
  // Phase 2, operations
  static int32_t radius_init[] = {20};
  model->setOperandValue(radius, radius_init, sizeof(int32_t) * 1);
  static float bias_init[] = {9.0f};
  model->setOperandValue(bias, bias_init, sizeof(float) * 1);
  static float alpha_init[] = {4.0f};
  model->setOperandValue(alpha, alpha_init, sizeof(float) * 1);
  static float beta_init[] = {0.5f};
  model->setOperandValue(beta, beta_init, sizeof(float) * 1);
  model->addOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, {input, radius, bias, alpha, beta}, {output});
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
