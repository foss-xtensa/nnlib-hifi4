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
// Generated file (from: rnn_state_relaxed.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type5(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {16, 16});
  OperandType type1(Type::TENSOR_FLOAT32, {16, 8});
  OperandType type3(Type::TENSOR_FLOAT32, {16});
  OperandType type4(Type::TENSOR_FLOAT32, {2, 16});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 8});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto weights = model->addOperand(&type1);
  auto recurrent_weights = model->addOperand(&type2);
  auto bias = model->addOperand(&type3);
  auto hidden_state_in = model->addOperand(&type4);
  auto activation_param = model->addOperand(&type5);
  auto hidden_state_out = model->addOperand(&type4);
  auto output = model->addOperand(&type4);
  // Phase 2, operations
  static int32_t activation_param_init[] = {1};
  model->setOperandValue(activation_param, activation_param_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_RNN, {input, weights, recurrent_weights, bias, hidden_state_in, activation_param}, {hidden_state_out, output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input, weights, recurrent_weights, bias, hidden_state_in},
    {hidden_state_out, output});
  // Phase 4: set relaxed execution
  model->relaxComputationFloat32toFloat16(true);
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {0};
  return ignore.find(i) != ignore.end();
}
