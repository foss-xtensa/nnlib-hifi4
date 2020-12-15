/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
// Generated file (from: lstm3_state2.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type9(Type::FLOAT32, {});
  OperandType type8(Type::INT32, {});
  OperandType type5(Type::TENSOR_FLOAT32, {0});
  OperandType type4(Type::TENSOR_FLOAT32, {16,20});
  OperandType type6(Type::TENSOR_FLOAT32, {2, 16});
  OperandType type7(Type::TENSOR_FLOAT32, {2, 20});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 5});
  OperandType type10(Type::TENSOR_FLOAT32, {2, 80});
  OperandType type2(Type::TENSOR_FLOAT32, {20, 16});
  OperandType type1(Type::TENSOR_FLOAT32, {20, 5});
  OperandType type3(Type::TENSOR_FLOAT32, {20});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto input_to_input_weights = model->addOperand(&type1);
  auto input_to_forget_weights = model->addOperand(&type1);
  auto input_to_cell_weights = model->addOperand(&type1);
  auto input_to_output_weights = model->addOperand(&type1);
  auto recurrent_to_intput_weights = model->addOperand(&type2);
  auto recurrent_to_forget_weights = model->addOperand(&type2);
  auto recurrent_to_cell_weights = model->addOperand(&type2);
  auto recurrent_to_output_weights = model->addOperand(&type2);
  auto cell_to_input_weights = model->addOperand(&type3);
  auto cell_to_forget_weights = model->addOperand(&type3);
  auto cell_to_output_weights = model->addOperand(&type3);
  auto input_gate_bias = model->addOperand(&type3);
  auto forget_gate_bias = model->addOperand(&type3);
  auto cell_gate_bias = model->addOperand(&type3);
  auto output_gate_bias = model->addOperand(&type3);
  auto projection_weights = model->addOperand(&type4);
  auto projection_bias = model->addOperand(&type5);
  auto output_state_in = model->addOperand(&type6);
  auto cell_state_in = model->addOperand(&type7);
  auto activation_param = model->addOperand(&type8);
  auto cell_clip_param = model->addOperand(&type9);
  auto proj_clip_param = model->addOperand(&type9);
  auto scratch_buffer = model->addOperand(&type10);
  auto output_state_out = model->addOperand(&type6);
  auto cell_state_out = model->addOperand(&type7);
  auto output = model->addOperand(&type6);
  // Phase 2, operations
  static int32_t activation_param_init[] = {4};
  model->setOperandValue(activation_param, activation_param_init, sizeof(int32_t) * 1);
  static float cell_clip_param_init[] = {0.0f};
  model->setOperandValue(cell_clip_param, cell_clip_param_init, sizeof(float) * 1);
  static float proj_clip_param_init[] = {0.0f};
  model->setOperandValue(proj_clip_param, proj_clip_param_init, sizeof(float) * 1);
  model->addOperation(ANEURALNETWORKS_LSTM, {input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_intput_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias, cell_gate_bias, output_gate_bias, projection_weights, projection_bias, output_state_in, cell_state_in, activation_param, cell_clip_param, proj_clip_param}, {scratch_buffer, output_state_out, cell_state_out, output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_intput_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias, cell_gate_bias, output_gate_bias, projection_weights, projection_bias, output_state_in, cell_state_in},
    {scratch_buffer, output_state_out, cell_state_out, output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {0};
  return ignore.find(i) != ignore.end();
}
