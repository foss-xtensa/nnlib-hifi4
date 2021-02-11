/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
//Do not modify
//Generated by ./examples_hifi.py using ../examples/lstm_state.example.cpp

std::vector<MixedTypedExample>& get_examples() {

    Float32Operands inputs_float, outputs_float;
    Int32Operands inputs_int32, outputs_int32;
    Quant8Operands inputs_quant8, outputs_quant8;
    MixedTyped inputs, outputs;
    MixedTypedExample eg;
    static std::vector<MixedTypedExample> examples;

    inputs_float[0] = {3.0f, 4.0f};
    inputs_float[1] = {-0.45018822f, -0.02338299f, -0.0870589f, -0.34550029f, 0.04266912f, -0.15680569f, -0.34856534f, 0.43890524f};
    inputs_float[2] = {0.09701663f, 0.20334584f, -0.50592935f, -0.31343272f, -0.40032279f, 0.44781327f, 0.01387155f, -0.35593212f};
    inputs_float[3] = {-0.50013041f, 0.1370284f, 0.11810488f, 0.2013163f, -0.20583314f, 0.44344562f, 0.22077113f, -0.29909778f};
    inputs_float[4] = {-0.25065863f, -0.28290087f, 0.04613829f, 0.40525138f, 0.44272184f, 0.03897077f, -0.1556896f, 0.19487578f};
    inputs_float[5] = {-0.0063535f, -0.2042388f, 0.31454784f, -0.35746509f, 0.28902304f, 0.08183324f, -0.16555229f, 0.02286911f, -0.13566875f, 0.03034258f, 0.48091322f, -0.12528998f, 0.24077177f, -0.51332325f, -0.33502164f, 0.10629296f};
    inputs_float[6] = {-0.48684245f, -0.06655136f, 0.42224967f, 0.2112639f, 0.27654213f, 0.20864892f, -0.07646349f, 0.45877004f, 0.00141793f, -0.14609534f, 0.36447752f, 0.09196436f, 0.28053468f, 0.01560611f, -0.20127171f, -0.01140004f};
    inputs_float[7] = {-0.3407414f, 0.24443203f, -0.2078532f, 0.26320225f, 0.05695659f, -0.00123841f, -0.4744786f, -0.35869038f, -0.06418842f, -0.13502428f, -0.501764f, 0.22830659f, -0.46367589f, 0.26016325f, -0.03894562f, -0.16368064f};
    inputs_float[8] = {0.43385774f, -0.17194885f, 0.2718237f, 0.09215671f, 0.24107647f, -0.39835793f, 0.18212086f, 0.01301402f, 0.48572797f, -0.50656658f, 0.20047462f, -0.20607421f, -0.51818722f, -0.15390486f, 0.0468148f, 0.39922136f};
    inputs_float[9] = {};
    inputs_float[10] = {};
    inputs_float[11] = {};
    inputs_float[12] = {0.0f, 0.0f, 0.0f, 0.0f};
    inputs_float[13] = {1.0f, 1.0f, 1.0f, 1.0f};
    inputs_float[14] = {0.0f, 0.0f, 0.0f, 0.0f};
    inputs_float[15] = {0.0f, 0.0f, 0.0f, 0.0f};
    inputs_float[16] = {};
    inputs_float[17] = {};
    inputs_float[18] = {-0.0297319f, 0.122947f, 0.208851f, -0.153588f};
    inputs_float[19] = {-0.145439f, 0.157475f, 0.293663f, -0.277353f};

    outputs_float[0] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    outputs_float[1] = {-0.0371611f, 0.125073f, 0.411934f, -0.208605f};
    outputs_float[2] = {-0.287121f, 0.148115f, 0.556837f, -0.388276f};
    outputs_float[3] = {-0.03716109f, 0.12507336f, 0.41193449f, -0.20860538f};
    inputs = std::make_tuple(inputs_float, inputs_int32, inputs_quant8);
    outputs = std::make_tuple(outputs_float, outputs_int32, outputs_quant8);

    eg = std::make_pair(inputs, outputs);

    examples.push_back(eg);
    inputs_float.clear();
    outputs_float.clear();
    inputs_int32.clear();
    outputs_int32.clear();
    inputs_quant8.clear();
    outputs_quant8.clear();

    return examples;
};
