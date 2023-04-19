/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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
//Generated by ./examples_hifi.py using ../examples/lstm2_state2.example.cpp

std::vector<MixedTypedExample>& get_examples() {

    Float32Operands inputs_float, outputs_float;
    Int32Operands inputs_int32, outputs_int32;
    Quant8Operands inputs_quant8, outputs_quant8;
    MixedTyped inputs, outputs;
    MixedTypedExample eg;
    static std::vector<MixedTypedExample> examples;

    inputs_float[0] = {1.0f, 1.0f};
    inputs_float[1] = {};
    inputs_float[2] = {-0.55291498f, -0.42866567f, 0.13056988f, -0.3633365f, -0.22755712f, 0.28253698f, 0.24407166f, 0.33826375f};
    inputs_float[3] = {-0.49770179f, -0.27711356f, -0.09624726f, 0.05100781f, 0.04717243f, 0.48944736f, -0.38535351f, -0.17212132f};
    inputs_float[4] = {0.10725588f, -0.02335852f, -0.55932593f, -0.09426838f, -0.44257352f, 0.54939759f, 0.01533556f, 0.42751634f};
    inputs_float[5] = {};
    inputs_float[6] = {-0.13832897f, -0.0515101f, -0.2359007f, -0.16661474f, -0.14340827f, 0.36986142f, 0.23414481f, 0.55899f, 0.10798943f, -0.41174671f, 0.17751795f, -0.34484994f, -0.35874045f, -0.11352962f, 0.27268326f, 0.54058349f};
    inputs_float[7] = {0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f, 0.42957711f, 0.01841056f, -0.32764608f, -0.33027974f, -0.10826075f, 0.20675004f, 0.19069612f, -0.03026325f, -0.54532051f, 0.33003211f, 0.44901288f, 0.21193194f};
    inputs_float[8] = {0.41613156f, 0.42610586f, -0.16495961f, -0.5663873f, 0.30579174f, -0.05115908f, -0.33941799f, 0.23364776f, 0.11178309f, 0.09481031f, -0.26424935f, 0.46261835f, 0.50248802f, 0.26114327f, -0.43736315f, 0.33149987f};
    inputs_float[9] = {};
    inputs_float[10] = {0.47485286f, -0.51955009f, -0.24458408f, 0.31544167f};
    inputs_float[11] = {-0.17135078f, 0.82760304f, 0.85573703f, -0.77109635f};
    inputs_float[12] = {};
    inputs_float[13] = {1.0f, 1.0f, 1.0f, 1.0f};
    inputs_float[14] = {0.0f, 0.0f, 0.0f, 0.0f};
    inputs_float[15] = {0.0f, 0.0f, 0.0f, 0.0f};
    inputs_float[16] = {};
    inputs_float[17] = {};
    inputs_float[18] = {-0.423122f, -0.0121822f, 0.24201f, -0.0812458f};
    inputs_float[19] = {-0.978419f, -0.139203f, 0.338163f, -0.0983904f};

    outputs_float[0] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    outputs_float[1] = {0, 0, 0, 0};
    outputs_float[2] = {0, 0, 0, 0};
    outputs_float[3] = {-0.358325f, -0.04621704f, 0.21641694f, -0.06471302f};
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
