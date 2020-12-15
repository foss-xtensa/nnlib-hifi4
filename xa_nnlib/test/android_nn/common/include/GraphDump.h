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
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_COMMON_GRAPH_DUMP_H
#define ANDROID_ML_NN_COMMON_GRAPH_DUMP_H

#include <android/hardware/neuralnetworks/1.1/types.h>

#include <iostream>

namespace android {
namespace nn {

// Write a representation of the model in Graphviz (.dot) format to
// the specified stream.  (See http://www.graphviz.org.)
//
// Operations are represented by boxes, and operands by ellipses.  The
// number in a box/ellipse is the operation/operand index.  In the
// case of an operation, we include the operation type (such as ADD).
// In the case of an operand, we include the an abbreviated form of
// the operand code (such as FLOAT32) and dimensions; and, if the
// operand has lifetime CONSTANT_COPY, CONSTANT_REFERENCE, or
// NO_VALUE, an abbreviated form of that lifetime.
//
// A graph edge from an operand to an operation is marked with the
// input index (position in the operation's input vector).  A graph
// edge from an operation to an operand is marked with the output
// index (position in the operation's output vector).  The number is
// omitted from a lone input edge or a lone output edge.
//
// A model input or output (operand) is shown in "reverse colors" --
// white text on a black background.
//
void graphDump(const char* name, const ::android::hardware::neuralnetworks::V1_1::Model& model,
               std::ostream& outStream = std::cout);

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_COMMON_GRAPH_DUMP_H
