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

// Top level driver for models and examples generated by test_generator.py

#include "Bridge.h"
#include "NeuralNetworksWrapper.h"
#include "TestHarness.h"

#ifndef HIFI_BUILD
#include <gtest/gtest.h>
#endif //HIFI_BUILD
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

// Uncomment the following line to generate DOT graphs.
//
// #define GRAPH GRAPH

namespace generated_tests {
using namespace android::nn::wrapper;
using namespace test_helper;

#ifndef HIFI_BUILD
void graphDump([[maybe_unused]] const char* name, [[maybe_unused]] const Model& model) {
#ifdef GRAPH
    ::android::nn::bridge_tests::graphDump(
         name,
         reinterpret_cast<const ::android::nn::ModelBuilder*>(model.getHandle()));
#endif
}

template <typename T>
static void print(std::ostream& os, const MixedTyped& test) {
    // dump T-typed inputs
    for_each<T>(test, [&os](int idx, const std::vector<T>& f) {
        os << "    aliased_output" << idx << ": [";
        for (size_t i = 0; i < f.size(); ++i) {
            os << (i == 0 ? "" : ", ") << +f[i];
        }
        os << "],\n";
    });
}

static void printAll(std::ostream& os, const MixedTyped& test) {
    print<float>(os, test);
    print<int32_t>(os, test);
    print<uint8_t>(os, test);
}
#endif //HIFI_BUILD

// Test driver for those generated from ml/nn/runtime/test/spec
#if 0 //ppn
static void execute(std::function<void(Model*)> createModel,
             std::function<bool(int)> isIgnored,
             std::vector<MixedTypedExampleType>& examples,
             std::string dumpFile = "") {
    Model model;
    createModel(&model);
    model.finish();

#ifndef HIFI_BUILD
    graphDump("", model);
    bool dumpToFile = !dumpFile.empty();

    std::ofstream s;
    if (dumpToFile) {
        s.open(dumpFile, std::ofstream::trunc);
        ASSERT_TRUE(s.is_open());
    }
#endif //HIFI_BUILD

    int exampleNo = 0;
    Compilation compilation(&model);
    compilation.finish();

    // TODO: Adjust the error limit based on testing.
    // If in relaxed mode, set the absolute tolerance to be 5ULP of FP16.
    float fpAtol = !model.isRelaxed() ? 1e-5f : 5.0f * 0.0009765625f;
    // Set the relative tolerance to be 5ULP of the corresponding FP precision.
    float fpRtol = !model.isRelaxed() ? 5.0f * 1.1920928955078125e-7f : 5.0f * 0.0009765625f;
    for (auto& example : examples) {
        SCOPED_TRACE(exampleNo);
        // TODO: We leave it as a copy here.
        // Should verify if the input gets modified by the test later.
        MixedTyped inputs = example.first;
        const MixedTyped& golden = example.second;

        Execution execution(&compilation);

        // Set all inputs
        for_all(inputs, [&execution](int idx, const void* p, size_t s) {
            const void* buffer = s == 0 ? nullptr : p;
            ASSERT_EQ(Result::NO_ERROR, execution.setInput(idx, buffer, s));
        });

        MixedTyped test;
        // Go through all typed outputs
        resize_accordingly(golden, test);
        for_all(test, [&execution](int idx, void* p, size_t s) {
            void* buffer = s == 0 ? nullptr : p;
            ASSERT_EQ(Result::NO_ERROR, execution.setOutput(idx, buffer, s));
        });

        Result r = execution.compute();
        ASSERT_EQ(Result::NO_ERROR, r);

#ifndef HIFI_BUILD
        // Dump all outputs for the slicing tool
        if (dumpToFile) {
            s << "output" << exampleNo << " = {\n";
            printAll(s, test);
            // all outputs are done
            s << "}\n";
        }
#endif //HIFI_BUILD

        // Filter out don't cares
        MixedTyped filteredGolden = filter(golden, isIgnored);
        MixedTyped filteredTest = filter(test, isIgnored);
        // We want "close-enough" results for float

        compare(filteredGolden, filteredTest, fpAtol, fpRtol);
        exampleNo++;
    }
}
#endif //ppn

};  // namespace generated_tests

using namespace android::nn::wrapper;

// Mixed-typed examples
typedef test_helper::MixedTypedExampleType MixedTypedExample;

#ifndef HIFI_BUILD
class GeneratedTests : public ::testing::Test {
protected:
    virtual void SetUp() {}
};
#endif //HIFI_BUILD

// Testcases generated from runtime/test/specs/*.mod.py
using namespace test_helper;
using namespace generated_tests;
#ifndef HIFI_BUILD
#include "generated/all_generated_tests.cpp"
#endif //HIFI_BUILD
// End of testcases generated from runtime/test/specs/*.mod.py
