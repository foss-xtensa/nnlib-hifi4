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

#include "NeuralNetworksWrapper.h"

#include "xa_nnlib_standards.h"

#ifndef NNTEST_ONLY_PUBLIC_API
#include "Manager.h"
#include "Utils.h"
#endif

#ifndef HIFI_BUILD
#include <gtest/gtest.h>
#endif //HIFI_BUILD

using namespace android::nn::wrapper;

#ifndef HIFI_BUILD
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

#ifndef NNTEST_ONLY_PUBLIC_API
    android::nn::initVLogMask();
#endif
    // Test with the installed drivers.
    int n1 = RUN_ALL_TESTS();
#ifdef NNTEST_ONLY_PUBLIC_API
    // Can't use non-public functionality, because we're linking against
    // the shared library version of the runtime.
    return n1;
#else
    // Test with the CPU driver only.
    android::nn::DeviceManager::get()->setUseCpuOnly(true);
    int n2 = RUN_ALL_TESTS();
    return n1 | n2;
#endif // NNTEST_ONLY_PUBLIC_API
}
#else

#include "TestGenerated.cpp"
#include "./generated/all_generated_tests_hifi.cpp"

using namespace all_tests;

#define EXPECT_NEAR_FLOAT(g, t, range) {\
    if(std::abs(g-t) >  range) \
    { \
        printf("Mismatch: Expected = %f\tActual = %f\n",g ,t); \
        totalNumberOfErrors++; \
    } \
}\

#define EXPECT_EQ_INT(g, t) {\
    if(g != t) \
    { \
        printf("Mismatch: Expected = %d\tActual = %d\n",g ,t); \
        totalNumberOfErrors++; \
    } \
}\

#define EXPECT_EQ_FINAL(g, t) {\
    if(g != t) { \
        printf("Test Failed\n"); \
    } else { \
        printf("Test Passed\n"); \
    } \
}\

#define SET_OPERAND_BUFFERS(type, operand_list, method) { \
            for (auto& i : std::get<MixedTypedIndex<type>::index>(operand_list)) \
            { \
                int idx = i.first; \
                std::vector<type> &in = i.second; \
                size_t s = in.size() * sizeof(type); \
                void *p = static_cast<void *>(in.data()); \
                void* buffer = s == 0 ? nullptr : p; \
                execution.method(idx, buffer, s); \
            } \
}

template <typename ty, size_t tuple_index>
void filter_internal(const MixedTyped& golden, MixedTyped* filtered) {
    for (auto& i : std::get<MixedTypedIndex<ty>::index>(golden)) {
                auto& g = std::get<tuple_index>(*filtered);
                if (!is_ignored(i.first)) g[i.first] = i.second;
    }
}

inline MixedTyped filter(const MixedTyped& golden) {
    MixedTyped filtered;
    filter_internal<float, 0>(golden, &filtered);
    filter_internal<int32_t, 1>(golden, &filtered);
    filter_internal<uint8_t, 2>(golden, &filtered);
    return filtered;
}

void compare_float(MixedTyped& filteredGolden, MixedTyped& filteredTest, size_t &totalNumberOfErrors, float fpAtol, float fpRtol)
{
    for (auto& i : std::get<MixedTypedIndex<float>::index>(filteredGolden)) 
    {
        const auto& test_operands = std::get<0>(filteredTest);
        const auto& test_ty = test_operands.find(i.first);
        //ASSERT_NE(test_ty, test_operands.end());
        for (unsigned int j = 0; j < i.second.size(); j++) {
            float g = i.second[j];
            float t = test_ty->second[j];
            float fpRange = fpAtol + fpRtol * std::abs(g);
            if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                EXPECT_NEAR_FLOAT(g, t, fpRange);
            }
            if (std::abs(g - t) > fpRange) {
                totalNumberOfErrors++;
            }
        }
    }
}

void compare_int32(MixedTyped& filteredGolden, MixedTyped& filteredTest, size_t &totalNumberOfErrors)
{
    for (auto& i : std::get<MixedTypedIndex<int32_t>::index>(filteredGolden)) 
    {
        const auto& test_operands = std::get<1>(filteredTest);
        const auto& test_ty = test_operands.find(i.first);
        //ASSERT_NE(test_ty, test_operands.end());
        for (unsigned int j = 0; j < i.second.size(); j++) {
            int32_t g = i.second[j];
            int32_t t = test_ty->second[j];
            if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                EXPECT_EQ_INT(g, t);
            }
            if (g != t) {
                totalNumberOfErrors++;
            }
        }
    }
}

void compare_uint8(MixedTyped& filteredGolden, MixedTyped& filteredTest, size_t &totalNumberOfErrors)
{
    for (auto& i : std::get<MixedTypedIndex<uint8_t>::index>(filteredGolden)) 
    {
        const auto& test_operands = std::get<2>(filteredTest);
        const auto& test_ty = test_operands.find(i.first);
        //ASSERT_NE(test_ty, test_operands.end());
        for (unsigned int j = 0; j < i.second.size(); j++) {
            uint8_t g = i.second[j];
            uint8_t t = test_ty->second[j];
            if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                EXPECT_EQ_INT(g, t);
            }
            if (g != t) {
                totalNumberOfErrors++;
            }
        }
    }
}

static void show_usage()
{
    printf ("Usage xt-run <binary>\n");
    printf ("The ann testbench does not require any additional command line options\n");
}

int main(int argc, char** argv) 
{
    /* Library name version etc print */
    fprintf(stderr, "\n--------------------------------------------------------\n");
    fprintf(stderr, "HiFi NN Library Android NN API Testbench\n");
    fprintf(stderr, "%s version %s\n",
            xa_nnlib_get_lib_name_string(),
            xa_nnlib_get_lib_version_string());
    fprintf(stderr, "Cadence Design Systems, Inc. http://www.cadence.com\n");
    fprintf(stderr, "--------------------------------------------------------\n");
    fprintf(stderr, "\n");

    uint8_t help = 0;
    Model model;
    if(argc > 1)
    {
      int argidx;
      for(argidx=1; argidx<argc; argidx++)
      {
        if(strncmp((argv[argidx]), "-", 1) != 0)
        {
          printf("Invalid argument: %s\n",argv[argidx]);
	        show_usage();
          exit(1);
        }
        if(!strcmp(argv[argidx],"-h") || !strcmp(argv[argidx],"-help") || !strcmp(argv[argidx],"--help"))
        {
          help = 1;
        }
        else
        {
          printf("Invalid argument: %s\n", argv[argidx]);
	        show_usage();
          exit(1);
        }
      }
    }

    if (help == 1){
      show_usage();
	    return 0;
    }

    printf("Creating model\n");
    CreateModel(&model);
    model.finish();

    printf("getting examples \n");
    std::vector<MixedTypedExampleType>& examples = get_examples();


    int exampleNo = 0;
    printf("Compiling \n");
    Compilation compilation(&model);
    compilation.finish();

    // TODO: Adjust the error limit based on testing.
    // If in relaxed mode, set the absolute tolerance to be 5ULP of FP16.
    float fpAtol = !model.isRelaxed() ? 1e-5f : 5.0f * 0.0009765625f;
    // Set the relative tolerance to be 5ULP of the corresponding FP precision.
    float fpRtol = !model.isRelaxed() ? 5.0f * 1.1920928955078125e-7f : 5.0f * 0.0009765625f;

    for (auto& example : examples) {

        //SCOPED_TRACE(exampleNo);
        // TODO: We leave it as a copy here.
        // Should verify if the input gets modified by the test later.
        MixedTyped inputs = example.first;
        const MixedTyped& golden = example.second;

        printf("Executing \n");
        Execution execution(&compilation);

        // Set all inputs
#if 0
        for_all(inputs, [&execution](int idx, const void* p, size_t s) {
                const void* buffer = s == 0 ? nullptr : p;
                ASSERT_EQ(Result::NO_ERROR, execution.setInput(idx, buffer, s));
                });
#else
        SET_OPERAND_BUFFERS(float, inputs, setInput);
        SET_OPERAND_BUFFERS(int32_t, inputs, setInput);
        SET_OPERAND_BUFFERS(uint8_t, inputs, setInput);
#endif

        MixedTyped test;
        // Go through all typed outputs
        resize_accordingly(golden, test);

#ifndef HIFI_BUILD
        for_all(test, [&execution](int idx, void* p, size_t s) {
                void* buffer = s == 0 ? nullptr : p;
                ASSERT_EQ(Result::NO_ERROR, execution.setOutput(idx, buffer, s));
                });
#else
        SET_OPERAND_BUFFERS(float, test, setOutput);
        SET_OPERAND_BUFFERS(int32_t, test, setOutput);
        SET_OPERAND_BUFFERS(uint8_t, test, setOutput);
#endif //HIFI_BUILD

        Result r = execution.compute();

#ifndef HIFI_BUILD
        ASSERT_EQ(Result::NO_ERROR, r);
#else //HIFI_BUILD
        (void)r;
#endif

        // Filter out don't cares
#ifndef HIFI_BUILD
        MixedTyped filteredGolden = filter(golden, is_ignored);
        MixedTyped filteredTest = filter(test, is_ignored);
#else
        MixedTyped filteredGolden = filter(golden);
        MixedTyped filteredTest = filter(test);
#endif //HIFI_BUILD

        // We want "close-enough" results for float
#ifndef HIFI_BUILD
        compare(filteredGolden, filteredTest, fpAtol, fpRtol);
#else
        size_t totalNumberOfErrors = 0;
        compare_float(filteredGolden, filteredTest, totalNumberOfErrors, fpAtol, fpRtol);
        compare_int32(filteredGolden, filteredTest, totalNumberOfErrors);
        compare_uint8(filteredGolden, filteredTest, totalNumberOfErrors);
        EXPECT_EQ_FINAL(size_t{0}, totalNumberOfErrors);
#endif //HIFI_BUILD
        exampleNo++;
    }
    (void)exampleNo; /* To remove LLVM15 warning */
    /*
       execute(all_tests::CreateModel,
       all_tests::is_ignored,
       all_tests::get_examples());
     */
    return 0;
}
#endif //HIFI_BUILD
