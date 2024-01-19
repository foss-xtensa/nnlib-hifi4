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

#ifndef ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H
#define ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H

#include "ExecutionPlan.h"
#include "NeuralNetworks.h"

#include <memory>
#include <vector>

namespace android {
namespace nn {

class Device;
class ExecutionBuilder;
class ModelBuilder;

class CompilationBuilder {
public:
    friend class ExecutionBuilder;  // TODO remove this

    CompilationBuilder(const ModelBuilder* model);

    int setPreference(int32_t preference);

    int setPartitioning(uint32_t partitioning);

    int finish();

    int finish(const std::vector<std::shared_ptr<Device>>& devices);

    int createExecution(ExecutionBuilder** execution);

    const ExecutionPlan& forTest_getExecutionPlan() const { return mPlan; }

private:
    const ModelBuilder* mModel;

    ExecutionPlan mPlan;

    // Whether the application prefers to go fast or use low power for this execution.
    int32_t mPreference = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER;

    // See class DeviceManager.  When CompilationBuilder is
    // instantiated, we capture partitioning from DeviceManager; but
    // we can override this later.
    uint32_t mPartitioning;

    // Once the compilation has been finished, we should not allow further
    // modifications to the compilation.
    bool mFinished = false;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_COMPILATION_BUILDER_H
