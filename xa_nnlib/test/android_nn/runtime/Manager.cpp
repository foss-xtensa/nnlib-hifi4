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

#define LOG_TAG "Manager"

#include "Manager.h"
#include "HalInterfaces.h"
#include "Utils.h"

#ifndef HIFI_BUILD
#include <android/hidl/manager/1.0/IServiceManager.h>
#include <hidl/HidlTransportSupport.h>
#include <hidl/ServiceManagement.h>
#endif //HIFI_BUILD

#include <algorithm>
#include <functional>

namespace android {
namespace nn {

#ifndef HIFI_BUILD
Device::Device(std::string name, const sp<V1_0::IDevice>& device) :
      mName(std::move(name)), mInterface(device) {}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
// TODO: handle errors from initialize correctly
bool Device::initialize() {
#ifdef NN_DEBUGGABLE
    static const char samplePrefix[] = "sample";

    mSupported =
            (mName.substr(0, sizeof(samplePrefix) - 1)  == samplePrefix)
            ? getProp("debug.nn.sample.supported") : 0;
#endif  // NN_DEBUGGABLE

    ErrorStatus status = ErrorStatus::GENERAL_FAILURE;
    Capabilities capabilities;
    std::tie(status, capabilities) = mInterface.getCapabilities();

    if (status != ErrorStatus::NONE) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "IDevice::getCapabilities returned the error " << toString(status);
#endif //HIFI_BUILD
    } else {
#ifndef HIFI_BUILD
        VLOG(MANAGER) << "Capab " << capabilities.float32Performance.execTime;
        VLOG(MANAGER) << "Capab " << capabilities.quantized8Performance.execTime;
        VLOG(MANAGER) << "Capab " << capabilities.relaxedFloat32toFloat16Performance.execTime;
#endif //HIFI_BUILD
        mFloat32Performance = capabilities.float32Performance;
        mQuantized8Performance = capabilities.quantized8Performance;
        mRelaxedFloat32toFloat16Performance = capabilities.relaxedFloat32toFloat16Performance;
    }

    return status == ErrorStatus::NONE;
}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
void Device::getSupportedOperations(const Model& hidlModel,
                                    hidl_vec<bool>* outSupportedOperations) {
    // Query the driver for what it can do.
    ErrorStatus status = ErrorStatus::GENERAL_FAILURE;
    hidl_vec<bool> supportedOperations;
    std::tie(status, supportedOperations) = mInterface.getSupportedOperations(hidlModel);

    if (status != ErrorStatus::NONE) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "IDevice::getSupportedOperations returned the error " << toString(status);
#endif //HIFI_BUILD
        // Set the supported operation vectors to all false, so we won't use this driver.
        outSupportedOperations->resize(hidlModel.operations.size());
        std::fill(outSupportedOperations->begin(), outSupportedOperations->end(), false);
        return;
    }
    if (supportedOperations.size() != hidlModel.operations.size()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "IDevice::getSupportedOperations returned a vector of length "
                   << supportedOperations.size() << " when expecting "
                   << hidlModel.operations.size();
#endif //HIFI_BUILD
        // Set the supported operation vectors to all false, so we won't use this driver.
        outSupportedOperations->resize(hidlModel.operations.size());
        std::fill(outSupportedOperations->begin(), outSupportedOperations->end(), false);
        return;
    }

    *outSupportedOperations = supportedOperations;

#ifdef NN_DEBUGGABLE
    if (mSupported != 1) {
        return;
    }

    const uint32_t baseAccumulator = std::hash<std::string>{}(mName);
    for (size_t operationIndex = 0; operationIndex < outSupportedOperations->size();
         operationIndex++) {
        if (!(*outSupportedOperations)[operationIndex]) {
            continue;
        }

        uint32_t accumulator = baseAccumulator;
        const Operation &operation = hidlModel.operations[operationIndex];
        accumulator ^= static_cast<uint32_t>(operation.type);
        auto accumulateOperands = [&hidlModel, &accumulator](const hidl_vec<uint32_t>& operands) {
            for (uint32_t operandIndex : operands) {
                const Operand& operand = hidlModel.operands[operandIndex];
                accumulator ^= static_cast<uint32_t>(operand.type);
                accumulator ^= operand.dimensions.size();
                for (uint32_t dimension : operand.dimensions) {
                    accumulator ^= dimension;
                    if (operand.lifetime == OperandLifeTime::CONSTANT_COPY ||
                        operand.lifetime == OperandLifeTime::CONSTANT_REFERENCE) {
                        accumulator ^= 1;
                    }
                }
            }
        };
        accumulateOperands(operation.inputs);
        accumulateOperands(operation.outputs);
        if (accumulator & 1) {
            (*outSupportedOperations)[operationIndex] = false;
        }
    }
#endif  // NN_DEBUGGABLE
}
#endif //HIFI_BUILD

#ifndef HIFI_BUILD
#else
// A special abstracted device for the CPU. Only one instance of this class will exist.
// Use get() to retrieve it.
class CpuDevice : public Device {
    DISALLOW_COPY_AND_ASSIGN(CpuDevice);

   public:
    // Returns the singleton CPU fallback device.
    static std::shared_ptr<CpuDevice> get() {
        static std::shared_ptr<CpuDevice> instance(new CpuDevice);
        return instance;
    }

    const char* getName() const override { return kName.c_str(); }
    const char* getVersionString() const override { return kVersionString.c_str(); }
#ifndef HIFI_BUILD
    VersionedIDevice* getInterface() override { return nullptr; }
#endif //HIFI_BUILD
    int64_t getFeatureLevel() override { return kFeatureLevel; }
    void getSupportedOperations(const Model& hidlModel, hidl_vec<bool>* supported) override;
    PerformanceInfo getFloat32Performance() const override { return kPerformance; }
    PerformanceInfo getQuantized8Performance() const override { return kPerformance; }
    PerformanceInfo getRelaxedFloat32toFloat16Performance() const override { return kPerformance; }

#ifndef HIFI_BUILD
    int prepareModel(const Model& hidlModel, ExecutionPreference executionPreference,
                     std::shared_ptr<VersionedIPreparedModel>* preparedModel) override;
#else
    int prepareModel(const Model& hidlModel, ExecutionPreference executionPreference,
                     void* preparedModel) override;
#endif //HIFI_BUILD

   private:
    CpuDevice() = default;
    const int64_t kFeatureLevel = __ANDROID_API__;
    const std::string kName = "google-cpu";
    const std::string kVersionString = "2"; //build::GetBuildNumber(); //ppn
    // Since the performance is a ratio compared to the CPU performance,
    // by definition the performance of the CPU is 1.0.
    const PerformanceInfo kPerformance = {.execTime = 1.0f, .powerUsage = 1.0f};
};

void CpuDevice::getSupportedOperations(const Model& hidlModel, hidl_vec<bool>* supported) {
    const size_t count = hidlModel.operations.size();
    hidl_vec<bool> supportedOperations(count);
    for (size_t i = 0; i < count; i++) {
        // TODO(b/119870033): Decide whether and how post-P operations would be supported on CPU.
        // CPU fallback should support all the operations except for OEM_OPERATION
        if (hidlModel.operations[i].type == OperationType::OEM_OPERATION) {
            supportedOperations[i] = false;
        } else {
            supportedOperations[i] = true;
        }
    }
    *supported = std::move(supportedOperations);
}

int CpuDevice::prepareModel(const Model& hidlModel, ExecutionPreference executionPreference,
#ifndef HIFI_BUILD
                            std::shared_ptr<VersionedIPreparedModel>* preparedModel) {
#else
                            void* preparedModel) {
#endif //HIFI_BUILD
    //*preparedModel = nullptr; //ppn
    if (!validateModel(hidlModel) || !validateExecutionPreference(executionPreference)) {
        return ANEURALNETWORKS_OP_FAILED;
    }
    return ANEURALNETWORKS_NO_ERROR;
}
#endif //HIFI_BUILD

DeviceManager* DeviceManager::get() {
    static DeviceManager manager;
    return &manager;
}

#ifndef HIFI_BUILD
#else
std::shared_ptr<Device> DeviceManager::getCpuDevice() {
    return CpuDevice::get();
}
#endif //HIFI_BUILD


void DeviceManager::findAvailableDevices() {
#ifndef HIFI_BUILD
    using ::android::hidl::manager::V1_0::IServiceManager;
    VLOG(MANAGER) << "findAvailableDevices";

    sp<IServiceManager> manager = hardware::defaultServiceManager();
    if (manager == nullptr) {
        LOG(ERROR) << "Unable to open defaultServiceManager";
        return;
    }

    manager->listByInterface(V1_0::IDevice::descriptor, [this](const hidl_vec<hidl_string>& names) {
        for (const auto& name : names) {
            VLOG(MANAGER) << "Found interface " << name.c_str();
            sp<V1_0::IDevice> device = V1_0::IDevice::getService(name);
            if (device == nullptr) {
                LOG(ERROR) << "Got a null IDEVICE for " << name.c_str();
                continue;
            }
            registerDevice(name.c_str(), device);
        }
    });
#else
    // register CPU fallback device
    mDevices.push_back(CpuDevice::get());
    mDevicesCpuOnly.push_back(CpuDevice::get());
#endif //HIFI_BUILD
}

#ifndef HIFI_BUILD
void DeviceManager::registerDevice(const char* name, const sp<V1_0::IDevice>& device) {
    auto d = std::make_shared<Device>(name, device);
    if (d->initialize()) {
        mDevices.push_back(d);
    }
}
#endif //HIFI_BUILD

DeviceManager::DeviceManager() {
#ifndef HIFI_BUILD
    VLOG(MANAGER) << "DeviceManager::DeviceManager";
#endif //HIFI_BUILD
    findAvailableDevices();
#ifdef NN_DEBUGGABLE
    mPartitioning = getProp("debug.nn.partition", kPartitioningDefault);
    mDebugNNCpuOnly = (getProp("debug.nn.cpuonly") != 0);
#endif  // NN_DEBUGGABLE
}

}  // namespace nn
}  // namespace android
