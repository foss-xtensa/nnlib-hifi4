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

#ifndef ANDROID_ML_NN_RUNTIME_MANAGER_H
#define ANDROID_ML_NN_RUNTIME_MANAGER_H

#include "HalInterfaces.h"
#include "Utils.h"
#ifndef HIFI_BUILD
#include "VersionedIDevice.h"
#include <android-base/macros.h>
#else
#include "VersionedInterfaces.h"
#endif //HIFI_BUILD

#include <map>
#include <unordered_set>
#include <vector>

namespace android {
namespace nn {

class ModelBuilder;

#ifndef HIFI_BUILD
class Device {
    DISALLOW_IMPLICIT_CONSTRUCTORS(Device);
public:
    Device(std::string name, const sp<V1_0::IDevice>& device);
    VersionedIDevice* getInterface() { return &mInterface; }
    const std::string& getName() const { return mName; }
    // Returns true if succesfully initialized.
    bool initialize();

    void getSupportedOperations(const Model& hidlModel, hidl_vec<bool>* supportedOperations);

    PerformanceInfo getFloat32Performance() const { return mFloat32Performance; }
    PerformanceInfo getQuantized8Performance() const { return mQuantized8Performance; }
    PerformanceInfo getRelaxedFloat32toFloat16Performance() const {
        return mRelaxedFloat32toFloat16Performance;
    }

private:
    std::string mName;
    VersionedIDevice mInterface;
    PerformanceInfo mFloat32Performance;
    PerformanceInfo mQuantized8Performance;
    PerformanceInfo mRelaxedFloat32toFloat16Performance;

#ifdef NN_DEBUGGABLE
    // For debugging: behavior of IDevice::getSupportedOperations for SampleDriver.
    // 0 - all operations reported by IDevice::getSupportedOperations() supported
    // 1 - some operations reported by IDevice::getSupportedOperations() supported
    uint32_t mSupported = 0;
#endif  // NN_DEBUGGABLE
};
#else //copied from later Android version
// A unified interface for actual driver devices as well as the CPU
class Device {
   public:
    virtual ~Device() {}

    // Get the handle of underlying VersionedIDevice, if any
#ifndef HIFI_BUILD
    virtual VersionedIDevice* getInterface() = 0;
#endif //HIFI_BUILD

    // Introspection methods returning device information
    virtual const char* getName() const = 0;
    virtual const char* getVersionString() const = 0;
    virtual int64_t getFeatureLevel() = 0;
    virtual void getSupportedOperations(const Model& hidlModel, hidl_vec<bool>* supported) = 0;
    virtual PerformanceInfo getFloat32Performance() const = 0;
    virtual PerformanceInfo getQuantized8Performance() const = 0;
    virtual PerformanceInfo getRelaxedFloat32toFloat16Performance() const = 0;

#ifndef HIFI_BUILD
    virtual int prepareModel(const Model& hidlModel, ExecutionPreference executionPreference,
                             std::shared_ptr<VersionedIPreparedModel>* preparedModel) = 0;
#else
    virtual int prepareModel(const Model& hidlModel, ExecutionPreference executionPreference,
                             void* preparedModel) = 0;
#endif //HIFI_BUILD
};
#endif //HIFI_BUILD

// Manages the NN HAL devices.  Only one instance of this class will exist.
// Use get() to retrieve it.
class DeviceManager {
   public:
    const std::vector<std::shared_ptr<Device>>& getDrivers() const {
        if (mSetCpuOnly || mDebugNNCpuOnly) {
#ifndef HIFI_BUILD
            return mNoDevices;
#else
            return mDevicesCpuOnly;
#endif //HIFI_BUILD
        }
        return mDevices;
    }

    // For testing only:
    void setUseCpuOnly(bool useCpuOnly) { mSetCpuOnly = useCpuOnly; }
#ifndef HIFI_BUILD
#else // copied from Android 2.0 preview
    bool getUseCpuOnly() const { return mSetCpuOnly; }
    void setSyncExecHal(bool val) {
        mSyncExecHal = val;
        mSyncExecHalSetter = true;
    }

    bool syncExecCpu() const { return mSyncExecCpu; }
    bool syncExecHal() const { return mSyncExecHal; }
    bool syncExecRuntime() const { return mSyncExecRuntime; }
#endif //HIFI_BUILD

    // How to handle graph partitioning?
    // 0 - Don't do graph partitioning.
    // 1 - Do graph partitioning; but fall back to non-partitioned
    //     execution if there is a partitioning failure.
    // 2 - Do graph partitioning, and rely on it; there is no fallback.
    enum {
        kPartitioningNo              = 0,
        kPartitioningWithFallback    = 1,
        kPartitioningWithoutFallback = 2
    };
    uint32_t getPartitioning() const { return mPartitioning; }
    static bool partitioningAllowsFallback(uint32_t partitioning) {
        return partitioning == kPartitioningWithFallback;
    }

    // Returns the singleton manager.
    static DeviceManager* get();

#ifndef HIFI_BUILD
#else
    // Returns the singleton Cpu device.
    static std::shared_ptr<Device> getCpuDevice();

    // These functions are solely intended for use by unit tests of
    // the introspection and control API.
    //
    // Register a test device.
#ifndef HIFI_BUILD
    void forTest_registerDevice(const char* name, const sp<V1_0::IDevice>& device) {
        registerDevice(name, device);
    }
    // Re-initialize the list of available devices.
    void forTest_reInitializeDeviceList() {
        mDevices.clear();
        findAvailableDevices();
    }
    // Make a test device
    static std::shared_ptr<Device> forTest_makeDriverDevice(const std::string& name,
                                                            const sp<V1_0::IDevice>& device);
#endif //HIFI_BUILD
#endif //HIFI_BUILD

private:
    // Builds the list of available drivers and queries their capabilities.
    DeviceManager();

#ifndef HIFI_BUILD
    // Adds a device for the manager to use.
    void registerDevice(const char* name, const sp<V1_0::IDevice>& device);
#endif //HIFI_BUILD

    void findAvailableDevices();

    // List of all the devices we discovered.
    std::vector<std::shared_ptr<Device>> mDevices;

#ifndef HIFI_BUILD
    // We leave this one always empty. To be used when mUseCpuOnly is true.
    std::vector<std::shared_ptr<Device>> mNoDevices;
#else
    // We set this one to have CpuDevice only. To be used when m*CpuOnly is true.
    std::vector<std::shared_ptr<Device>> mDevicesCpuOnly;
#endif //HIFI_BUILD

    // If either of these is true, we'll ignore the drivers that are
    // on the device and run everything on the CPU.
    bool mSetCpuOnly = false;      // set by setUseCpuOnly()
    bool mDebugNNCpuOnly = false;  // derived from system property debug.nn.cpuonly
#ifndef HIFI_BUILD
#else
    // synchronous execution
    bool mSyncExecCpu = false;
    bool mSyncExecHal = true;         // Call executeSynchronously() when available on device.
    bool mSyncExecHalSetter = false;  // Has mSyncExecHal been set by setSyncExecHal()?
                                      // If so, don't allow the setting to be overridden
                                      //     by system property debug.nn.syncexec-hal
    bool mSyncExecRuntime = false;
#endif //HIFI_BUILD

    static const uint32_t kPartitioningDefault = kPartitioningWithFallback;
    uint32_t mPartitioning = kPartitioningDefault;
};

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_RUNTIME_MANAGER_H
