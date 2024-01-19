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
 * Copyright (C) 2018 The Android Open Source Project
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

#include "VersionedIDevice.h"

#include "Utils.h"

#include <android-base/logging.h>

namespace android {
namespace nn {

// HIDL guarantees all V1_1 interfaces inherit from their corresponding V1_0 interfaces.
VersionedIDevice::VersionedIDevice(sp<V1_0::IDevice> device) :
        mDeviceV1_0(device),
        mDeviceV1_1(V1_1::IDevice::castFrom(mDeviceV1_0).withDefault(nullptr)) {}

std::pair<ErrorStatus, Capabilities> VersionedIDevice::getCapabilities() {
    std::pair<ErrorStatus, Capabilities> result;

    if (mDeviceV1_1 != nullptr) {
        Return<void> ret = mDeviceV1_1->getCapabilities_1_1(
            [&result](ErrorStatus error, const Capabilities& capabilities) {
                result = std::make_pair(error, capabilities);
            });
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "getCapabilities_1_1 failure: " << ret.description();
#endif //HIFI_BUILD
            return {ErrorStatus::GENERAL_FAILURE, {}};
        }
    } else if (mDeviceV1_0 != nullptr) {
        Return<void> ret = mDeviceV1_0->getCapabilities(
            [&result](ErrorStatus error, const V1_0::Capabilities& capabilities) {
                result = std::make_pair(error, convertToV1_1(capabilities));
            });
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "getCapabilities failure: " << ret.description();
#endif //HIFI_BUILD
            return {ErrorStatus::GENERAL_FAILURE, {}};
        }
    } else {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Device not available!";
#endif //HIFI_BUILD
        return {ErrorStatus::DEVICE_UNAVAILABLE, {}};
    }

    return result;
}

std::pair<ErrorStatus, hidl_vec<bool>> VersionedIDevice::getSupportedOperations(
        const Model& model) {
    std::pair<ErrorStatus, hidl_vec<bool>> result;

    if (mDeviceV1_1 != nullptr) {
        Return<void> ret = mDeviceV1_1->getSupportedOperations_1_1(
            model, [&result](ErrorStatus error, const hidl_vec<bool>& supported) {
                result = std::make_pair(error, supported);
            });
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "getSupportedOperations_1_1 failure: " << ret.description();
#endif //HIFI_BUILD
            return {ErrorStatus::GENERAL_FAILURE, {}};
        }
    } else if (mDeviceV1_0 != nullptr && compliantWithV1_0(model)) {
        Return<void> ret = mDeviceV1_0->getSupportedOperations(
            convertToV1_0(model), [&result](ErrorStatus error, const hidl_vec<bool>& supported) {
                result = std::make_pair(error, supported);
            });
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "getSupportedOperations failure: " << ret.description();
#endif //HIFI_BUILD
            return {ErrorStatus::GENERAL_FAILURE, {}};
        }
    } else {
        // TODO: partition the model such that v1.1 ops are not passed to v1.0
        // device
#ifndef HIFI_BUILD
        LOG(ERROR) << "Could not handle getSupportedOperations!";
#endif //HIFI_BUILD
        return {ErrorStatus::GENERAL_FAILURE, {}};
    }

    return result;
}

ErrorStatus VersionedIDevice::prepareModel(const Model& model, ExecutionPreference preference,
                                           const sp<IPreparedModelCallback>& callback) {
    if (mDeviceV1_1 != nullptr) {
        Return<ErrorStatus> ret = mDeviceV1_1->prepareModel_1_1(model, preference, callback);
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "prepareModel_1_1 failure: " << ret.description();
#endif //HIFI_BUILD
            return ErrorStatus::GENERAL_FAILURE;
        }
        return static_cast<ErrorStatus>(ret);
    } else if (mDeviceV1_0 != nullptr && compliantWithV1_0(model)) {
        Return<ErrorStatus> ret = mDeviceV1_0->prepareModel(convertToV1_0(model), callback);
        if (!ret.isOk()) {
#ifndef HIFI_BUILD
            LOG(ERROR) << "prepareModel failure: " << ret.description();
#endif //HIFI_BUILD
            return ErrorStatus::GENERAL_FAILURE;
        }
        return static_cast<ErrorStatus>(ret);
    } else {
        // TODO: partition the model such that v1.1 ops are not passed to v1.0
        // device
#ifndef HIFI_BUILD
        LOG(ERROR) << "Could not handle prepareModel!";
#endif //HIFI_BUILD
        return ErrorStatus::GENERAL_FAILURE;
    }
}

DeviceStatus VersionedIDevice::getStatus() {
    if (mDeviceV1_0 == nullptr) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "Device not available!";
#endif //HIFI_BUILD
        return DeviceStatus::UNKNOWN;
    }

    Return<DeviceStatus> ret = mDeviceV1_0->getStatus();

    if (!ret.isOk()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "getStatus failure: " << ret.description();
#endif //HIFI_BUILD
        return DeviceStatus::UNKNOWN;
    }
    return static_cast<DeviceStatus>(ret);
}

bool VersionedIDevice::operator==(nullptr_t) {
    return mDeviceV1_0 == nullptr;
}

bool VersionedIDevice::operator!=(nullptr_t) {
    return mDeviceV1_0 != nullptr;
}

}  // namespace nn
}  // namespace android
