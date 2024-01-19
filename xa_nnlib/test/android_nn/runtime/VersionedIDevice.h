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

#ifndef ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H
#define ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H

#include "HalInterfaces.h"

#include <android-base/macros.h>
#include <string>
#include <tuple>

namespace android {
namespace nn {

/**
 * This object is either a V1_1::IDevice or V1_0::IDevice object. This class
 * abstracts away version differences, allowing the remainder of the runtime to
 * always use the most up-to-date version of all HIDL types. As such, any
 * reference to a HIDL type in the rest of the runtime will--by default--be the
 * latest HIDL version.
 *
 * This class will attempt to call the latest version of each interface method
 * if possible. If the latest method is unavailable, the VersionedIDevice class
 * will attempt to upcast the type (e.g., V1_1::Model to V1_0::Model), and
 * invoke the latest interface method possible. If the VersionedIDevice class
 * fails to find a matching applicable function, it will return an error.
 */
class VersionedIDevice {
    DISALLOW_IMPLICIT_CONSTRUCTORS(VersionedIDevice);
public:
    /**
     * Constructor for the VersionedIDevice object.
     *
     * VersionedIDevice is constructed with the V1_0::IDevice object, which
     * represents a device that is at least v1.0 of the interface. The
     * constructor downcasts to the latest version of the IDevice interface, and
     * will default to using the latest version of all IDevice interface
     * methods automatically.
     *
     * @param device A device object that is least version 1.0 of the IDevice
     *               interface.
     */
    VersionedIDevice(sp<V1_0::IDevice> device);

    /**
     * Gets the capabilities of a driver.
     *
     * @return status Error status of the call, must be:
     *                - NONE if successful
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     * @return capabilities Capabilities of the driver.
     */
    std::pair<ErrorStatus, Capabilities> getCapabilities();

    /**
     * Gets the supported operations in a model.
     *
     * getSupportedSubgraph indicates which operations of a model are fully
     * supported by the vendor driver. If an operation may not be supported for
     * any reason, getSupportedOperations must return false for that operation.
     *
     * @param model A model whose operations--and their corresponding
     *              operands--are to be verified by the driver.
     * @return status Error status of the call, must be:
     *                - NONE if successful
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     *                - INVALID_ARGUMENT if provided model is invalid
     * @return supportedOperations A list of supported operations, where true
     *                             indicates the operation is supported and
     *                             false indicates the operation is not
     *                             supported. The index of "supported"
     *                             corresponds with the index of the operation
     *                             it is describing.
     */
    std::pair<ErrorStatus, hidl_vec<bool>> getSupportedOperations(const Model& model);

    /**
     * Creates a prepared model for execution.
     *
     * prepareModel is used to make any necessary transformations or alternative
     * representations to a model for execution, possiblly including
     * transformations on the constant data, optimization on the model's graph,
     * or compilation into the device's native binary format. The model itself
     * is not changed.
     *
     * The model is prepared asynchronously with respect to the caller. The
     * prepareModel function must verify the inputs to the prepareModel function
     * are correct. If there is an error, prepareModel must immediately invoke
     * the callback with the appropriate ErrorStatus value and nullptr for the
     * IPreparedModel, then return with the same ErrorStatus. If the inputs to
     * the prepareModel function are valid and there is no error, prepareModel
     * must launch an asynchronous task to prepare the model in the background,
     * and immediately return from prepareModel with ErrorStatus::NONE. If the
     * asynchronous task fails to launch, prepareModel must immediately invoke
     * the callback with ErrorStatus::GENERAL_FAILURE and nullptr for the
     * IPreparedModel, then return with ErrorStatus::GENERAL_FAILURE.
     *
     * When the asynchronous task has finished preparing the model, it must
     * immediately invoke the callback function provided as an input to
     * prepareModel. If the model was prepared successfully, the callback object
     * must be invoked with an error status of ErrorStatus::NONE and the
     * produced IPreparedModel object. If an error occurred preparing the model,
     * the callback object must be invoked with the appropriate ErrorStatus
     * value and nullptr for the IPreparedModel.
     *
     * The only information that may be unknown to the model at this stage is
     * the shape of the tensors, which may only be known at execution time. As
     * such, some driver services may return partially prepared models, where
     * the prepared model can only be finished when it is paired with a set of
     * inputs to the model. Note that the same prepared model object can be
     * used with different shapes of inputs on different (possibly concurrent)
     * executions.
     *
     * Multiple threads can call prepareModel on the same model concurrently.
     *
     * @param model The model to be prepared for execution.
     * @param callback A callback object used to return the error status of
     *                 preparing the model for execution and the prepared model
     *                 if successful, nullptr otherwise. The callback object's
     *                 notify function must be called exactly once, even if the
     *                 model could not be prepared.
     * @return status Error status of launching a task which prepares the model
     *                in the background; must be:
     *                - NONE if preparation task is successfully launched
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     *                - INVALID_ARGUMENT if one of the input arguments is
     *                  invalid
     */
    ErrorStatus prepareModel(const Model& model, ExecutionPreference preference,
                             const sp<IPreparedModelCallback>& callback);

    /**
     * Returns the current status of a driver.
     *
     * @return status Status of the driver, one of:
     *                - DeviceStatus::AVAILABLE
     *                - DeviceStatus::BUSY
     *                - DeviceStatus::OFFLINE
     *                - DeviceStatus::UNKNOWN
     */
    DeviceStatus getStatus();

    /**
     * Returns whether this handle to an IDevice object is valid or not.
     *
     * @return bool true if V1_0::IDevice (which could be V1_1::IDevice) is
     *              valid, false otherwise.
     */
    bool operator!=(nullptr_t);

    /**
     * Returns whether this handle to an IDevice object is valid or not.
     *
     * @return bool true if V1_0::IDevice (which could be V1_1::IDevice) is
     *              invalid, false otherwise.
     */
    bool operator==(nullptr_t);

private:
    /**
     * Both versions of IDevice are necessary because the driver could be v1.0,
     * v1.1, or a later version. These two pointers logically represent the same
     * object.
     *
     * The general strategy is: HIDL returns a V1_0 device object, which
     * (if not nullptr) could be v1.0, v1.1, or a greater version. The V1_0
     * object is then "dynamically cast" to a V1_1 object. If successful,
     * mDeviceV1_1 will point to the same object as mDeviceV1_0; otherwise,
     * mDeviceV1_1 will be nullptr.
     *
     * In general:
     * * If the device is truly v1.0, mDeviceV1_0 will point to a valid object
     *   and mDeviceV1_1 will be nullptr.
     * * If the device is truly v1.1 or later, both mDeviceV1_0 and mDeviceV1_1
     *   will point to the same valid object.
     *
     * Idiomatic usage: if mDeviceV1_1 is non-null, do V1_1 dispatch; otherwise,
     * do V1_0 dispatch.
     */
    sp<V1_0::IDevice> mDeviceV1_0;
    sp<V1_1::IDevice> mDeviceV1_1;
};

}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_VERSIONED_IDEVICE_H
