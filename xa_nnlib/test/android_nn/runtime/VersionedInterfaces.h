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

#ifndef ANDROID_ML_NN_RUNTIME_VERSIONED_INTERFACES_H
#define ANDROID_ML_NN_RUNTIME_VERSIONED_INTERFACES_H

#include "HalInterfaces.h"

#include <android-base/macros.h>
#include <memory>
#include <string>
#include <tuple>

#ifndef HIFI_BUILD
namespace android {
namespace nn {

/**
 * Each class (VersionedIDevice, VersionedIPreparedModel) wraps a HIDL interface
 * of any version to abstract away version differences. It allows the remainder
 * of the runtime to always use the most up-to-date version of all HIDL types.
 * As such, any reference to a HIDL type in the rest of the runtime
 * will--by default--be the latest HIDL version.
 *
 * Each class will attempt to call the latest version of each interface method
 * if possible. If the latest method is unavailable, the versioned class
 * will attempt to upcast the type (e.g., V1_1::Model to V1_0::Model), and
 * invoke the latest interface method possible. If the versioned class
 * fails to find a matching applicable function, it will return an error.
 */

/** This class wraps an IDevice object of any version. */
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
     * Returns the feature level of a driver.
     *
     * @return featureLevel The API level of the most advanced feature this driver implements.
     *                      For example, if the driver implements the features introduced in
     *                      Android P, the value would be 28.
     */
    int64_t getFeatureLevel();

    /**
     * Get the version string of the driver implementation.
     *
     * The version string must be a unique token among the set of version strings of
     * drivers of a specific device. The token identifies the device driver's
     * implementation. The token must not be confused with the feature level which is solely
     * defined by the interface version. This API is opaque to the Android framework, but the
     * Android framework may use the information for debugging or to pass on to NNAPI applications.
     *
     * Application developers sometimes have specific requirements to ensure good user experiences,
     * and they need more information to make intelligent decisions when the Android framework
     * cannot. For example, combined with the device name and other information, the token can help
     * NNAPI applications filter devices based on their needs:
     *     - An application demands a certain level of performance, but a specific version of
     *       the driver cannot meet that requirement because of a performance regression.
     *       The application can blacklist the driver based on the version provided.
     *     - An application has a minimum precision requirement, but certain versions of
     *       the driver cannot meet that requirement because of bugs or certain optimizations.
     *       The application can filter out versions of these drivers.
     *
     * @return status Error status returned from querying the version string. Must be:
     *     - NONE if the query was successful
     *     - DEVICE_UNAVAILABLE if driver is offline or busy
     *     - GENERAL_FAILURE if the query resulted in an
     *       unspecified error
     * @return version The version string of the device implementation.
     *     Must have nonzero length if the query is successful, and must be an empty string if not.
     */
    std::pair<ErrorStatus, hidl_string> getVersionString();

    /**
     * Returns whether this handle to an IDevice object is valid or not.
     *
     * @return bool true if V1_0::IDevice (which could be V1_1::IDevice) is
     *              valid, false otherwise.
     */
    bool operator!=(nullptr_t) const;

    /**
     * Returns whether this handle to an IDevice object is valid or not.
     *
     * @return bool true if V1_0::IDevice (which could be V1_1::IDevice) is
     *              invalid, false otherwise.
     */
    bool operator==(nullptr_t) const;

   private:
    /**
     * All versions of IDevice are necessary because the driver could be v1.0,
     * v1.1, or a later version. All these pointers logically represent the same
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
    sp<V1_1::IDevice> mDeviceV1_2;
    //sp<V1_2::IDevice> mDeviceV1_2;
};

/** This class wraps an IPreparedModel object of any version. */
class VersionedIPreparedModel {
    DISALLOW_IMPLICIT_CONSTRUCTORS(VersionedIPreparedModel);

   public:
    /**
     * Constructor for the VersionedIPreparedModel object.
     *
     * VersionedIPreparedModel is constructed with the V1_0::IPreparedModel object, which
     * represents a device that is at least v1.0 of the interface. The constructor downcasts
     * to the latest version of the IPreparedModel interface, and will default to using the
     * latest version of all IPreparedModel interface methods automatically.
     *
     * @param preparedModel A prepared model object that is least version 1.0 of the
     *                      IPreparedModel interface.
     */
    VersionedIPreparedModel(sp<V1_0::IPreparedModel> preparedModel)
        : mPreparedModelV1_0(preparedModel),
          mPreparedModelV1_2(
                  nullptr) {}
                  //V1_2::IPreparedModel::castFrom(mPreparedModelV1_0).withDefault(nullptr)) {}

    /**
     * Launches an asynchronous execution on a prepared model.
     *
     * The execution is performed asynchronously with respect to the caller.
     * execute must verify the inputs to the function are correct. If there is
     * an error, execute must immediately invoke the callback with the
     * appropriate ErrorStatus value, then return with the same ErrorStatus. If
     * the inputs to the function are valid and there is no error, execute must
     * launch an asynchronous task to perform the execution in the background,
     * and immediately return with ErrorStatus::NONE. If the asynchronous task
     * fails to launch, execute must immediately invoke the callback with
     * ErrorStatus::GENERAL_FAILURE, then return with
     * ErrorStatus::GENERAL_FAILURE.
     *
     * When the asynchronous task has finished its execution, it must
     * immediately invoke the callback object provided as an input to the
     * execute function. This callback must be provided with the ErrorStatus of
     * the execution.
     *
     * If the prepared model was prepared from a model wherein all
     * tensor operands have fully specified dimensions, and the inputs
     * to the function are valid, then the execution should launch
     * and complete successfully (ErrorStatus::NONE). There must be
     * no failure unless the device itself is in a bad state.
     *
     * Multiple threads can call the execute and ExecuteSynchronously functions
     * on the same VersionedIPreparedModel object concurrently with different
     * requests.
     *
     * @param request The input and output information on which the prepared
     *                model is to be executed.
     * @param callback A callback object used to return the error status of
     *                 the execution. The callback object's notify function must
     *                 be called exactly once, even if the execution was
     *                 unsuccessful.
     * @return status Error status of the call, must be:
     *                - NONE if task is successfully launched
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     *                - OUTPUT_INSUFFICIENT_SIZE if provided output buffer is
     *                  not large enough to store the resultant values
     *                - INVALID_ARGUMENT if one of the input arguments is
     *                  invalid
     */
    ErrorStatus execute(const Request& request, const sp<IExecutionCallback>& callback);

    /**
     * Performs a synchronous execution on a prepared model.
     *
     * The execution is performed synchronously with respect to the caller.
     * executeSynchronously must verify the inputs to the function are
     * correct. If there is an error, executeSynchronously must immediately
     * return with the appropriate ErrorStatus value. If the inputs to the
     * function are valid and there is no error, executeSynchronously must
     * perform the execution, and must not return until the execution is
     * complete.
     *
     * If the prepared model was prepared from a model wherein all tensor
     * operands have fully specified dimensions, and the inputs to the function
     * are valid, then the execution should complete successfully
     * (ErrorStatus::NONE). There must be no failure unless the device itself is
     * in a bad state.
     *
     * Any number of calls to the execute and executeSynchronously
     * functions, in any combination, may be made concurrently, even on the same
     * VersionedIPreparedModel object.
     *
     * @param request The input and output information on which the prepared
     *                model is to be executed.
     * @return status Error status of the execution, must be:
     *                - NONE if execution is performed successfully
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     *                - OUTPUT_INSUFFICIENT_SIZE if provided output buffer is
     *                  not large enough to store the resultant values
     *                - INVALID_ARGUMENT if one of the input arguments is
     *                  invalid
     */
    ErrorStatus executeSynchronously(const Request& request);

    /**
     * Returns whether this handle to an IPreparedModel object is valid or not.
     *
     * @return bool true if V1_0::IPreparedModel (which could be V1_2::IPreparedModel) is
     *              valid, false otherwise.
     */
    bool operator!=(nullptr_t) const;

    /**
     * Returns whether this handle to an IPreparedModel object is valid or not.
     *
     * @return bool true if V1_0::IPreparedModel (which could be V1_2::IPreparedModel) is
     *              invalid, false otherwise.
     */
    bool operator==(nullptr_t) const;

   private:
    /**
     * All versions of IPreparedModel are necessary because the preparedModel could be v1.0,
     * v1.2, or a later version. All these pointers logically represent the same object.
     *
     * The general strategy is: HIDL returns a V1_0 prepared model object, which
     * (if not nullptr) could be v1.0, v1.2, or a greater version. The V1_0
     * object is then "dynamically cast" to a V1_2 object. If successful,
     * mPreparedModelV1_2 will point to the same object as mPreparedModelV1_0; otherwise,
     * mPreparedModelV1_2 will be nullptr.
     *
     * In general:
     * * If the prepared model is truly v1.0, mPreparedModelV1_0 will point to a valid object
     *   and mPreparedModelV1_2 will be nullptr.
     * * If the prepared model is truly v1.2 or later, both mPreparedModelV1_0 and
     *   mPreparedModelV1_2 will point to the same valid object.
     *
     * Idiomatic usage: if mPreparedModelV1_2 is non-null, do V1_2 dispatch; otherwise,
     * do V1_0 dispatch.
     */
    sp<V1_0::IPreparedModel> mPreparedModelV1_0;
    sp<V1_0::IPreparedModel> mPreparedModelV1_2;
    //sp<V1_2::IPreparedModel> mPreparedModelV1_2;
};

}  // namespace nn
}  // namespace android
#endif //HIFI_BUILD

#endif  // ANDROID_ML_NN_RUNTIME_VERSIONED_INTERFACES_H
