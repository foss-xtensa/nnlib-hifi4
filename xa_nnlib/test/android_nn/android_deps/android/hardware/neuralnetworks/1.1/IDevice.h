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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_IDEVICE_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_IDEVICE_H

#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hardware/neuralnetworks/1.1/types.h>

#include <android/hidl/manager/1.0/IServiceNotification.h>

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_1 {

struct IDevice : public ::android::hardware::neuralnetworks::V1_0::IDevice {
    typedef android::hardware::details::i_tag _hidl_tag;

    // Forward declaration for forward reference support:

    /**
     * This interface represents a device driver.
     */
    virtual bool isRemote() const override { return false; }


    using getCapabilities_cb = std::function<void(::android::hardware::neuralnetworks::V1_0::ErrorStatus status, const ::android::hardware::neuralnetworks::V1_0::Capabilities& capabilities)>;
    /**
     * Gets the capabilities of a driver.
     * 
     * @return status Error status of the call, must be:
     *                - NONE if successful
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     * @return capabilities Capabilities of the driver.
     */
    virtual ::android::hardware::Return<void> getCapabilities(getCapabilities_cb _hidl_cb) = 0;

    using getSupportedOperations_cb = std::function<void(::android::hardware::neuralnetworks::V1_0::ErrorStatus status, const ::android::hardware::hidl_vec<bool>& supportedOperations)>;
    /**
     * Gets the supported operations in a model.
     * 
     * getSupportedOperations indicates which operations of a model are fully
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
    virtual ::android::hardware::Return<void> getSupportedOperations(const ::android::hardware::neuralnetworks::V1_0::Model& model, getSupportedOperations_cb _hidl_cb) = 0;

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
    virtual ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::ErrorStatus> prepareModel(const ::android::hardware::neuralnetworks::V1_0::Model& model, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& callback) = 0;

    /**
     * Returns the current status of a driver.
     * 
     * @return status Status of the driver, one of:
     *                - DeviceStatus::AVAILABLE
     *                - DeviceStatus::BUSY
     *                - DeviceStatus::OFFLINE
     *                - DeviceStatus::UNKNOWN
     */
    virtual ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::DeviceStatus> getStatus() = 0;

    using getCapabilities_1_1_cb = std::function<void(::android::hardware::neuralnetworks::V1_0::ErrorStatus status, const ::android::hardware::neuralnetworks::V1_1::Capabilities& capabilities)>;
    /**
     * Gets the capabilities of a driver.
     * 
     * Note that @1.1::Capabilities provides performance information
     * on relaxed calculations, whereas @1.0::Capabilities does not.
     * 
     * @return status Error status of the call, must be:
     *                - NONE if successful
     *                - DEVICE_UNAVAILABLE if driver is offline or busy
     *                - GENERAL_FAILURE if there is an unspecified error
     * @return capabilities Capabilities of the driver.
     */
    virtual ::android::hardware::Return<void> getCapabilities_1_1(getCapabilities_1_1_cb _hidl_cb) = 0;

    using getSupportedOperations_1_1_cb = std::function<void(::android::hardware::neuralnetworks::V1_0::ErrorStatus status, const ::android::hardware::hidl_vec<bool>& supportedOperations)>;
    /**
     * Gets the supported operations in a model.
     * 
     * getSupportedOperations indicates which operations of a model are fully
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
    virtual ::android::hardware::Return<void> getSupportedOperations_1_1(const ::android::hardware::neuralnetworks::V1_1::Model& model, getSupportedOperations_1_1_cb _hidl_cb) = 0;

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
     * @param preference Indicates the intended execution behavior of a prepared
     *                   model.
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
    virtual ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::ErrorStatus> prepareModel_1_1(const ::android::hardware::neuralnetworks::V1_1::Model& model, ::android::hardware::neuralnetworks::V1_1::ExecutionPreference preference, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& callback) = 0;

    using interfaceChain_cb = std::function<void(const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& descriptors)>;
    virtual ::android::hardware::Return<void> interfaceChain(interfaceChain_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> debug(const ::android::hardware::hidl_handle& fd, const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& options) override;

    using interfaceDescriptor_cb = std::function<void(const ::android::hardware::hidl_string& descriptor)>;
    virtual ::android::hardware::Return<void> interfaceDescriptor(interfaceDescriptor_cb _hidl_cb) override;

    using getHashChain_cb = std::function<void(const ::android::hardware::hidl_vec<::android::hardware::hidl_array<uint8_t, 32>>& hashchain)>;
    virtual ::android::hardware::Return<void> getHashChain(getHashChain_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> setHALInstrumentation() override;

    virtual ::android::hardware::Return<bool> linkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient, uint64_t cookie) override;

    virtual ::android::hardware::Return<void> ping() override;

    using getDebugInfo_cb = std::function<void(const ::android::hidl::base::V1_0::DebugInfo& info)>;
    virtual ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb) override;

    virtual ::android::hardware::Return<void> notifySyspropsChanged() override;

    virtual ::android::hardware::Return<bool> unlinkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient) override;
    // cast static functions
    static ::android::hardware::Return<::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>> castFrom(const ::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>& parent, bool emitError = false);
    static ::android::hardware::Return<::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>> castFrom(const ::android::sp<::android::hardware::neuralnetworks::V1_0::IDevice>& parent, bool emitError = false);
    static ::android::hardware::Return<::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>> castFrom(const ::android::sp<::android::hidl::base::V1_0::IBase>& parent, bool emitError = false);

    static const char* descriptor;

    static ::android::sp<IDevice> tryGetService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IDevice> tryGetService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return tryGetService(str, getStub); }
    static ::android::sp<IDevice> tryGetService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return tryGetService(str, getStub); }
    static ::android::sp<IDevice> tryGetService(bool getStub) { return tryGetService("default", getStub); }
    static ::android::sp<IDevice> getService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IDevice> getService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return getService(str, getStub); }
    static ::android::sp<IDevice> getService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return getService(str, getStub); }
    static ::android::sp<IDevice> getService(bool getStub) { return getService("default", getStub); }
    __attribute__ ((warn_unused_result))::android::status_t registerAsService(const std::string &serviceName="default");
    static bool registerForNotifications(
            const std::string &serviceName,
            const ::android::sp<::android::hidl::manager::V1_0::IServiceNotification> &notification);
};

//
// type declarations for package
//

static inline std::string toString(const ::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>& o);

//
// type header definitions for package
//

static inline std::string toString(const ::android::sp<::android::hardware::neuralnetworks::V1_1::IDevice>& o) {
    std::string os = "[class or subclass of ";
    os += ::android::hardware::neuralnetworks::V1_1::IDevice::descriptor;
    os += "]";
    os += o->isRemote() ? "@remote" : "@local";
    return os;
}


}  // namespace V1_1
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

//
// global type declarations for package
//


#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_IDEVICE_H
