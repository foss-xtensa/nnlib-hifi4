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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_IPREPAREDMODELCALLBACK_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_IPREPAREDMODELCALLBACK_H

#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
#include <android/hidl/base/1.0/IBase.h>

#include <android/hidl/manager/1.0/IServiceNotification.h>

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {

struct IPreparedModelCallback : public ::android::hidl::base::V1_0::IBase {
    typedef android::hardware::details::i_tag _hidl_tag;

    // Forward declaration for forward reference support:

    /**
     * IPreparedModelCallback must be used to return a prepared model produced by an
     * asynchronous task launched from IDevice::prepareModel.
     */
    virtual bool isRemote() const override { return false; }


    /**
     * notify must be invoked immediately after the asynchronous task holding
     * this callback has finished preparing the model. If the model was
     * successfully prepared, notify must be invoked with ErrorStatus::NONE and
     * the prepared model. If the model was not able to be successfully
     * prepared, notify must be invoked with the appropriate ErrorStatus and
     * nullptr as the IPreparedModel. If the asynchronous task holding this
     * callback fails to launch or if the model provided to
     * IDevice::prepareModel is invalid, notify must be invoked with the
     * appropriate error as well as nullptr for the IPreparedModel.
     * 
     * @param status Error status returned from the asynchronous model
     *               preparation task; must be:
     *               - NONE if the asynchronous task successfully prepared the
     *                 model
     *               - DEVICE_UNAVAILABLE if driver is offline or busy
     *               - GENERAL_FAILURE if the asynchronous task resulted in an
     *                 unspecified error
     *               - INVALID_ARGUMENT if one of the input arguments to
     *                 prepareModel is invalid
     * @param preparedModel A model that has been asynchronously prepared for
     *                      execution. If the model was unable to be prepared
     *                      due to an error, nullptr must be passed in place of
     *                      the IPreparedModel object.
     */
    virtual ::android::hardware::Return<void> notify(::android::hardware::neuralnetworks::V1_0::ErrorStatus status, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModel>& preparedModel) = 0;

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
    static ::android::hardware::Return<::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>> castFrom(const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& parent, bool emitError = false);
    static ::android::hardware::Return<::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>> castFrom(const ::android::sp<::android::hidl::base::V1_0::IBase>& parent, bool emitError = false);

    static const char* descriptor;

    static ::android::sp<IPreparedModelCallback> tryGetService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IPreparedModelCallback> tryGetService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return tryGetService(str, getStub); }
    static ::android::sp<IPreparedModelCallback> tryGetService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return tryGetService(str, getStub); }
    static ::android::sp<IPreparedModelCallback> tryGetService(bool getStub) { return tryGetService("default", getStub); }
    static ::android::sp<IPreparedModelCallback> getService(const std::string &serviceName="default", bool getStub=false);
    static ::android::sp<IPreparedModelCallback> getService(const char serviceName[], bool getStub=false)  { std::string str(serviceName ? serviceName : "");      return getService(str, getStub); }
    static ::android::sp<IPreparedModelCallback> getService(const ::android::hardware::hidl_string& serviceName, bool getStub=false)  { std::string str(serviceName.c_str());      return getService(str, getStub); }
    static ::android::sp<IPreparedModelCallback> getService(bool getStub) { return getService("default", getStub); }
    __attribute__ ((warn_unused_result))::android::status_t registerAsService(const std::string &serviceName="default");
    static bool registerForNotifications(
            const std::string &serviceName,
            const ::android::sp<::android::hidl::manager::V1_0::IServiceNotification> &notification);
};

//
// type declarations for package
//

static inline std::string toString(const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& o);

//
// type header definitions for package
//

static inline std::string toString(const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& o) {
    std::string os = "[class or subclass of ";
    os += ::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback::descriptor;
    os += "]";
    os += o->isRemote() ? "@remote" : "@local";
    return os;
}


}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

//
// global type declarations for package
//


#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_IPREPAREDMODELCALLBACK_H
