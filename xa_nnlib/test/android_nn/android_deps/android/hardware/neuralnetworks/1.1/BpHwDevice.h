/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_BPHWDEVICE_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_BPHWDEVICE_H

#include <hidl/HidlTransportSupport.h>

#include <android/hardware/neuralnetworks/1.1/IHwDevice.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_1 {

struct BpHwDevice : public ::android::hardware::BpInterface<IDevice>, public ::android::hardware::details::HidlInstrumentor {
    explicit BpHwDevice(const ::android::sp<::android::hardware::IBinder> &_hidl_impl);

    typedef IDevice Pure;

    typedef android::hardware::details::bphw_tag _hidl_tag;

    virtual bool isRemote() const override { return true; }

    // Methods from ::android::hardware::neuralnetworks::V1_1::IDevice follow.
    static ::android::hardware::Return<void>  _hidl_getCapabilities_1_1(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, getCapabilities_1_1_cb _hidl_cb);
    static ::android::hardware::Return<void>  _hidl_getSupportedOperations_1_1(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, const ::android::hardware::neuralnetworks::V1_1::Model& model, getSupportedOperations_1_1_cb _hidl_cb);
    static ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::ErrorStatus>  _hidl_prepareModel_1_1(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, const ::android::hardware::neuralnetworks::V1_1::Model& model, ::android::hardware::neuralnetworks::V1_1::ExecutionPreference preference, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& callback);

    // Methods from ::android::hardware::neuralnetworks::V1_0::IDevice follow.
    ::android::hardware::Return<void> getCapabilities(getCapabilities_cb _hidl_cb) override;
    ::android::hardware::Return<void> getSupportedOperations(const ::android::hardware::neuralnetworks::V1_0::Model& model, getSupportedOperations_cb _hidl_cb) override;
    ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::ErrorStatus> prepareModel(const ::android::hardware::neuralnetworks::V1_0::Model& model, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& callback) override;
    ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::DeviceStatus> getStatus() override;

    // Methods from ::android::hardware::neuralnetworks::V1_1::IDevice follow.
    ::android::hardware::Return<void> getCapabilities_1_1(getCapabilities_1_1_cb _hidl_cb) override;
    ::android::hardware::Return<void> getSupportedOperations_1_1(const ::android::hardware::neuralnetworks::V1_1::Model& model, getSupportedOperations_1_1_cb _hidl_cb) override;
    ::android::hardware::Return<::android::hardware::neuralnetworks::V1_0::ErrorStatus> prepareModel_1_1(const ::android::hardware::neuralnetworks::V1_1::Model& model, ::android::hardware::neuralnetworks::V1_1::ExecutionPreference preference, const ::android::sp<::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback>& callback) override;

    // Methods from ::android::hidl::base::V1_0::IBase follow.
    ::android::hardware::Return<void> interfaceChain(interfaceChain_cb _hidl_cb) override;
    ::android::hardware::Return<void> debug(const ::android::hardware::hidl_handle& fd, const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& options) override;
    ::android::hardware::Return<void> interfaceDescriptor(interfaceDescriptor_cb _hidl_cb) override;
    ::android::hardware::Return<void> getHashChain(getHashChain_cb _hidl_cb) override;
    ::android::hardware::Return<void> setHALInstrumentation() override;
    ::android::hardware::Return<bool> linkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient, uint64_t cookie) override;
    ::android::hardware::Return<void> ping() override;
    ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb) override;
    ::android::hardware::Return<void> notifySyspropsChanged() override;
    ::android::hardware::Return<bool> unlinkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient) override;

private:
    std::mutex _hidl_mMutex;
    std::vector<::android::sp<::android::hardware::hidl_binder_death_recipient>> _hidl_mDeathRecipients;
};

}  // namespace V1_1
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_1_BPHWDEVICE_H
