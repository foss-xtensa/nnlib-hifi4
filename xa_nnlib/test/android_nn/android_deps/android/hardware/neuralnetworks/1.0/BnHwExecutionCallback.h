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
#ifndef HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_BNHWEXECUTIONCALLBACK_H
#define HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_BNHWEXECUTIONCALLBACK_H

#include <android/hardware/neuralnetworks/1.0/IHwExecutionCallback.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {

struct BnHwExecutionCallback : public ::android::hidl::base::V1_0::BnHwBase {
    explicit BnHwExecutionCallback(const ::android::sp<IExecutionCallback> &_hidl_impl);
    explicit BnHwExecutionCallback(const ::android::sp<IExecutionCallback> &_hidl_impl, const std::string& HidlInstrumentor_package, const std::string& HidlInstrumentor_interface);

    virtual ~BnHwExecutionCallback();

    ::android::status_t onTransact(
            uint32_t _hidl_code,
            const ::android::hardware::Parcel &_hidl_data,
            ::android::hardware::Parcel *_hidl_reply,
            uint32_t _hidl_flags = 0,
            TransactCallback _hidl_cb = nullptr) override;


    typedef IExecutionCallback Pure;

    typedef android::hardware::details::bnhw_tag _hidl_tag;

    ::android::sp<IExecutionCallback> getImpl() { return _hidl_mImpl; }
    // Methods from ::android::hardware::neuralnetworks::V1_0::IExecutionCallback follow.
    static ::android::status_t _hidl_notify(
            ::android::hidl::base::V1_0::BnHwBase* _hidl_this,
            const ::android::hardware::Parcel &_hidl_data,
            ::android::hardware::Parcel *_hidl_reply,
            TransactCallback _hidl_cb);



private:
    // Methods from ::android::hardware::neuralnetworks::V1_0::IExecutionCallback follow.

    // Methods from ::android::hidl::base::V1_0::IBase follow.
    ::android::hardware::Return<void> ping();
    using getDebugInfo_cb = ::android::hidl::base::V1_0::IBase::getDebugInfo_cb;
    ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb);

    ::android::sp<IExecutionCallback> _hidl_mImpl;
};

}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_NEURALNETWORKS_V1_0_BNHWEXECUTIONCALLBACK_H
