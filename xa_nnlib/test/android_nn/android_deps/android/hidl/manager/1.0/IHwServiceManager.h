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
#ifndef HIDL_GENERATED_ANDROID_HIDL_MANAGER_V1_0_IHWSERVICEMANAGER_H
#define HIDL_GENERATED_ANDROID_HIDL_MANAGER_V1_0_IHWSERVICEMANAGER_H

#include <android/hidl/manager/1.0/IServiceManager.h>

#include <android/hidl/base/1.0/BnHwBase.h>
#include <android/hidl/base/1.0/BpHwBase.h>
#include <android/hidl/base/1.0/hwtypes.h>
#include <android/hidl/manager/1.0/BnHwServiceNotification.h>
#include <android/hidl/manager/1.0/BpHwServiceNotification.h>

#include <hidl/Status.h>
#include <hwbinder/IBinder.h>
#include <hwbinder/Parcel.h>

namespace android {
namespace hidl {
namespace manager {
namespace V1_0 {
::android::status_t readEmbeddedFromParcel(
        const ::android::hidl::manager::V1_0::IServiceManager::InstanceDebugInfo &obj,
        const ::android::hardware::Parcel &parcel,
        size_t parentHandle,
        size_t parentOffset);

::android::status_t writeEmbeddedToParcel(
        const ::android::hidl::manager::V1_0::IServiceManager::InstanceDebugInfo &obj,
        ::android::hardware::Parcel *parcel,
        size_t parentHandle,
        size_t parentOffset);

}  // namespace V1_0
}  // namespace manager
}  // namespace hidl
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HIDL_MANAGER_V1_0_IHWSERVICEMANAGER_H
