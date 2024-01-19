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

#include "Callbacks.h"
#ifndef HIFI_BUILD
#include <android-base/logging.h>
#endif //HIFI_BUILD

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace implementation {

#ifndef HIFI_BUILD
CallbackBase::CallbackBase() : mNotified(false) {}

CallbackBase::~CallbackBase() {
    // Note that we cannot call CallbackBase::join_thread from here:
    // CallbackBase is intended to be reference counted, and it is possible that
    // the reference count drops to zero in the bound thread, causing the
    // bound thread to call this destructor. If a thread tries to join
    // itself, it throws an exception, producing a message like the
    // following:
    //
    //     terminating with uncaught exception of type std::__1::system_error:
    //     thread::join failed: Resource deadlock would occur
}

void CallbackBase::wait() {
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [this]{return mNotified;});
    join_thread_locked();
}

bool CallbackBase::on_finish(std::function<bool(void)> post_work) {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mPostWork != nullptr) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "CallbackBase::on_finish -- a post-work function has already been bound to "
                   "this callback object";
#endif //HIFI_BUILD
        return false;
    }
    if (post_work == nullptr) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "CallbackBase::on_finish -- the new post-work function is invalid";
#endif //HIFI_BUILD
        return false;
    }
    mPostWork = std::move(post_work);
    return true;
}

bool CallbackBase::bind_thread(std::thread&& asyncThread) {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mThread.joinable()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "CallbackBase::bind_thread -- a thread has already been bound to this "
                   "callback object";
#endif //HIFI_BUILD
        return false;
    }
    if (!asyncThread.joinable()) {
#ifndef HIFI_BUILD
        LOG(ERROR) << "CallbackBase::bind_thread -- the new thread is not joinable";
#endif //HIFI_BUILD
        return false;
    }
    mThread = std::move(asyncThread);
    return true;
}

void CallbackBase::join_thread() {
    std::lock_guard<std::mutex> lock(mMutex);
    join_thread_locked();
}

void CallbackBase::notify() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mNotified = true;
        if (mPostWork != nullptr) {
            bool success = mPostWork();
            if (!success) {
#ifndef HIFI_BUILD
                LOG(ERROR) << "CallbackBase::notify -- post work failed";
#endif //HIFI_BUILD
            }
        }
    }
    mCondition.notify_all();
}

void CallbackBase::join_thread_locked() {
    if (mThread.joinable()) {
        mThread.join();
    }
}

PreparedModelCallback::PreparedModelCallback() :
        mErrorStatus(ErrorStatus::GENERAL_FAILURE), mPreparedModel(nullptr) {}

PreparedModelCallback::~PreparedModelCallback() {}

Return<void> PreparedModelCallback::notify(ErrorStatus errorStatus,
                                           const sp<IPreparedModel>& preparedModel) {
    mErrorStatus = errorStatus;
    mPreparedModel = preparedModel;
    CallbackBase::notify();
    return Void();
}

ErrorStatus PreparedModelCallback::getStatus() {
    wait();
    return mErrorStatus;
}

sp<IPreparedModel> PreparedModelCallback::getPreparedModel() {
    wait();
    return mPreparedModel;
}
#endif //HIFI_BUILD

ExecutionCallback::ExecutionCallback() : mErrorStatus(ErrorStatus::GENERAL_FAILURE) {}

ExecutionCallback::~ExecutionCallback() {}

#ifndef HIFI_BUILD
Return<void> ExecutionCallback::notify(ErrorStatus errorStatus) {
    mErrorStatus = errorStatus;
    CallbackBase::notify();
    return Void();
}
#else
void  ExecutionCallback::notify(ErrorStatus errorStatus) {
    mErrorStatus = errorStatus;
}

#endif //HIFI_BUILD

ErrorStatus ExecutionCallback::getStatus() {
#ifndef HIFI_BUILD
    wait();
#endif //HIFI_BUILD
    return mErrorStatus;
}

}  // namespace implementation
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
