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
#ifndef HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H
#define HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H

#include <hidl/HidlSupport.h>
#include <hidl/MQDescriptor.h>
#include <utils/NativeHandle.h>
#include <utils/misc.h>

namespace android {
namespace hidl {
namespace base {
namespace V1_0 {

// Forward declaration for forward reference support:
struct DebugInfo;

struct DebugInfo final {
    // Forward declaration for forward reference support:
    enum class Architecture : int32_t;

    enum class Architecture : int32_t {
        UNKNOWN = 0,
        IS_64BIT = 1 /* (::android::hidl::base::V1_0::DebugInfo::Architecture.UNKNOWN implicitly + 1) */,
        IS_32BIT = 2 /* (::android::hidl::base::V1_0::DebugInfo::Architecture.IS_64BIT implicitly + 1) */,
    };

    int32_t pid __attribute__ ((aligned(4)));
    uint64_t ptr __attribute__ ((aligned(8)));
    ::android::hidl::base::V1_0::DebugInfo::Architecture arch __attribute__ ((aligned(4)));
};

static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, pid) == 0, "wrong offset");
static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, ptr) == 8, "wrong offset");
static_assert(offsetof(::android::hidl::base::V1_0::DebugInfo, arch) == 16, "wrong offset");
static_assert(sizeof(::android::hidl::base::V1_0::DebugInfo) == 24, "wrong size");
static_assert(__alignof(::android::hidl::base::V1_0::DebugInfo) == 8, "wrong alignment");

//
// type declarations for package
//

template<typename>
static inline std::string toString(int32_t o);
static inline std::string toString(::android::hidl::base::V1_0::DebugInfo::Architecture o);

constexpr int32_t operator|(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const int32_t lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(lhs | static_cast<int32_t>(rhs));
}
constexpr int32_t operator|(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) | rhs);
}
constexpr int32_t operator&(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const int32_t lhs, const ::android::hidl::base::V1_0::DebugInfo::Architecture rhs) {
    return static_cast<int32_t>(lhs & static_cast<int32_t>(rhs));
}
constexpr int32_t operator&(const ::android::hidl::base::V1_0::DebugInfo::Architecture lhs, const int32_t rhs) {
    return static_cast<int32_t>(static_cast<int32_t>(lhs) & rhs);
}
static int32_t &operator|=(int32_t& v, const ::android::hidl::base::V1_0::DebugInfo::Architecture e) {
    v |= static_cast<int32_t>(e);
    return v;
}
static int32_t &operator&=(int32_t& v, const ::android::hidl::base::V1_0::DebugInfo::Architecture e) {
    v &= static_cast<int32_t>(e);
    return v;
}

static inline std::string toString(const ::android::hidl::base::V1_0::DebugInfo& o);
static inline bool operator==(const ::android::hidl::base::V1_0::DebugInfo& lhs, const ::android::hidl::base::V1_0::DebugInfo& rhs);
static inline bool operator!=(const ::android::hidl::base::V1_0::DebugInfo& lhs, const ::android::hidl::base::V1_0::DebugInfo& rhs);

//
// type header definitions for package
//

template<>
inline std::string toString<::android::hidl::base::V1_0::DebugInfo::Architecture>(int32_t o) {
    using ::android::hardware::details::toHexString;
    std::string os;
    ::android::hardware::hidl_bitfield<::android::hidl::base::V1_0::DebugInfo::Architecture> flipped = 0;
    bool first = true;
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN)) {
        os += (first ? "" : " | ");
        os += "UNKNOWN";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN;
    }
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT)) {
        os += (first ? "" : " | ");
        os += "IS_64BIT";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT;
    }
    if ((o & ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT) == static_cast<int32_t>(::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT)) {
        os += (first ? "" : " | ");
        os += "IS_32BIT";
        first = false;
        flipped |= ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT;
    }
    if (o != flipped) {
        os += (first ? "" : " | ");
        os += toHexString(o & (~flipped));
    }os += " (";
    os += toHexString(o);
    os += ")";
    return os;
}

static inline std::string toString(::android::hidl::base::V1_0::DebugInfo::Architecture o) {
    using ::android::hardware::details::toHexString;
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN) {
        return "UNKNOWN";
    }
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT) {
        return "IS_64BIT";
    }
    if (o == ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT) {
        return "IS_32BIT";
    }
    std::string os;
    os += toHexString(static_cast<int32_t>(o));
    return os;
}

static inline std::string toString(const ::android::hidl::base::V1_0::DebugInfo& o) {
    using ::android::hardware::toString;
    std::string os;
    os += "{";
    os += ".pid = ";
    os += ::android::hardware::toString(o.pid);
    os += ", .ptr = ";
    os += ::android::hardware::toString(o.ptr);
    os += ", .arch = ";
    os += ::android::hidl::base::V1_0::toString(o.arch);
    os += "}"; return os;
}

static inline bool operator==(const ::android::hidl::base::V1_0::DebugInfo& lhs, const ::android::hidl::base::V1_0::DebugInfo& rhs) {
    if (lhs.pid != rhs.pid) {
        return false;
    }
    if (lhs.ptr != rhs.ptr) {
        return false;
    }
    if (lhs.arch != rhs.arch) {
        return false;
    }
    return true;
}

static inline bool operator!=(const ::android::hidl::base::V1_0::DebugInfo& lhs, const ::android::hidl::base::V1_0::DebugInfo& rhs){
    return !(lhs == rhs);
}


}  // namespace V1_0
}  // namespace base
}  // namespace hidl
}  // namespace android

//
// global type declarations for package
//

#if 0 //ppn
namespace android {
namespace hardware {
namespace details {
template<> constexpr std::array<::android::hidl::base::V1_0::DebugInfo::Architecture, 3> hidl_enum_values<::android::hidl::base::V1_0::DebugInfo::Architecture> = {
    ::android::hidl::base::V1_0::DebugInfo::Architecture::UNKNOWN,
    ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_64BIT,
    ::android::hidl::base::V1_0::DebugInfo::Architecture::IS_32BIT,
};
}  // namespace details
}  // namespace hardware
}  // namespace android
#endif //ppn


#endif  // HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_TYPES_H
