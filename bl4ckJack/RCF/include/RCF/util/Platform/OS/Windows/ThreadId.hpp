
//******************************************************************************
// RCF - Remote Call Framework
//
// Copyright (c) 2005 - 2010, Delta V Software. All rights reserved.
// http://www.deltavsoft.com
//
// RCF is distributed under dual licenses - closed source or GPL.
// Consult your particular license for conditions of use.
//
// Version: 1.3
// Contact: jarl.lindrud <at> deltavsoft.com 
//
//******************************************************************************

#ifndef INCLUDE_UTIL_PLATFORM_OS_WINDOWS_THREADID_HPP
#define INCLUDE_UTIL_PLATFORM_OS_WINDOWS_THREADID_HPP

#include "windows.h"

namespace Platform {

    namespace OS {

        typedef int ThreadId;
        inline ThreadId GetCurrentThreadId() { return ::GetCurrentThreadId(); }

    }
}

#endif // ! INCLUDE_UTIL_PLATFORM_OS_WINDOWS_THREADID_HPP
