
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

#ifndef INCLUDE_UTIL_PLATFORM_OS_THREADID_HPP
#define INCLUDE_UTIL_PLATFORM_OS_THREADID_HPP

#include <boost/config.hpp>

#if defined(BOOST_WINDOWS) || defined(_WIN32)
#include "Windows/ThreadId.hpp"
#elif defined(__CYGWIN__)
#include "Unix/ThreadId.hpp"
#elif defined(__unix__)
#include "Unix/ThreadId.hpp"
#elif defined(__APPLE__)
#include "Unix/ThreadId.hpp"
#else
#include "UnknownOS/ThreadId.hpp"
#endif

#endif // ! INCLUDE_UTIL_PLATFORM_OS_THREADID_HPP
