
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

#ifndef INCLUDE_UTIL_PLATFORM_OS_UNIX_THREADID_HPP
#define INCLUDE_UTIL_PLATFORM_OS_UNIX_THREADID_HPP

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/mpl/if.hpp>

#ifdef BOOST_HAS_PTHREADS
#include <pthread.h>
#endif

namespace Platform {

    namespace OS {

#ifdef BOOST_HAS_PTHREADS

#if defined(__CYGWIN__)

        typedef int ThreadId;
        inline ThreadId GetCurrentThreadId() { return reinterpret_cast<ThreadId>(static_cast<void*>(pthread_self())); }

#elif defined(__MACH__) && defined(__APPLE__)

        typedef boost::mpl::if_c<
            sizeof(void *) == 4, 
            boost::int32_t, 
            boost::int64_t>::type ThreadId;

        inline ThreadId GetCurrentThreadId() { return reinterpret_cast<ThreadId>(static_cast<void*>(pthread_self())); }

#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)

        typedef boost::mpl::if_c<
            sizeof(void *) == 4, 
            boost::int32_t, 
            boost::int64_t>::type ThreadId;

        inline ThreadId GetCurrentThreadId() { return reinterpret_cast<ThreadId>(static_cast<void*>(pthread_self())); }

#else

        typedef int ThreadId;
        inline ThreadId GetCurrentThreadId() { return static_cast<ThreadId>(pthread_self()); }

#endif

#else // BOOST_HAS_PTHREADS

        typedef int ThreadId;
        inline ThreadId GetCurrentThreadId() { return 0; }

#endif // BOOST_HAS_PTHREADS

    }
}

#endif // ! INCLUDE_UTIL_PLATFORM_OS_UNIX_THREADID_HPP
