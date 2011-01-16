
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

#include <RCF/InitDeinit.hpp>

#include <RCF/ThreadLibrary.hpp>

// std::size_t
#include <cstdlib>

// std::size_t for vc6
#include <boost/config.hpp>

namespace RCF {

    Mutex gInitRefCountMutex;
    std::size_t gInitRefCount = 0;

    void init()
    {
        Lock lock(gInitRefCountMutex);
        if (gInitRefCount == 0)
        {
            util::invokeInitCallbacks();
        }
        ++gInitRefCount;
    }

    void deinit()
    {
        Lock lock(gInitRefCountMutex);
        --gInitRefCount;
        if (gInitRefCount == 0)
        {
            util::invokeDeinitCallbacks();
        }
    }

    RcfInitDeinit::RcfInitDeinit()
    {
        init();
    }

    RcfInitDeinit::~RcfInitDeinit()
    {
        deinit();
    }

    // In some situations, Win32 DLL's in particular, the user needs to be able 
    // to explicitly initialize and deinitialize the framework. To do that, define 
    // RCF_NO_AUTO_INIT_DEINIT and then manually call RCF::init() / RCF::deinit().

#ifndef RCF_NO_AUTO_INIT_DEINIT

    static RcfInitDeinit rcfInitDeinit;

#endif

} // namespace RCF
