
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

#include <SF/string.hpp>

#include <RCF/ClientStub.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ThreadLocalData.hpp>

namespace SF {

    bool getCurrentNativeWstringSerialization()
    {
        bool useNativeWstringSerialization = RCF::getDefaultNativeWstringSerialization();
        RCF::ClientStub * pClientStub = RCF::getCurrentClientStubPtr();
        RCF::RcfSession * pRcfSession = RCF::getCurrentRcfSessionPtr();

        if (pClientStub)
        {
            useNativeWstringSerialization = pClientStub->getNativeWstringSerialization();
        }
        else if (pRcfSession)
        {
            useNativeWstringSerialization = pRcfSession->getNativeWstringSerialization();
        }
        return useNativeWstringSerialization;
    }

}