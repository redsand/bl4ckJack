
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

#include <RCF/IpClientTransport.hpp>

#include <RCF/Exception.hpp>
#include <RCF/Tools.hpp>
#include <RCF/util/Platform/OS/BsdSockets.hpp>

namespace RCF {

    I_IpClientTransport::~I_IpClientTransport()
    {
    }

    IpAddress I_IpClientTransport::getLocalIp()
    {
        return mLocalIp;
    }

    void I_IpClientTransport::setLocalIp(const IpAddress & localIp)
    {
        mLocalIp = localIp;
    }

    IpAddress I_IpClientTransport::getAssignedLocalIp()
    {
        return mAssignedLocalIp;
    }

} // namespace RCF
