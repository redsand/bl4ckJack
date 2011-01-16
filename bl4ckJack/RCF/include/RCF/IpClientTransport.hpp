
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

#ifndef INCLUDE_RCF_IPCLIENTTRANSPORT_HPP
#define INCLUDE_RCF_IPCLIENTTRANSPORT_HPP

#include <string>

#include <RCF/Export.hpp>
#include <RCF/IpAddress.hpp>

namespace RCF {

    class RCF_EXPORT I_IpClientTransport
    {
    public:
        virtual             ~I_IpClientTransport();
        
        IpAddress           getLocalIp();
        void                setLocalIp(const IpAddress & localIp);

        IpAddress           getAssignedLocalIp();

    protected:
        IpAddress           mLocalIp;
        IpAddress           mAssignedLocalIp;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_IPCLIENTTRANSPORT_HPP
