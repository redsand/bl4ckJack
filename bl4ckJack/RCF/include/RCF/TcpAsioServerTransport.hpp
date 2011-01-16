
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

#ifndef INCLUDE_RCF_TCPASIOSERVERTRANSPORT_HPP
#define INCLUDE_RCF_TCPASIOSERVERTRANSPORT_HPP

#include <RCF/AsioServerTransport.hpp>

namespace RCF {

    class RCF_EXPORT TcpAsioServerTransport : 
        public AsioServerTransport,
        public I_IpServerTransport
    {
    public:
        TcpAsioServerTransport(const IpAddress & ipAddress);
        TcpAsioServerTransport(const std::string & ip, int port);

        ServerTransportPtr clone();

        // I_IpServerTransport implementation
        int                    getPort() const;

    private:

        AsioSessionStatePtr     implCreateSessionState();
        void                    implOpen();

        void                    onServerStart(RcfServer & server);

        ClientTransportAutoPtr  implCreateClientTransport(
                                    const I_Endpoint &endpoint);

    private:
        IpAddress               mIpAddress;

        int                     mAcceptorFd;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_TCPASIOSERVERTRANSPORT_HPP
