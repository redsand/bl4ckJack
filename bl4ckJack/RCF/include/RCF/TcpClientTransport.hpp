
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

#ifndef INCLUDE_RCF_TCPCLIENTTRANSPORT_HPP
#define INCLUDE_RCF_TCPCLIENTTRANSPORT_HPP

#include <RCF/AsyncFilter.hpp>
#include <RCF/BsdClientTransport.hpp>
#include <RCF/ByteOrdering.hpp>
#include <RCF/ClientProgress.hpp>
#include <RCF/ClientTransport.hpp>
#include <RCF/Exception.hpp>
#include <RCF/Export.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/IpClientTransport.hpp>

namespace RCF {

    class RCF_EXPORT TcpClientTransport : 
        public BsdClientTransport, 
        public I_IpClientTransport
    {
    public:
        TcpClientTransport(const TcpClientTransport &rhs);
        TcpClientTransport(const IpAddress &remoteAddr);
        TcpClientTransport(const std::string & ip, int port);
        TcpClientTransport(int fd);

        ~TcpClientTransport();

        ClientTransportAutoPtr  clone() const;

        void                    implConnect(
                                    I_ClientTransportCallback &clientStub, 
                                    unsigned int timeoutMs);

        void                    implConnectAsync(
                                    I_ClientTransportCallback &clientStub, 
                                    unsigned int timeoutMs);

        void                    implClose();
        EndpointPtr             getEndpointPtr() const;

        void                    setRemoteAddr(const IpAddress &remoteAddr);
        IpAddress               getRemoteAddr() const;

        

    private:

        void                    beginDnsLookup();

        void                    endDnsLookup(
                                    OverlappedAmiPtr overlappedPtr,
                                    IpAddress ipAddress, 
                                    ExceptionPtr e);

        static void             dnsLookupTask(
                                    OverlappedAmiPtr overlappedPtr,
                                    IpAddress ipAddress);

        void                    setupSocket();
        void                    setupSocket(Exception & e);

        IpAddress               mRemoteAddr;
    };
    
} // namespace RCF

#endif // ! INCLUDE_RCF_TCPCLIENTTRANSPORT_HPP
