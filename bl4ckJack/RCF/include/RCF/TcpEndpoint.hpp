
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

#ifndef INCLUDE_RCF_TCPENDPOINT_HPP
#define INCLUDE_RCF_TCPENDPOINT_HPP

#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <RCF/Endpoint.hpp>
#include <RCF/Export.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/TypeTraits.hpp>

namespace RCF {

    class I_ServerTransport;
    class I_ClientTransport;

    class RCF_EXPORT TcpEndpoint : public I_Endpoint
    {
    public:

        TcpEndpoint();
        TcpEndpoint(int port);
        TcpEndpoint(const std::string &ip, int port);
        TcpEndpoint(const IpAddress & ipAddress);
        TcpEndpoint(const TcpEndpoint &rhs);

        std::auto_ptr<I_ServerTransport>    createServerTransport() const;
        std::auto_ptr<I_ClientTransport>    createClientTransport() const;
        EndpointPtr                         clone() const;
        std::string                         getIp() const;
        int                                 getPort() const;
        std::string                         asString() const;
        IpAddress                           getIpAddress() const;

        bool operator<(const TcpEndpoint &rhs) const
        {
            return mIpAddress < rhs.mIpAddress;
        }

#ifdef RCF_USE_SF_SERIALIZATION

        /// Serializes the TcpEndpoint object.
        void serialize(SF::Archive &ar);

#endif

    private:
        IpAddress mIpAddress;
    };

    class TcpEndpointV4 : public TcpEndpoint
    {
    public:
        TcpEndpointV4(const std::string & ip, int port) : 
            TcpEndpoint( IpAddressV4(ip, port) )
        {
        }
    };

    class TcpEndpointV6 : public TcpEndpoint
    {
    public:
        TcpEndpointV6(const std::string & ip, int port) : 
            TcpEndpoint( IpAddressV6(ip, port) )
        {
        }
    };

} // namespace RCF

RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::TcpEndpoint)

#endif // ! INCLUDE_RCF_TCPENDPOINT_HPP
