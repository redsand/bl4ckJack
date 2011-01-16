
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

#include <RCF/TcpEndpoint.hpp>

#include <boost/config.hpp>

#include <RCF/InitDeinit.hpp>
#include <RCF/SerializationProtocol.hpp>

#ifdef RCF_USE_BOOST_ASIO

#include <RCF/TcpAsioServerTransport.hpp>
#include <RCF/TcpClientTransport.hpp>

#elif defined(BOOST_WINDOWS)

#include <RCF/TcpIocpServerTransport.hpp>
#include <RCF/TcpClientTransport.hpp>

#else

#include <RCF/TcpClientTransport.hpp>

#endif

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/Registry.hpp>
#include <SF/SerializeParent.hpp>
#endif

namespace RCF {

    TcpEndpoint::TcpEndpoint()
    {}

    TcpEndpoint::TcpEndpoint(int port) :
        mIpAddress("127.0.0.1", port)
    {}

    TcpEndpoint::TcpEndpoint(const std::string &ip, int port) :
        mIpAddress(ip, port)
    {}

    TcpEndpoint::TcpEndpoint(const IpAddress & ipAddress) :
        mIpAddress(ipAddress)
    {}

    TcpEndpoint::TcpEndpoint(const TcpEndpoint &rhs) :
        mIpAddress(rhs.mIpAddress)
    {}

    EndpointPtr TcpEndpoint::clone() const
    {
        return EndpointPtr(new TcpEndpoint(*this));
    }

    std::string TcpEndpoint::getIp() const
    {
        return mIpAddress.getIp();
    }

    int TcpEndpoint::getPort() const
    {
        return mIpAddress.getPort();
    }

    std::string TcpEndpoint::asString() const
    {
        std::ostringstream os;
        std::string ip = getIp();
        if (ip.empty())
        {
            ip = "127.0.0.1";
        }
        os << "TCP endpoint " << ip << ":" << getPort();
        return os.str();
    }

    IpAddress TcpEndpoint::getIpAddress() const
    {
        return mIpAddress;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void TcpEndpoint::serialize(SF::Archive &ar)
    {
        // TODO: versioning.
        // ...

        serializeParent( (I_Endpoint*) 0, ar, *this);
        ar & mIpAddress;
    }

#endif

#ifdef RCF_USE_BOOST_ASIO

    std::auto_ptr<I_ServerTransport> TcpEndpoint::createServerTransport() const
    {
        return std::auto_ptr<I_ServerTransport>(
            new RCF::TcpAsioServerTransport(mIpAddress));
    }

    std::auto_ptr<I_ClientTransport> TcpEndpoint::createClientTransport() const
    {
        return std::auto_ptr<I_ClientTransport>(
            new RCF::TcpClientTransport(mIpAddress));
    }

#elif defined(BOOST_WINDOWS)

    std::auto_ptr<I_ServerTransport> TcpEndpoint::createServerTransport() const
    {
        return std::auto_ptr<I_ServerTransport>(
            new RCF::TcpIocpServerTransport(mIpAddress));
    }

    std::auto_ptr<I_ClientTransport> TcpEndpoint::createClientTransport() const
    {
        return std::auto_ptr<I_ClientTransport>(
            new RCF::TcpClientTransport(mIpAddress));
    }

#else

    std::auto_ptr<I_ServerTransport> TcpEndpoint::createServerTransport() const
    {
        // On non Windows platforms, server side RCF code requires 
        // RCF_USE_BOOST_ASIO to be defined, and the Boost.Asio library to 
        // be available.
        RCF_ASSERT(0);
        return std::auto_ptr<I_ServerTransport>();
    }

    std::auto_ptr<I_ClientTransport> TcpEndpoint::createClientTransport() const
    {
        return std::auto_ptr<I_ClientTransport>(
            new RCF::TcpClientTransport(mIpAddress));
    }

#endif

    inline void initTcpEndpointSerialization()
    {
#ifdef RCF_USE_SF_SERIALIZATION
        SF::registerType( (TcpEndpoint *) 0, "RCF::TcpEndpoint");
        SF::registerBaseAndDerived( (I_Endpoint *) 0, (TcpEndpoint *) 0);
#endif
    }

    RCF_ON_INIT_NAMED( initTcpEndpointSerialization(), InitTcpEndpointSerialization );

} // namespace RCF
