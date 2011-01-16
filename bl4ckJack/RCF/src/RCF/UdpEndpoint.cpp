
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

#include <RCF/UdpEndpoint.hpp>

#include <RCF/InitDeinit.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/UdpClientTransport.hpp>
#include <RCF/UdpServerTransport.hpp>

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/Registry.hpp>
#endif

namespace RCF {

    UdpEndpoint::UdpEndpoint() :
        mEnableSharedAddressBinding(false)
    {}

    UdpEndpoint::UdpEndpoint(int port) :
        mIp("127.0.0.1", port),
        mEnableSharedAddressBinding(false)
    {}

    UdpEndpoint::UdpEndpoint(const std::string &ip, int port) :
        mIp(ip, port),
        mEnableSharedAddressBinding(false)
    {}

    UdpEndpoint::UdpEndpoint(const IpAddress & ipAddress) :
        mIp(ipAddress),
        mEnableSharedAddressBinding(false)
    {}

    UdpEndpoint::UdpEndpoint(const UdpEndpoint &rhs) :
        mIp(rhs.mIp),
        mMulticastIp(rhs.mMulticastIp),
        mEnableSharedAddressBinding(rhs.mEnableSharedAddressBinding)
    {}

    UdpEndpoint & UdpEndpoint::enableSharedAddressBinding(bool enable)
    {
        mEnableSharedAddressBinding = enable;
        return *this;
    }

    UdpEndpoint & UdpEndpoint::listenOnMulticast(const IpAddress & multicastIp)
    {
        mMulticastIp = multicastIp;

        if (!mMulticastIp.empty())
        {
            mEnableSharedAddressBinding = true;
        }

        return *this;
    }

    UdpEndpoint & UdpEndpoint::listenOnMulticast(const std::string & multicastIp)
    {
        return listenOnMulticast(IpAddress(multicastIp));
    }

    EndpointPtr UdpEndpoint::clone() const
    {
        return EndpointPtr(new UdpEndpoint(*this));
    }

    std::string UdpEndpoint::getIp() const
    {
        return mIp.getIp();
    }

    int UdpEndpoint::getPort() const
    {
        return mIp.getPort();
    }

    ServerTransportAutoPtr UdpEndpoint::createServerTransport() const
    {
        std::auto_ptr<UdpServerTransport> udpServerTransportPtr(
            new UdpServerTransport(mIp, mMulticastIp));

        if (mEnableSharedAddressBinding)
        {
            udpServerTransportPtr->enableSharedAddressBinding();
        }

        return ServerTransportAutoPtr(udpServerTransportPtr.release());
    }

    std::auto_ptr<I_ClientTransport> UdpEndpoint::createClientTransport() const
    {
        return std::auto_ptr<I_ClientTransport>(
            new UdpClientTransport(mIp));
    }

    std::string UdpEndpoint::asString() const
    {
        std::ostringstream os;
        os << "UDP endpoint " << mIp.string();
        return os.str();
    }


#ifdef RCF_USE_SF_SERIALIZATION

    void UdpEndpoint::serialize(SF::Archive &ar)
    {
        // TODO: versioning.
        // ...

        serializeParent( (I_Endpoint*) 0, ar, *this);
        ar & mIp;
    }

#endif

    inline void initUdpEndpointSerialization()
    {
#ifdef RCF_USE_SF_SERIALIZATION
        SF::registerType( (UdpEndpoint *) 0, "RCF::UdpEndpoint");
        SF::registerBaseAndDerived( (I_Endpoint *) 0, (UdpEndpoint *) 0);
#endif
    }

    RCF_ON_INIT_NAMED( initUdpEndpointSerialization(), InitUdpEndpointSerialization );

} // namespace RCF
