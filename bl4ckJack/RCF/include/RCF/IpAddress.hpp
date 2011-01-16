
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

#ifndef INCLUDE_RCF_IPADDRESS_HPP
#define INCLUDE_RCF_IPADDRESS_HPP

#include <string>

#include <RCF/Export.hpp>
#include <RCF/Config.hpp>
#include <RCF/ServerTransport.hpp>

#include <RCF/util/Platform/OS/BsdSockets.hpp>

namespace RCF {

#ifdef RCF_USE_IPV6

    typedef in6_addr In6Addr;
    typedef sockaddr_in6 SockAddrIn6;
    typedef sockaddr_storage SockAddrStorage;

#else

    // Dummy IPv6 structures.

    struct In6Addr {
        union {
            u_char Byte[16];
        } u;
    };

    struct SockAddrIn6 {
        short sin6_family;
        u_short sin6_port;
        u_long sin6_flowinfo;
        struct In6Addr sin6_addr;
        u_long sin6_scope_id;
    };

    struct SockAddrStorage
    {
        sockaddr_in mSockAddrIn;
    };

#endif

    class Exception;
    typedef boost::shared_ptr<Exception> ExceptionPtr;

    class RCF_EXPORT IpAddress : public I_RemoteAddress
    {
    public:

        enum Type { V4_or_V6, V4, V6 };

        static void setDefaultResolveProtocol(Type type);
        static Type getDefaultResolveProtocol();
        
        IpAddress();
        explicit IpAddress(Type restrictTo);
        explicit IpAddress(const std::string & ip);
        explicit IpAddress(const std::string & ip, int port);
        explicit IpAddress(const std::string & ip, int port, Type restrictTo);
        explicit IpAddress(const sockaddr_in &addr);
        explicit IpAddress(const SockAddrIn6 &addr);
        explicit IpAddress(const sockaddr &addr, std::size_t addrLen, Type type);
        explicit IpAddress(int fd, Type type);

        int             createSocket(int socketType = SOCK_STREAM, int protocol = IPPROTO_TCP) const;
        void            getSockAddr(sockaddr *&, Platform::OS::BsdSockets::socklen_t &) const;
        std::string     getIp() const;
        int             getPort() const;
        Type            getType();

        void            resolve() const;
        void            resolve(ExceptionPtr & e) const;
        std::string     string() const;
        bool            empty() const;
        bool            isResolved() const;
        bool            isBroadcast() const;
        bool            isMulticast() const;
        bool            matches(const IpAddress & rhs, std::size_t bits = std::size_t(-1)) const;

        void            setPort(int port);

        bool            operator==(const IpAddress & rhs) const;
        bool            operator<(const IpAddress &rhs) const;

#ifdef RCF_USE_SF_SERIALIZATION
        void            serialize(SF::Archive &ar);
#endif // RCF_USE_SF_SERIALIZATION

    private:

        void            extractIpAndPort();
        
        mutable Type            mType;
        mutable bool            mResolved;
        mutable sockaddr_in     mAddrV4;
        mutable SockAddrIn6     mAddrV6;

        std::string             mIp;
        int                     mPort;
    };

    class RCF_EXPORT IpAddressV4 : public IpAddress
    {
    public:

        IpAddressV4() : IpAddress(V4) 
        {}

        IpAddressV4(const std::string & ip) : IpAddress(ip, 0, V4) 
        {}
        
        IpAddressV4(const std::string & ip, int port) : IpAddress(ip, port, V4) 
        {}

        IpAddressV4(const sockaddr_in & addr) : IpAddress(addr) 
        {}
    };

    class RCF_EXPORT IpAddressV6 : public IpAddress
    {
    public:

        IpAddressV6() : IpAddress(V6) 
        {}

        IpAddressV6(const std::string & ip) : IpAddress(ip, 0, V6) 
        {}

        IpAddressV6(const std::string & ip, int port) : IpAddress(ip, port, V6) 
        {}

        IpAddressV6(const SockAddrIn6 & addr) : IpAddress(addr) 
        {}
    };

} // namespace RCF

RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::IpAddress)
RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::IpAddress::Type)

#endif // ! INCLUDE_RCF_IPADDRESS_HPP
