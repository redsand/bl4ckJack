
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

#ifndef INCLUDE_RCF_UNIXLOCALSERVERTRANSPORT_HPP
#define INCLUDE_RCF_UNIXLOCALSERVERTRANSPORT_HPP

#if defined(BOOST_WINDOWS)
#error Unix domain sockets not supported on Windows.
#endif

#include <boost/version.hpp>
#if BOOST_VERSION < 103600
#error Need Boost 1.36.0 or later for Unix domain socket support.
#endif

#include <RCF/AsioServerTransport.hpp>
#include <RCF/Export.hpp>

namespace RCF {

    class RCF_EXPORT UnixLocalServerTransport : 
        public AsioServerTransport
    {
    public:

        UnixLocalServerTransport(const std::string & fileName);

        ServerTransportPtr clone();

        AsioSessionStatePtr implCreateSessionState();
        void implOpen();
        ClientTransportAutoPtr implCreateClientTransport(
            const I_Endpoint &endpoint);

        std::string getPipeName() const;

        void onServerStart(RcfServer & server);
        void onServerStop(RcfServer & server);

    private:

        const std::string               mFileName;      
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_UNIXLOCALSERVERTRANSPORT_HPP
