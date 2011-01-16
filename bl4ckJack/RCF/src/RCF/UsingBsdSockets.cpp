
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

#include <iostream>
#include <string>

#include <boost/config.hpp>

#include <RCF/Exception.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/Tools.hpp>
#include <RCF/util/Platform/OS/BsdSockets.hpp>

#ifdef BOOST_WINDOWS

namespace RCF {

    inline void initWinsock()
    {
        WORD wVersion = MAKEWORD( 1, 0 );
        WSADATA wsaData;
        int ret = WSAStartup(wVersion, &wsaData);
        int err = Platform::OS::BsdSockets::GetLastError();
        RCF_VERIFY(ret == 0, Exception( _RcfError_Socket(), err, RcfSubsystem_Os, "WSAStartup()") );
    }

    inline void deinitWinsock()
    {
        WSACleanup();
    }

    RCF_ON_INIT_DEINIT_NAMED( initWinsock(), deinitWinsock(), WinsockInitDeinit )

} // namespace RCF

#endif


