
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

#ifndef INCLUDE_RCF_THREADLOCALDATA_HPP
#define INCLUDE_RCF_THREADLOCALDATA_HPP

#include <boost/shared_ptr.hpp>

#include <RCF/ByteBuffer.hpp>
#include <RCF/Export.hpp>
#include <RCF/RecursionLimiter.hpp>

namespace RCF {

    class ObjectCache;
    class ClientStub;
    class RcfSession;
    class ThreadInfo;
    class UdpSessionState;
    class I_Future;
    class AmiNotification;

    typedef boost::shared_ptr<ClientStub>       ClientStubPtr;
    typedef boost::shared_ptr<RcfSession>       RcfSessionPtr;
    typedef boost::shared_ptr<ThreadInfo>       ThreadInfoPtr;
    typedef boost::shared_ptr<UdpSessionState>  UdpSessionStatePtr;   

    RCF_EXPORT ObjectCache &        getThreadLocalObjectCache();
    RCF_EXPORT ClientStub *         getCurrentClientStubPtr();
    
    RCF_EXPORT void                 pushCurrentClientStub(
                                        ClientStub * pClientStub);

    RCF_EXPORT void                 popCurrentClientStub();

    RCF_EXPORT RcfSession *         getCurrentRcfSessionPtr();

    RCF_EXPORT void                 setCurrentRcfSessionPtr(
                                        RcfSession * pRcfSession = NULL);

    RCF_EXPORT ThreadInfoPtr        getThreadInfoPtr();

    RCF_EXPORT void                 setThreadInfoPtr(
                                        ThreadInfoPtr threadInfoPtr);

    RCF_EXPORT UdpSessionStatePtr   getCurrentUdpSessionStatePtr();

    RCF_EXPORT void                 setCurrentUdpSessionStatePtr(
                                        UdpSessionStatePtr udpSessionStatePtr);

    RCF_EXPORT RcfSession &         getCurrentRcfSession();

    RecursionState<int, int> &      getCurrentRcfSessionRecursionState();

    AmiNotification &                getCurrentAmiNotification();

} // namespace RCF

#endif // ! INCLUDE_RCF_THREADLOCALDATA_HPP
