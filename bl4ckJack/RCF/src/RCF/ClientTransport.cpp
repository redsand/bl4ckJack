
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

#include <RCF/ClientTransport.hpp>

#include <RCF/Exception.hpp>
#include <RCF/ServerTransport.hpp>

namespace RCF {

    I_ClientTransport::I_ClientTransport() :
        mMaxMessageLength(getDefaultMaxMessageLength()),
        mLastRequestSize(0),
        mLastResponseSize(0)
    {}

    bool I_ClientTransport::isConnected()
    {
        return true;
    }

    void I_ClientTransport::setMaxMessageLength(std::size_t maxMessageLength)
    {
        mMaxMessageLength = maxMessageLength;
    }

    std::size_t I_ClientTransport::getMaxMessageLength() const
    {
        return mMaxMessageLength;
    }

    RcfSessionWeakPtr I_ClientTransport::getRcfSession()
    {
        return mRcfSessionWeakPtr;
    }

    void I_ClientTransport::setRcfSession(RcfSessionWeakPtr rcfSessionWeakPtr)
    {
        mRcfSessionWeakPtr = rcfSessionWeakPtr;
    }

    std::size_t I_ClientTransport::getLastRequestSize()
    {
        return mLastRequestSize;
    }

    std::size_t I_ClientTransport::getLastResponseSize()
    {
        return mLastResponseSize;
    }

    void I_ClientTransportCallback::setAsyncDispatcher(RcfServer & server)
    {
        RCF_ASSERT(!mpAsyncDispatcher);
        mpAsyncDispatcher = &server;
    }

    RcfServer * I_ClientTransportCallback::getAsyncDispatcher()
    {
        return mpAsyncDispatcher;
    }

} // namespace RCF
