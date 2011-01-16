
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

#include <RCF/ServerTransport.hpp>

#include <RCF/Service.hpp>

namespace RCF {

    I_ServerTransport::I_ServerTransport() :
        mReadWriteMutex(WriterPriority),
        mMaxMessageLength(getDefaultMaxMessageLength()),
        mConnectionLimit(RCF_DEFAULT_INIT)
    {}

    I_ServerTransport & I_ServerTransport::setMaxMessageLength(std::size_t maxMessageLength)
    {
        WriteLock writeLock(mReadWriteMutex);
        mMaxMessageLength = maxMessageLength;

        return *this;
    }

    std::size_t I_ServerTransport::getMaxMessageLength() const
    {
        ReadLock readLock(mReadWriteMutex);
        return mMaxMessageLength;
    }

    std::size_t I_ServerTransport::getConnectionLimit() const
    {
        ReadLock readLock(mReadWriteMutex);
        return mConnectionLimit;
    }

    I_ServerTransport & I_ServerTransport::setConnectionLimit(
        std::size_t connectionLimit)
    {
        WriteLock writeLock(mReadWriteMutex);
        mConnectionLimit = connectionLimit;

        return *this;
    }

    I_ServerTransport & I_ServerTransport::setThreadPool(
        ThreadPoolPtr threadPoolPtr)
    {
        I_Service & svc = dynamic_cast<I_Service &>(*this);
        svc.setThreadPool(threadPoolPtr);
        return *this;
    }

    std::size_t gDefaultMaxMessageLength = 1024*1024; // 1 Mb

    std::size_t getDefaultMaxMessageLength()
    {
        return gDefaultMaxMessageLength;
    }

    void setDefaultMaxMessageLength(std::size_t maxMessageLength)
    {
        gDefaultMaxMessageLength = maxMessageLength;
    }

    I_SessionState::I_SessionState() :
        mEnableReconnect(true)
    {
    }

    void I_SessionState::setEnableReconnect(bool enableReconnect)
    {
        mEnableReconnect = enableReconnect;
    }

    bool I_SessionState::getEnableReconnect()
    {
        return mEnableReconnect;
    }

} // namespace RCF
