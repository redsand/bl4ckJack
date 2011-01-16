
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

#include <RCF/SessionTimeoutService.hpp>

#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>

#include <RCF/util/Platform/OS/Sleep.hpp>

namespace RCF {

    SessionTimeoutService::SessionTimeoutService(
        boost::uint32_t sessionTimeoutMs,
        boost::uint32_t reapingIntervalMs) : 
            mSessionTimeoutMs(sessionTimeoutMs),
            mLastRunTimer(0),
            mReapingIntervalMs(reapingIntervalMs),
            mStopFlag(RCF_DEFAULT_INIT),
            mpRcfServer(RCF_DEFAULT_INIT)
    {
    }

    void SessionTimeoutService::onServiceAdded(RcfServer &server)
    {
        mpRcfServer = & server;

        mStopFlag = false;

        mTaskEntries.clear();

        mTaskEntries.push_back( TaskEntry(
            boost::bind(&SessionTimeoutService::cycle, this, _1, _2),
            boost::bind(&SessionTimeoutService::stop, this),
            "RCF Session Timeout"));
    }

    void SessionTimeoutService::onServiceRemoved(RcfServer &server)
    {
        RCF_UNUSED_VARIABLE(server);
        mpRcfServer = NULL;
    }

    void SessionTimeoutService::stop()
    {
        mStopFlag = true;
    }

    bool SessionTimeoutService::cycle(
        int timeoutMs,
        const volatile bool &stopFlag)
    {
        RCF_UNUSED_VARIABLE(timeoutMs);

        if (    !mLastRunTimer.elapsed(mReapingIntervalMs)
            &&  !stopFlag 
            &&  !mStopFlag)
        {
            Platform::OS::Sleep(1);
            return false;
        }

        mLastRunTimer.restart();

        mSessionsTemp.resize(0);

        mpRcfServer->enumerateSessions(std::back_inserter(mSessionsTemp));

        for (std::size_t i=0; i<mSessionsTemp.size(); ++i)
        {
            RcfSessionPtr rcfSessionPtr = mSessionsTemp[i].lock();
            if (rcfSessionPtr)
            {
                RCF::Timer touchTimer( rcfSessionPtr->getTouchTimestamp() );
                if (touchTimer.elapsed(mSessionTimeoutMs))
                {
                    rcfSessionPtr->disconnect();
                }
            }
        }

        return stopFlag || mStopFlag;
    }

} //namespace RCF
