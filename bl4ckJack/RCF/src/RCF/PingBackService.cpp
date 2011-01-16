
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

#include <RCF/PingBackService.hpp>

#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>

namespace RCF {

    PingBackService::PingBackService() : 
        mStopFlag(RCF_DEFAULT_INIT)
    {
    }

    void PingBackService::onServiceAdded(RcfServer &server)
    {
        RCF_UNUSED_VARIABLE(server);

        mStopFlag = false;

        mTaskEntries.clear();

        mTaskEntries.push_back( TaskEntry(
            boost::bind(&PingBackService::cycle, this, _1, _2),
            boost::bind(&PingBackService::stop, this),
            "RCF Pingback"));
    }

    void PingBackService::onServiceRemoved(RcfServer &server)
    {
        RCF_UNUSED_VARIABLE(server);
    }
    void PingBackService::stop()
    {
        mStopFlag = true;
    }

    bool PingBackService::cycle(
        int timeoutMs,
        const volatile bool &stopFlag)
    {
        mTimerHeap.rebase();

        PingBackTimerEntry entry;

        while (     !stopFlag 
                &&  !mStopFlag 
                &&  mTimerHeap.getExpiredEntry(entry))
        {
            // Is the session still alive?
            RcfSessionPtr rcfSessionPtr( entry.second.lock() );
            if (rcfSessionPtr)
            {
                Lock lock(rcfSessionPtr->mIoStateMutex);

                // Is the timer entry still there? It can't change while we hold mIoStateMutex.
                if (mTimerHeap.compareTop(entry))
                {
                    // Calculate the next ping back time.
                    boost::uint32_t pingBackIntervalMs = rcfSessionPtr->getPingBackIntervalMs();
                    pingBackIntervalMs = RCF_MAX(pingBackIntervalMs, boost::uint32_t(1000));

                    boost::uint32_t nextFireMs = 
                        Platform::OS::getCurrentTimeMs() + pingBackIntervalMs;

                    PingBackTimerEntry nextEntry(nextFireMs, rcfSessionPtr);

                    mTimerHeap.remove(entry);
                    mTimerHeap.add(nextEntry);
                    rcfSessionPtr->mPingBackTimerEntry = nextEntry;

                    // Only send a pingback, if previous one has completed.
                    if (!rcfSessionPtr->mWritingPingBack)
                    {
                        rcfSessionPtr->sendPingBack();
                    }

                    mTimerHeap.add(nextEntry);
                }
            }
        } 

        boost::uint32_t queueTimeoutMs = RCF_MIN(
            static_cast<boost::uint32_t>(timeoutMs),
            mTimerHeap.getNextEntryTimeoutMs());

        if (!stopFlag && !mStopFlag)
        {
            Lock lock(mMutex);
            mCondition.timed_wait(lock, queueTimeoutMs);
        }            

        return stopFlag || mStopFlag;
    }

    PingBackService::Entry PingBackService::registerSession(RcfSessionPtr rcfSessionPtr)
    {
        Lock lock(mMutex);

        // First ping back is sent after 1 second, after that the requested ping 
        // back interval is used.
        boost::uint32_t nextFireMs = Platform::OS::getCurrentTimeMs() + 1000;
        
        Entry entry(nextFireMs, rcfSessionPtr);
        mTimerHeap.add(entry);
        mCondition.notify_all();
        return entry;
    }

    void PingBackService::unregisterSession(const Entry & entry)
    {
        mTimerHeap.remove(entry);
    }

}
