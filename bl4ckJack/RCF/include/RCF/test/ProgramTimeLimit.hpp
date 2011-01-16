
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

#ifndef INCLUDE_RCF_TEST_PROGRAMTIMELIMIT_HPP
#define INCLUDE_RCF_TEST_PROGRAMTIMELIMIT_HPP

#include <iostream>
#include <boost/bind.hpp>
#include <boost/config.hpp>

#include <RCF/ThreadLibrary.hpp>
#include <RCF/ThreadPool.hpp>
#include <RCF/util/InitDeinit.hpp>
#include <RCF/util/Platform/OS/GetCurrentTime.hpp>
#include <RCF/util/CommandLine.hpp>

class ProgramTimeLimit
{
public:
    ProgramTimeLimit(unsigned int timeLimitS)
    {
        mStartTimeMs = Platform::OS::getCurrentTimeMs();
        mTimeLimitMs = timeLimitS*1000;
        mStopFlag = false;
        if (timeLimitS)
        {
            mThreadPtr.reset( new RCF::Thread( boost::bind(&ProgramTimeLimit::poll, this)));
        }
    }

    ~ProgramTimeLimit()
    {
        if (mThreadPtr)
        {
            {
                RCF::Lock lock(mStopMutex);
                mStopFlag = true;
                mStopCondition.notify_all();
            }
            mThreadPtr->join();
        }
    }

private:

    void poll()
    {
        // Set our thread name.
        RCF::setWin32ThreadName( static_cast<boost::uint32_t>(-1), "RCF Program Time Limit");

        while (true)
        {
            unsigned int pollIntervalMs = 1000;
            RCF::Lock lock(mStopMutex);
            mStopCondition.timed_wait(lock, pollIntervalMs);
            if (mStopFlag)
            {
                break;
            }
            else
            {
                unsigned int currentTimeMs = Platform::OS::getCurrentTimeMs();
                if (currentTimeMs - mStartTimeMs > mTimeLimitMs)
                {
                    std::cout 
                        << "Time limit expired (" << mTimeLimitMs/1000 << " (s) ). Killing this test." 
                        << std::endl;

#if defined(_MSC_VER) && _MSC_VER >= 1310

                    // By simulating an access violation , we will trigger the 
                    // creation  of a minidump, which will aid postmortem analysis.

                    int * pn = 0;
                    *pn = 1;

#elif defined(BOOST_WINDOWS)
                    
                    TerminateProcess(GetCurrentProcess(), 1);

#else

                    abort();

#endif
                }
            }
        }
    }

    unsigned int mStartTimeMs;
    unsigned int mTimeLimitMs;
    RCF::ThreadPtr mThreadPtr;
    bool mStopFlag;

    RCF::Mutex mStopMutex;
    RCF::Condition mStopCondition;
};

ProgramTimeLimit * gpProgramTimeLimit;

UTIL_ON_DEINIT_NAMED( delete gpProgramTimeLimit; gpProgramTimeLimit = NULL; , PtoCloDeinitialize )

#endif // ! INCLUDE_RCF_TEST_PROGRAMTIMELIMIT_HPP
