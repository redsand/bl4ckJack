
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

#ifndef INCLUDE_RCF_TEST_WITHSTOPSERVER_HPP
#define INCLUDE_RCF_TEST_WITHSTOPSERVER_HPP

#include <boost/bind.hpp>

#include <RCF/ThreadLibrary.hpp>

namespace RCF {

    class WithStopServer
    {
    public:
        WithStopServer() : mStop()
        {
        }

        void stopServer()
        {
            RCF::Lock lock(mMutex);
            mStop = true;
            mCondition.notify_one();
        }

        void wait()
        {
            RCF::Lock lock(mMutex);
            if (!mStop)
            {
                mCondition.wait(lock);
            }
        }

    private:

        RCF::Mutex mMutex;
        RCF::Condition mCondition;
        bool mStop;

    };

} // namespace RCF

#endif // ! INCLUDE_RCF_TEST_WITHSTOPSERVER_HPP

