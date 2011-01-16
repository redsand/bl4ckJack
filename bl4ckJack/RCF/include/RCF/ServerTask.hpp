
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

#ifndef INCLUDE_RCF_SERVERTASK_HPP
#define INCLUDE_RCF_SERVERTASK_HPP

#include <RCF/Export.hpp>
#include <RCF/ThreadPool.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    class RcfServer;

    class RCF_EXPORT TaskEntry
    {
    public:
        TaskEntry(
            Task                    task,
            StopFunctor             stopFunctor,
            const std::string &     threadName = "");

        TaskEntry(
            MuxerType               muxerType);

        ThreadPool &   
                getThreadPool();

        void    setThreadPoolPtr(ThreadPoolPtr threadPoolPtr);
        Task    getTask();
        void    start(const volatile bool &stopFlag);
        void    stop(bool wait = true);

        void    resetMuxers();

    private:

        friend class    RcfServer;

        MuxerType       mMuxerType;

        Task            mTask;
        StopFunctor     mStopFunctor;
        std::string     mThreadName;

        ThreadPoolPtr   mLocalThreadPoolPtr;

        ThreadPoolPtr   mWhichThreadPoolPtr;
    };

    typedef std::vector<TaskEntry> TaskEntries;

} // namespace RCF

#endif // ! INCLUDE_RCF_SERVERTASK_HPP
