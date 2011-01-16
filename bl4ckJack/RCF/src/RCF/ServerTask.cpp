
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

#include <RCF/ServerTask.hpp>

namespace RCF {

    TaskEntry::TaskEntry(
        Task                    task,
        StopFunctor             stopFunctor,
        const std::string &     threadName) :
            mMuxerType(Mt_None),
            mTask(task),
            mStopFunctor(stopFunctor),
            mThreadName(threadName)
    {
    }

    TaskEntry::TaskEntry(MuxerType muxerType) : mMuxerType(muxerType)
    {
    }

    ThreadPool & TaskEntry::getThreadPool()
    {
        return *mWhichThreadPoolPtr;
    }

    void TaskEntry::setThreadPoolPtr(ThreadPoolPtr threadPoolPtr)
    {
        mLocalThreadPoolPtr = threadPoolPtr;
        threadPoolPtr->setTask(mTask);
        threadPoolPtr->setStopFunctor(mStopFunctor);
    }

    Task TaskEntry::getTask()
    {
        return mTask;
    }

    void TaskEntry::start(const volatile bool &stopFlag)
    {
        mWhichThreadPoolPtr->start(stopFlag);
    }

    void TaskEntry::stop(bool wait)
    {
        if (mWhichThreadPoolPtr)
        {
            mWhichThreadPoolPtr->stop(wait);
        }
    }

    void TaskEntry::resetMuxers()
    {
        if (mLocalThreadPoolPtr)
        {
            mWhichThreadPoolPtr.reset();
            mLocalThreadPoolPtr->resetMuxers();
        }
    }

} // namespace RCF
