
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

#ifndef INCLUDE_RCF_THREADMANAGER_HPP
#define INCLUDE_RCF_THREADMANAGER_HPP

#include <vector>

#include <boost/bind.hpp>
#include <boost/cstdint.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <RCF/Export.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/Timer.hpp>
#include <RCF/Tools.hpp>

namespace boost {
    namespace asio {
        class io_service;
    }
}

namespace RCF {

    class                                                               RcfServer;
    typedef boost::function3<bool, int, const volatile bool &, bool>    Task;
    class                                                               TaskEntry;
    typedef boost::function0<void>                                      StopFunctor;

    typedef unsigned int                                ThreadId;

    class                                               ThreadPool;
    typedef boost::shared_ptr<ThreadPool>               ThreadPoolPtr;
    class                                               Iocp;

    class                                               AsioMuxer;
    typedef boost::asio::io_service                     AsioIoService;

    typedef boost::shared_ptr<ThreadPool>               ThreadPoolPtr;
    class                                               ShouldStop;

    class ThreadInfo
    {
    public:
        ThreadInfo(ThreadPool & threadPool);
        void touch();
        void notifyBusy();

    private:
        friend class ThreadPool;
        friend class ShouldStop;

        ThreadPool &    mThreadPool;
        bool            mBusy;
        bool            mStopFlag;
        RCF::Timer      mTouchTimer;
    };

    typedef boost::shared_ptr<ThreadInfo> ThreadInfoPtr;

    enum MuxerType
    {
        Mt_None,
        Mt_Iocp,
        Mt_Asio
    };

#ifdef RCF_USE_BOOST_ASIO
    static const MuxerType DefaultMuxerType = Mt_Asio;
#else
    static const MuxerType DefaultMuxerType = Mt_Iocp;
#endif

    class RCF_EXPORT ThreadPool : 
        public boost::enable_shared_from_this<ThreadPool>
    {
    public:

        typedef boost::function0<void> ThreadInitFunctor;
        typedef boost::function0<void> ThreadDeinitFunctor;

        ThreadPool(
            std::size_t         threadCount,
            const std::string & threadName = "");

        ThreadPool(
            std::size_t         threadTargetCount,
            std::size_t         threadMaxCount,
            const std::string & threadName = "",
            boost::uint32_t     threadIdleTimeoutMs = 30*1000,
            bool                reserveLastThread = true);

        ~ThreadPool();
        
        void            start(const volatile bool &stopFlag);
        void            stop(bool wait = true);
        bool            isStarted();

        void            addThreadInitFunctor(
                            ThreadInitFunctor threadInitFunctor);

        void            addThreadDeinitFunctor(
                            ThreadDeinitFunctor threadDeinitFunctor);

        void            setThreadName(const std::string &threadName);
        std::string     getThreadName();

        Iocp *          getIocp();
        AsioIoService * getIoService();

        void            enableMuxerType(MuxerType muxerType);
        void            resetMuxers();

        std::size_t     getThreadCount();

        bool            launchThread(const volatile bool &userStopFlag);
        void            notifyBusy();
        void            notifyReady();

        void            repeatTask(
                            RCF::ThreadInfoPtr threadInfoPtr,
                            int timeoutMs,
                            const volatile bool &stopFlag);

        void            setTask(Task task);
        void            setStopFunctor(StopFunctor stopFunctor);

    private:

        void            onInit();
        void            onDeinit();
        void            setMyThreadName();
        

        void            cycle(int timeoutMs, ShouldStop & shouldStop);

        friend class                        TaskEntry;
        friend class                        RcfServer;

        Mutex                               mInitDeinitMutex;
        std::vector<ThreadInitFunctor>      mThreadInitFunctors;
        std::vector<ThreadDeinitFunctor>    mThreadDeinitFunctors;
        std::string                         mThreadName;
        boost::shared_ptr<Iocp>             mIocpPtr;
        boost::shared_ptr<AsioMuxer>        mAsioMuxerPtr;

        bool                                mStarted;
        std::size_t                         mThreadTargetCount;
        std::size_t                         mThreadMaxCount;
        bool                                mReserveLastThread;
        boost::uint32_t                     mThreadIdleTimeoutMs;

        Task                                mTask;
        StopFunctor                         mStopFunctor;

        const volatile bool *               mpUserStopFlag;

        typedef std::map<ThreadInfoPtr, ThreadPtr> ThreadMap;

        Mutex                               mThreadsMutex;
        ThreadMap                           mThreads;
        std::size_t                         mBusyCount;
    };    

    class ThreadTouchGuard
    {
    public:
        ThreadTouchGuard();
        ~ThreadTouchGuard();
    private:
        ThreadInfoPtr mThreadInfoPtr;
    };

    class ShouldStop
    {
    public:

        ShouldStop(
            const volatile bool & stopFlag, 
            ThreadInfoPtr threadInfoPtr);

        bool operator()() const;

    private:
        friend class ThreadPool;

        const volatile bool & mStopFlag;
        volatile bool mTaskFlag;
        ThreadInfoPtr mThreadInfoPtr;
    };

    RCF_EXPORT void setWin32ThreadName(boost::uint32_t dwThreadID, const char * szThreadName);

} // namespace RCF

#endif // ! INCLUDE_RCF_THREADMANAGER_HPP
