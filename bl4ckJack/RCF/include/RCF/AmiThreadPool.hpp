
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

#ifndef INCLUDE_RCF_AMITHREADPOOL_HPP
#define INCLUDE_RCF_AMITHREADPOOL_HPP

#include <memory>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>

#include <RCF/Heap.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    class AmiNotification
    {
    public:

        typedef boost::function0<void> Cb;

        // Need mutexPtr so that the mutex doesn't die before the lock.
        void set(Cb cb, std::auto_ptr<Lock> lockPtr, boost::shared_ptr<Mutex> mutexPtr)
        {
            mCb = cb;
            mLockPtr = lockPtr;
            mMutexPtr = mutexPtr;
        }

        void run()
        {
            mLockPtr.reset();
            if (mCb)
            {
                Cb cb = mCb;
                mCb = Cb();
                cb();
            }
        }

        void clear()
        {
            mLockPtr.reset();
            mMutexPtr.reset();
            mCb = Cb();
        }

    private:
        Cb mCb;
        boost::shared_ptr<Mutex> mMutexPtr;
        std::auto_ptr<Lock> mLockPtr;
    };

} // namespace RCF

#ifdef BOOST_WINDOWS

#include <RCF/Iocp.hpp>

namespace RCF {

    class OverlappedAmi;
    typedef boost::shared_ptr<OverlappedAmi> OverlappedAmiPtr;

    class I_OverlappedAmi : public IocpOverlapped
    {
    public:
        virtual void onCompletion(
            BOOL ret, 
            DWORD dwErr,
            ULONG_PTR completionKey, 
            DWORD dwNumBytes)
        {
            RCF_UNUSED_VARIABLE(ret);
            RCF_UNUSED_VARIABLE(dwErr);
            RCF_UNUSED_VARIABLE(completionKey);

            onCompletion(dwNumBytes);
        }

        virtual void onCompletion(std::size_t numBytes) = 0;

        virtual void onError(const RCF::Exception & e) = 0;

        typedef std::pair<boost::uint32_t, OverlappedAmiPtr> TimerEntry;

        virtual void onTimerExpired(const TimerEntry & timerEntry) = 0;
    };

    class RCF_EXPORT AmiThreadPool
    {
    public:
        AmiThreadPool();
        ~AmiThreadPool();

        static void start(std::size_t threadCount);
        static void stop();

        typedef std::pair<boost::uint32_t, OverlappedAmiPtr> TimerEntry;

        TimerEntry addTimerEntry(
            OverlappedAmiPtr overlappedAmiptr, 
            boost::uint32_t timeoutMs);

        void removeTimerEntry(
            const TimerEntry & timerEntry);

        // TODO: make private (currently used to associate sockets with the iocp)
        Iocp                            mIocp;

        Exception connect(
            int fd, 
            sockaddr & addr, 
            Platform::OS::BsdSockets::socklen_t addrSize, 
            OverlappedAmiPtr overlappedAmiPtr);

        void cancelConnect(int fd);
          
    private:

        void cycleIo();
        void cycleTimer();
        void cycleConnect();

        Exception addConnectFd(int fd, OverlappedAmiPtr overlappedAmiPtr);
        void removeConnectFd(int fd);


        
        RCF::ThreadPtr                  mIoThreadPtr;
        RCF::ThreadPtr                  mTimerThreadPtr;
        RCF::ThreadPtr                  mConnectThreadPtr;
        bool                            mStopFlag;

        Mutex                           mTimerMutex;
        Condition                       mTimerCondition;

        TimerHeap<OverlappedAmiPtr>     mTimerHeap;

        typedef std::map<int,OverlappedAmiPtr> Fds;

        Mutex                           mConnectFdsMutex;
        Fds                             mConnectFds;
    };

    
    RCF_EXPORT void amiInit();
    RCF_EXPORT void amiDeinit();

    class RCF_EXPORT AmiInitDeinit
    {
    public:
        AmiInitDeinit()
        {
            amiInit();
        }
        ~AmiInitDeinit()
        {
            amiDeinit();
        }
    };

} // namespace RCF

#else // BOOST_WINDOWS

namespace RCF {

    // TODO: Non-Windows implementation.

    class OverlappedAmi;
    typedef boost::shared_ptr<OverlappedAmi> OverlappedAmiPtr;

    class I_OverlappedAmi
    {
    public:
        I_OverlappedAmi() {}
        virtual ~I_OverlappedAmi() {}
        virtual void onCompletion(std::size_t numBytes) = 0;

        typedef std::pair<boost::uint32_t, OverlappedAmiPtr> TimerEntry;

        virtual void onTimerExpired(const TimerEntry & timerEntry) = 0;
    };

    class AmiThreadPool
    {
    public:
        void cancelConnect(int fd)
        {
        }
    };


} // namespace RCF

#endif // !BOOST_WINDOWS

namespace RCF {

    extern boost::scoped_ptr<AmiThreadPool> gAmiThreadPoolPtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_AMITHREADPOOL_HPP
