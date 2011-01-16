
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

#include <RCF/ThreadLocalData.hpp>
#include <RCF/ThreadLocalCache.hpp>

#include <RCF/AmiThreadPool.hpp>
#include <RCF/ByteBuffer.hpp>
#include <RCF/InitDeinit.hpp>

namespace RCF {

    class ThreadLocalData
    {
    public:
        ThreadLocalData()
        {
            clear();
        }

        ObjectCache                     mObjectCache;
        std::vector<ClientStub *>       mCurrentClientStubs;
        RcfSession *                    mpCurrentRcfSession;
        ThreadInfoPtr                   mThreadInfoPtr;
        UdpSessionStatePtr              mUdpSessionStatePtr;
        RecursionState<int, int>        mRcfSessionRecursionState;
        AmiNotification                 mAmiNotification;

        void clear()
        {
            mObjectCache.clear();
            mCurrentClientStubs.clear();
            mpCurrentRcfSession = NULL;
            mThreadInfoPtr.reset();
            mUdpSessionStatePtr.reset();
            mRcfSessionRecursionState = RecursionState<int, int>();
            mAmiNotification.clear();
        }
    };

    typedef ThreadSpecificPtr<ThreadLocalData>::Val ThreadLocalDataPtr;

    ThreadLocalDataPtr *pThreadLocalDataPtr = NULL;

    ThreadLocalData &getThreadLocalData()
    {
        if (NULL == pThreadLocalDataPtr->get())
        {
            pThreadLocalDataPtr->reset( new ThreadLocalData());
        }
        return *(*pThreadLocalDataPtr);
    }

    // Solaris 10 on x86 crashes when trying to delete the thread specific pointer
#if defined(sun) || defined(__sun) || defined(__sun__)

    RCF_ON_INIT_NAMED(if (!pThreadLocalDataPtr) pThreadLocalDataPtr = new ThreadLocalDataPtr; , ThreadLocalDataInit)
    //RCF_ON_DEINIT_NAMED( (*pThreadLocalDataPtr)->clear(); , ThreadLocalDataDeinit)

#else

    RCF_ON_INIT_NAMED(pThreadLocalDataPtr = new ThreadLocalDataPtr;, ThreadLocalDataInit)
    RCF_ON_DEINIT_NAMED( delete pThreadLocalDataPtr; pThreadLocalDataPtr = NULL; , ThreadLocalDataDeinit)

#endif

    // access to the various thread local entities

    ObjectCache &getThreadLocalObjectCache()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mObjectCache;
    }

    ClientStub * getCurrentClientStubPtr()
    {
        ThreadLocalData & tld = getThreadLocalData();
        if (!tld.mCurrentClientStubs.empty())
        {
            return tld.mCurrentClientStubs.back();
        }
        return NULL;
    }

    void pushCurrentClientStub(ClientStub * pClientStub)
    {
        ThreadLocalData & tld = getThreadLocalData();
        tld.mCurrentClientStubs.push_back(pClientStub);
    }

    void popCurrentClientStub()
    {
        ThreadLocalData & tld = getThreadLocalData();
        tld.mCurrentClientStubs.pop_back();
    }

    RcfSession * getCurrentRcfSessionPtr()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mpCurrentRcfSession;
    }

    void setCurrentRcfSessionPtr(RcfSession * pRcfSessionPtr)
    {
        ThreadLocalData & tld = getThreadLocalData();
        tld.mpCurrentRcfSession = pRcfSessionPtr;
    }

    ThreadInfoPtr getThreadInfoPtr()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mThreadInfoPtr;
    }

    void setThreadInfoPtr(ThreadInfoPtr threadInfoPtr)
    {
        ThreadLocalData & tld = getThreadLocalData();
        tld.mThreadInfoPtr = threadInfoPtr;
    }

    UdpSessionStatePtr getCurrentUdpSessionStatePtr()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mUdpSessionStatePtr;
    }

    void setCurrentUdpSessionStatePtr(UdpSessionStatePtr udpSessionStatePtr)
    {
        ThreadLocalData & tld = getThreadLocalData();
        tld.mUdpSessionStatePtr = udpSessionStatePtr;
    }

    RcfSession & getCurrentRcfSession()
    {
        return *getCurrentRcfSessionPtr();
    }

    RecursionState<int, int> & getCurrentRcfSessionRecursionState()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mRcfSessionRecursionState;
    }

    AmiNotification & getCurrentAmiNotification()
    {
        ThreadLocalData & tld = getThreadLocalData();
        return tld.mAmiNotification;
    }

} // namespace RCF
