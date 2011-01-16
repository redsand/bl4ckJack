
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

#include <RCF/RcfSession.hpp>

#include <RCF/ClientTransport.hpp>
#include <RCF/Marshal.hpp>
#include <RCF/PerformanceData.hpp>
#include <RCF/PingBackService.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/SessionTimeoutService.hpp>
#include <RCF/ThreadLocalCache.hpp>
#include <RCF/Version.hpp>

#include <boost/bind.hpp>

namespace RCF {

    RcfSession::RcfSession(RcfServer &server) :
        mStopCallInProgress(RCF_DEFAULT_INIT),
        mRcfServer(server),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion()),
        mUseNativeWstringSerialization(RCF::getDefaultNativeWstringSerialization()),
        mTransportFiltersLocked(RCF_DEFAULT_INIT),
        mFiltered(RCF_DEFAULT_INIT),
        mCloseSessionAfterWrite(RCF_DEFAULT_INIT),
        mPingTimestamp(RCF_DEFAULT_INIT),
        mPingIntervalMs(RCF_DEFAULT_INIT),
        mTouchTimestamp(Platform::OS::getCurrentTimeMs()),
        mWritingPingBack(false),
        mpParameters(RCF_DEFAULT_INIT),
        mParmsVec(1+15), // return value + max 15 arguments
        mAutoSend(true),
        mpSessionState(NULL)
    {
        Lock lock(getPerformanceData().mMutex);
        ++getPerformanceData().mRcfSessions;
    }

    RcfSession::~RcfSession()
    {
        RCF_DTOR_BEGIN

            {
                Lock lock(getPerformanceData().mMutex);
                --getPerformanceData().mRcfSessions;
            }

            mRcfServer.unregisterSession(mWeakThisPtr);

            // no locks here, relying on dtor thread safety of reference counted objects
            clearParameters();
            if (mOnDestroyCallback)
            {
                mOnDestroyCallback(*this);
            }
        RCF_DTOR_END
    }

    I_SessionState & RcfSession::getSessionState() const
    {
        return *mpSessionState;
    }

    void RcfSession::setSessionState(I_SessionState & sessionState)
    {
        mpSessionState = &sessionState;
    }

    void RcfSession::clearParameters()
    {
        if (mpParameters)
        {
            //mpParameters->~I_Parameters();

            // need to be elaborate here, for borland compiler
            typedef I_Parameters P;
            P &p = *mpParameters;
            p.~P();
            mpParameters = NULL;
        }

    }

    void RcfSession::setOnDestroyCallback(OnDestroyCallback onDestroyCallback)
    {
        Lock lock(mMutex);
        mOnDestroyCallback = onDestroyCallback;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void RcfSession::enableSfSerializationPointerTracking(bool enable)
    {
        mOut.mOutProtocol1.setCustomizationCallback(
            boost::bind(enableSfPointerTracking_1, _1, enable) );

        //mOut.mOutProtocol2.setCustomizationCallback(
        //    boost::bind(enableSfPointerTracking_2, _1, enable) );
    }

#else

    void RcfSession::enableSfSerializationPointerTracking(bool enable)
    {}

#endif

    void RcfSession::addOnWriteCompletedCallback(
        const OnWriteCompletedCallback &onWriteCompletedCallback)
    {
        Lock lock(mMutex);
        mOnWriteCompletedCallbacks.push_back(onWriteCompletedCallback);
    }

    void RcfSession::extractOnWriteCompletedCallbacks(
        std::vector<OnWriteCompletedCallback> &onWriteCompletedCallbacks)
    {
        Lock lock(mMutex);
        onWriteCompletedCallbacks.clear();
        onWriteCompletedCallbacks.swap( mOnWriteCompletedCallbacks );
    }

    const RCF::I_RemoteAddress &RcfSession::getRemoteAddress()
    {
        return getSessionState().getRemoteAddress();
    }

    void RcfSession::disconnect()
    {
        SessionStatePtr sessionStatePtr = getSessionState().shared_from_this();
        sessionStatePtr->setEnableReconnect(false);
        sessionStatePtr->postClose();
    }

    bool RcfSession::hasDefaultServerStub()
    {
        Lock lock(mMutex);
        return mDefaultStubEntryPtr;
    }

    StubEntryPtr RcfSession::getDefaultStubEntryPtr()
    {
        Lock lock(mMutex);
        return mDefaultStubEntryPtr;
    }

    void RcfSession::setDefaultStubEntryPtr(StubEntryPtr stubEntryPtr)
    {
        Lock lock(mMutex);
        mDefaultStubEntryPtr = stubEntryPtr;
    }

    void RcfSession::setCachedStubEntryPtr(StubEntryPtr stubEntryPtr)
    {
        mCachedStubEntryPtr = stubEntryPtr;
    }

    void RcfSession::getMessageFilters(std::vector<FilterPtr> &filters)
    {
        filters = mFilters;
    }

    void RcfSession::getTransportFilters(std::vector<FilterPtr> &filters)
    {
        getSessionState().getTransportFilters(filters);
    }

    boost::uint32_t RcfSession::getRuntimeVersion()
    {
        return mRuntimeVersion;
    }

    void RcfSession::setRuntimeVersion(boost::uint32_t version)
    {
        mRuntimeVersion = version;
    }

    boost::uint32_t RcfSession::getArchiveVersion()
    {
        return mArchiveVersion;
    }

    void RcfSession::setArchiveVersion(boost::uint32_t version)
    {
        mArchiveVersion = version;
    }

    bool RcfSession::getNativeWstringSerialization()    
    {
        return mUseNativeWstringSerialization;
    }

    void RcfSession::setNativeWstringSerialization(bool useNativeWstringSerialization)
    {
        mUseNativeWstringSerialization = useNativeWstringSerialization;
    }

    void RcfSession::setUserData(boost::any userData)
    {
        mUserData = userData;
    }

    boost::any RcfSession::getUserData()
    {
        return mUserData;
    }

    void RcfSession::lockTransportFilters()
    {
        mTransportFiltersLocked = true;
    }

    void RcfSession::unlockTransportFilters()
    {
        mTransportFiltersLocked = false;
    }

    bool RcfSession::transportFiltersLocked()
    {
        return mTransportFiltersLocked;
    }

    SerializationProtocolIn & RcfSession::getSpIn()
    {
        return mIn;
    }

    SerializationProtocolOut & RcfSession::getSpOut()
    {
        return mOut;
    }

    bool RcfSession::getFiltered()
    {
        return mFiltered;
    }

    void RcfSession::setFiltered(bool filtered)
    {
        mFiltered = filtered;
    }

    std::vector<FilterPtr> & RcfSession::getFilters()
    {
        return mFilters;
    }

    RcfServer & RcfSession::getRcfServer()
    {
        return mRcfServer;
    }

    void RcfSession::setCloseSessionAfterWrite(bool close)
    {
        mCloseSessionAfterWrite = close;
    }

    boost::uint32_t RcfSession::getPingBackIntervalMs()
    {
        return mRequest.getPingBackIntervalMs();
    }

    boost::uint32_t RcfSession::getPingTimestamp()
    {
        Lock lock(mMutex);
        return mPingTimestamp;
    }

    void RcfSession::setPingTimestamp()
    {
        Lock lock(mMutex);
        mPingTimestamp = Platform::OS::getCurrentTimeMs();
    }

    boost::uint32_t RcfSession::getPingIntervalMs()
    {
        return mPingIntervalMs;
    }

    void RcfSession::setPingIntervalMs(boost::uint32_t pingIntervalMs)
    {
        mPingIntervalMs = pingIntervalMs;
    }

    boost::uint32_t RcfSession::getTouchTimestamp()
    {
        Lock lock(mMutex);
        return mTouchTimestamp;
    }

    void RcfSession::touch()
    {
        Lock lock(mMutex);
        mTouchTimestamp = Platform::OS::getCurrentTimeMs();
    }

    void RcfSession::registerForPingBacks()
    {
        // Register for ping backs if appropriate.

        if (    mRequest.getPingBackIntervalMs() > 0 
            &&  !mRequest.getOneway())
        {
            PingBackServicePtr pbsPtr = mRcfServer.getPingBackServicePtr();
            if (pbsPtr)
            {
                // Disable reconnecting for this session. After sending a 
                // pingback, a server I/O thread would get a write completion 
                // notification, and if it happened to be an error (very unlikely 
                // but possible), we definitely would not want a reconnect, as 
                // the session would still in use.
                getSessionState().setEnableReconnect(false);

                PingBackTimerEntry pingBackTimerEntry = 
                    pbsPtr->registerSession(shared_from_this());

                Lock lock(mIoStateMutex);
                RCF_ASSERT_EQ( mPingBackTimerEntry.first , 0 );
                mPingBackTimerEntry = pingBackTimerEntry;
            }
            else
            {
                // TODO: something more efficient than throwing
                Exception e(_RcfError_NoPingBackService());
                RCF_THROW(e);
            }
        }
    }

    void RcfSession::unregisterForPingBacks()
    {
        // Unregister for ping backs if appropriate.

        if (    mRequest.getPingBackIntervalMs() > 0 
            &&  !mRequest.getOneway())
        {
            PingBackServicePtr pbsPtr = mRcfServer.getPingBackServicePtr();
            if (pbsPtr)
            {
                pbsPtr->unregisterSession(mPingBackTimerEntry);
                mPingBackTimerEntry = PingBackTimerEntry();
            }
        }
    }

    void RcfSession::sendPingBack()
    {
        //RCF_ASSERT( mIoStateMutex.locked() );

        mWritingPingBack = true;

        ThreadLocalCached< std::vector<ByteBuffer> > tlcByteBuffers;
        std::vector<ByteBuffer> &byteBuffers = tlcByteBuffers.get();

        byteBuffers.push_back(mPingBackByteBuffer);

        boost::uint32_t pingBackIntervalMs = getPingBackIntervalMs();

        encodeServerError(
            mRcfServer,
            byteBuffers.front(),
            RcfError_PingBack,
            pingBackIntervalMs,
            0);

        getSessionState().postWrite(byteBuffers);
    }

    bool RcfSession::getAutoSend()
    {
        return mAutoSend;
    }

    void RcfSession::setWeakThisPtr()
    {
        mWeakThisPtr = shared_from_this();
    }

    void RcfSession::setRequestUserData(const std::string & userData)
    {
        mRequest.mRequestUserData = ByteBuffer(userData);
    }

    std::string RcfSession::getRequestUserData()
    {
        if ( mRequest.mRequestUserData.isEmpty() )
        {
            return std::string();
        }

        return std::string(
            mRequest.mRequestUserData.getPtr(), 
            mRequest.mRequestUserData.getLength());
    }

    void RcfSession::setResponseUserData(const std::string & userData)
    {
        mRequest.mResponseUserData = ByteBuffer(userData);
    }

    std::string RcfSession::getResponseUserData()
    {
        if ( mRequest.mResponseUserData.isEmpty() )
        {
            return std::string();
        }

        return std::string(
            mRequest.mResponseUserData.getPtr(), 
            mRequest.mResponseUserData.getLength());    
    }

#ifdef RCF_USE_BOOST_FILESYSTEM

    void RcfSession::cancelDownload()
    {
        Lock lock(mMutex);
        if (mDownloadInfoPtr)
        {
            mDownloadInfoPtr->mCancel = true;
        }
    }

    void RcfSession::addDownloadStream(
        boost::uint32_t sessionLocalId, 
        FileStream fileStream)
    {
        Lock lock(mMutex);
        mSessionDownloads[sessionLocalId].mImplPtr = fileStream.mImplPtr;
    }

#else

    void RcfSession::cancelDownload()
    {

    }

#endif

} // namespace RCF
