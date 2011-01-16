
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

#include <RCF/Marshal.hpp>

#include <algorithm>

#include <boost/function.hpp>

#include <RCF/AmiThreadPool.hpp>
#include <RCF/ClientProgress.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/ThreadLocalCache.hpp>
#include <RCF/ThreadLocalData.hpp>

namespace RCF {

    bool serializeOverride(SerializationProtocolOut &out, ByteBuffer & u)
    {
        int runtimeVersion = out.getRuntimeVersion();

        if (runtimeVersion <= 3)
        {
            // Legacy behavior - no metadata for ByteBuffer.
            int len = static_cast<int>(u.getLength());
            serialize(out, len);
            out.insert(u);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool serializeOverride(SerializationProtocolOut &out, ByteBuffer * pu)
    {
        RCF_ASSERT(pu);
        return serializeOverride(out, *pu);
    }

    bool deserializeOverride(SerializationProtocolIn &in, ByteBuffer & u)
    {
        int runtimeVersion = in.getRuntimeVersion();

        if (runtimeVersion <= 3)
        {
            // Legacy behavior - no metadata for ByteBuffer.
            int len = 0;
            deserialize(in, len);
            in.extractSlice(u, len);
            return true;
        }
        else
        {
            return false;
        }
    }

    Mutex * gpCandidatesMutex = NULL;
    Candidates * gpCandidates = NULL;

    Mutex & gCandidatesMutex()
    {
        return *gpCandidatesMutex; 
    }

    Candidates & gCandidates()
    {
        return *gpCandidates;
    }

    RCF_ON_INIT_DEINIT_NAMED( 
        gpCandidatesMutex = new Mutex; gpCandidates = new Candidates; ,
        delete gpCandidatesMutex; gpCandidatesMutex = NULL; delete gpCandidates; gpCandidates = NULL,
        InitDeinitCandidates)

    void ClientStub::enrol(I_Future *pFuture)
    {
        mFutures.push_back(pFuture);
        pFuture->setClientStub(this);
    }

    void ClientStub::init( 
        const std::string &subInterface, 
        int fnId, 
        RCF::RemoteCallSemantics rcs)
    {        
        mRequest.init(
            getTargetToken(),
            getTargetName(),
            subInterface,
            fnId,
            getSerializationProtocol(),
            mMarshalingProtocol,
            (rcs == RCF::Oneway),
            false,
            getRuntimeVersion(),
            false,
            mPingBackIntervalMs,
            mArchiveVersion,
            mUseNativeWstringSerialization);

        mOut.reset(
            getSerializationProtocol(),
            32,
            mRequest.encodeRequestHeader(),
            mRuntimeVersion,
            mArchiveVersion);

        bool asyncParameters = false;
        {
            ::RCF::CurrentClientStubSentry sentry(*this);
            mpParameters->write(mOut);

            mFutures.clear();
            asyncParameters = mpParameters->enrolFutures(this);
        }
        if (asyncParameters)
        {
            setAsync(true);
        }
    }

    void ClientStub::connect()
    {
        CurrentClientStubSentry sentry(*this);
        instantiateTransport();
        if (    !mConnected 
            ||    (    mConnected 
                &&    mAutoReconnect 
                &&    mRcs == Twoway 
                &&    !mTransport->isConnected()))
        {
            std::string endpoint;
            if (mEndpoint.get())
            {
                endpoint = mEndpoint->asString();
            }

            RCF_LOG_2()(this)(endpoint)(mConnectTimeoutMs) 
                << "RcfClient - connect to server.";

            mTransport->disconnect(mConnectTimeoutMs);

            if (mAsync)
            {
                mAsyncTimerReason = Connect;
                mAsyncTimerEntry = mTransport->setTimer(mConnectTimeoutMs, this);
            }

            mTransport->connect(*this, mConnectTimeoutMs);            
        }
        else
        {
            onConnectCompleted(true);
        }
    }

    void ClientStub::connectAsync(boost::function0<void> onCompletion)
    {
        setAsync(true);
        instantiateTransport();
        mTransport->setAsync(mAsync);
        setAsyncCallback(onCompletion);
        connect();
    }

    void ClientStub::waitAsync(
        boost::function0<void> onCompletion, 
        boost::uint32_t timeoutMs)
    {
        setAsync(true);
        instantiateTransport();
        mTransport->setAsync(mAsync);
        setAsyncCallback(onCompletion);

        RCF_ASSERT_EQ(mAsyncTimerReason , None);
        mAsyncTimerReason = Wait;
        mAsyncTimerEntry = mTransport->setTimer(timeoutMs, this);
    }

    void ClientStub::onConnectCompleted(bool alreadyConnected)
    {
        if (!alreadyConnected)
        {
            if (mAsync)
            {
                if (mAsyncTimerEntry != TimerEntry())
                {
                    mTransport->killTimer(mAsyncTimerEntry);
                    mAsyncTimerEntry = TimerEntry();
                }
                mAsyncTimerReason = None;
            }

            std::vector<FilterPtr> filters;
            mTransport->getTransportFilters(filters);

            std::for_each(
                filters.begin(),
                filters.end(),
                boost::bind(&Filter::reset, _1));

            mTransport->setTransportFilters(std::vector<FilterPtr>());
            if (!filters.empty())
            {
                requestTransportFilters(filters);
            }
            mConnected = true;

            if (
                mClientProgressPtr.get() &&
                (mClientProgressPtr->mTriggerMask & ClientProgress::Event))
            {
                ClientProgress::Action action = ClientProgress::Continue;

                mClientProgressPtr->mProgressCallback(
                    0,
                    0,
                    ClientProgress::Event,
                    ClientProgress::Connect,
                    action);

                RCF_VERIFY(
                    action != ClientProgress::Cancel,
                    Exception(_RcfError_ClientCancel()));
            }
        }

        if (!mEncodedByteBuffers.empty())
        {
            unsigned int timeoutMs = generateTimeoutMs(mEndTimeMs);

            if (mAsync)
            {
                mAsyncTimerReason = Write;
                mAsyncTimerEntry = mTransport->setTimer(timeoutMs, this);
            }

            // Prepend the length field.
            BOOST_STATIC_ASSERT(sizeof(unsigned int) == 4);

            unsigned int messageLength = static_cast<unsigned int>(
                lengthByteBuffers(mEncodedByteBuffers));

            ByteBuffer & byteBuffer = mEncodedByteBuffers.front();
            byteBuffer.expandIntoLeftMargin(4);
            memcpy(byteBuffer.getPtr(), &messageLength, 4);
            RCF::machineToNetworkOrder(byteBuffer.getPtr(), 4, 1);

            if (mBatchMode)
            {
                RCF_ASSERT_EQ(mRcs , Oneway);
                RCF_ASSERT(!mAsync);
                RCF_ASSERT(mBatchBufferPtr);

                // Accumulate in the batch buffer.
                std::size_t appendLen = lengthByteBuffers(mEncodedByteBuffers);
                std::size_t currentSize = mBatchBufferPtr->size();

                bool usingTempBuffer = false;

                // If this message will cause us to exceed the limit, then flush first.
                if (    mBatchMaxMessageLength 
                    &&  currentSize + appendLen > mBatchMaxMessageLength)
                {
                    mBatchBufferTemp.resize(appendLen);
                    copyByteBuffers(mEncodedByteBuffers, & mBatchBufferTemp[0] );
                    usingTempBuffer = true;

                    flushBatch(timeoutMs);
                    currentSize = mBatchBufferPtr->size();
                }
                
                mBatchBufferPtr->resize( currentSize + appendLen);

                if (usingTempBuffer)
                {
                    memcpy(
                        & (*mBatchBufferPtr)[currentSize], 
                        &mBatchBufferTemp[0], 
                        mBatchBufferTemp.size());
                }
                else
                {
                    copyByteBuffers(
                        mEncodedByteBuffers, 
                        & (*mBatchBufferPtr)[currentSize] );
                }

                ++mBatchMessageCount;
            }
            else
            {
                int err = getTransport().send(*this, mEncodedByteBuffers, timeoutMs);
                RCF_UNUSED_VARIABLE(err);
            }
        }
        else
        {
            if (mAsync)
            {
                prepareAmiNotification();
            }
        }        
    }

    void ClientStub::send()
    {
        CurrentClientStubSentry sentry(*this);

        unsigned int totalTimeoutMs = getRemoteCallTimeoutMs();
        unsigned int startTimeMs = getCurrentTimeMs();
        mEndTimeMs = startTimeMs + totalTimeoutMs;            

        ThreadLocalCached< std::vector<ByteBuffer> > tlcByteBuffers;
        std::vector<ByteBuffer> &byteBuffers = tlcByteBuffers.get();

        mOut.extractByteBuffers(byteBuffers);
        int protocol = mOut.getSerializationProtocol();
        RCF_UNUSED_VARIABLE(protocol);

        mEncodedByteBuffers.resize(0);

        mRequest.encodeRequest(
            byteBuffers,
            mEncodedByteBuffers,
            getMessageFilters());

        instantiateTransport();

        mTransport->setAsync(mAsync);

        WithProgressCallback *pWithCallbackProgress =
            dynamic_cast<WithProgressCallback *>(&getTransport());

        if (pWithCallbackProgress)
        {
            pWithCallbackProgress->setClientProgressPtr(
                getClientProgressPtr());
        }

        // TODO: make sure timeouts behave as expected, esp. when connect() is 
        // doing round trip filter setup calls
        connect();

    }

    void ClientStub::onSendCompleted()
    {
        mEncodedByteBuffers.resize(0);
        if (mRcs == RCF::Oneway)
        {
            // TODO: Refactor. This code is identical to what happens when a 
            // receive completes, in a twoway call.

            if (mAsync)
            {
                if (mAsyncTimerEntry != TimerEntry())
                {
                    mTransport->killTimer(mAsyncTimerEntry);
                    mAsyncTimerEntry = TimerEntry();
                }            
                mAsyncTimerReason = None;

                prepareAmiNotification();
            }
        }
        else
        {
            receive();
        }
    }

    void ClientStub::receive()
    {
        if (mPingBackIntervalMs && mRuntimeVersion >= 5)
        {
            mPingBackCheckIntervalMs = 3 * mPingBackIntervalMs;

            mNextPingBackCheckMs = 
                Platform::OS::getCurrentTimeMs() + mPingBackCheckIntervalMs;

            // So we avoid the special value 0.
            mNextPingBackCheckMs |= 1;
        }

        if (mAsync)
        {
            mAsyncTimerReason = Read;
        }

        unsigned int timeoutMs = generateTimeoutMs(mEndTimeMs);

        mEncodedByteBuffer.clear();
        int err = getTransport().receive(*this, mEncodedByteBuffer, timeoutMs);
        RCF_UNUSED_VARIABLE(err);
    }

    void ClientStub::onException(const Exception & e)
    {
        if (mAsync)
        {
            onError(e);
        }
        else
        {
            e.throwSelf();
        }
    }

    void ClientStub::onReceiveCompleted()
    {
        if (mAsync)
        {
            if (mAsyncTimerEntry != TimerEntry())
            {
                mTransport->killTimer(mAsyncTimerEntry);
                mAsyncTimerEntry = TimerEntry();
            }            
            mAsyncTimerReason = None;
        }

        ByteBuffer unfilteredByteBuffer;

        MethodInvocationResponse response;

        mRequest.decodeResponse(
            mEncodedByteBuffer,
            unfilteredByteBuffer,
            response,
            getMessageFilters());

        mEncodedByteBuffer.clear();

        mIn.reset(
            unfilteredByteBuffer,
            mOut.getSerializationProtocol(),
            mRuntimeVersion,
            mArchiveVersion);

        RCF_LOG_3()(this)(response) << "RcfClient - received response.";

        if (response.isException())
        {
            std::auto_ptr<RemoteException> remoteExceptionAutoPtr(
                response.getExceptionPtr());

            if (!remoteExceptionAutoPtr.get())
            {
                int runtimeVersion = mRequest.mRuntimeVersion;
                if (runtimeVersion < 8)
                {
                    deserialize(mIn, remoteExceptionAutoPtr);
                }
                else
                {
                    RemoteException * pRe = NULL;
                    deserialize(mIn, pRe);
                    remoteExceptionAutoPtr.reset(pRe);
                }
            }

            onException(*remoteExceptionAutoPtr);
        }
        else if (response.isError())
        {
            int err = response.getError();
            if (err == RcfError_VersionMismatch)
            {
                int serverRuntimeVersion = response.getArg0();
                int serverArchiveVersion = response.getArg1();

                int clientRuntimeVersion = getRuntimeVersion();
                int clientArchiveVersion = getArchiveVersion();

                // We should only get this error response, if server runtime or
                // archive version is older.

                RCF_VERIFY(
                        serverRuntimeVersion < clientRuntimeVersion 
                    ||  serverArchiveVersion < clientArchiveVersion,
                    Exception(_RcfError_Encoding()))
                        (serverRuntimeVersion)(serverArchiveVersion)
                        (clientRuntimeVersion)(clientArchiveVersion);

                RCF_VERIFY(
                    serverRuntimeVersion <= clientRuntimeVersion, 
                    Exception(_RcfError_Encoding()))
                        (serverRuntimeVersion)(clientRuntimeVersion);

                RCF_VERIFY(
                    serverArchiveVersion <= clientArchiveVersion, 
                    Exception(_RcfError_Encoding()))
                        (serverArchiveVersion)(clientArchiveVersion);

                if (getAutoVersioning() && getTries() == 0)
                {
                    setRuntimeVersion(serverRuntimeVersion);
                    if (serverArchiveVersion)
                    {
                        setArchiveVersion(serverArchiveVersion);
                    }
                    setTries(1);

                    init(mRequest.getSubInterface(), mRequest.getFnId(), mRcs);
                    send();
                }
                else
                {
                    onException( VersioningException(
                        serverRuntimeVersion, 
                        serverArchiveVersion));
                }
            }
            else if (err == RcfError_PingBack)
            {
                // A ping back message carries a parameter specifying
                // the ping back interval in ms. The client can use that
                // to make informed decisions about whether the connection
                // has died or not.

                mPingBackIntervalMs = response.getArg0();

                // Record a timestamp and go back to receiving.

                ++mPingBackCount;
                mPingBackTimeStamp = Platform::OS::getCurrentTimeMs();

                receive();
            }
            else
            {
                onException(RemoteException( Error(response.getError()) ));
            }
        }
        else
        {
            RCF::CurrentClientStubSentry sentry(*this);
            mpParameters->read(mIn);
            mIn.clearByteBuffer();

#ifdef RCF_USE_BOOST_FILESYSTEM

            // Check for any file downloads.
            {
                if (!mDownloadStreams.empty())
                {
                    std::vector<FileDownload> downloadStreams;
                    downloadStreams.swap(mDownloadStreams);
                    for (std::size_t i=0; i<downloadStreams.size(); ++i)
                    {
                        downloadStreams[i].download(*this);
                    }
                }
            }

#endif

            if (mAsync)
            {
                prepareAmiNotification();
            }
        }
    }

    void ClientStub::prepareAmiNotification()
    {
        std::auto_ptr<Lock> lockPtr( new Lock(*mSignalledMutex) );

        mSignalled = true;
        mSignalledCondition->notify_all();

        boost::function0<void> cb;
        if (mAsyncCallback)
        {
            cb = mAsyncCallback;
            mAsyncCallback = boost::function0<void>();                
        }

        getCurrentAmiNotification().set(cb, lockPtr, mSignalledMutex);
    }

    bool ClientStub::ready()
    {
        Lock lock(*mSignalledMutex);
        return mSignalled;
    }

    void ClientStub::wait(boost::uint32_t timeoutMs)
    {
        Lock lock(*mSignalledMutex);
        if (!mSignalled)
        {
            if (timeoutMs)
            {
                mSignalledCondition->timed_wait(lock, timeoutMs);
            }
            else
            {
                mSignalledCondition->wait(lock);
            }            
        }
    }

    void ClientStub::cancel()
    {
        mTransport->cancel();
        getCurrentAmiNotification().run();
    }

    void ClientStub::setSubRcfClientPtr(RcfClientPtr clientStubPtr)
    {
        Lock lock(mSubRcfClientMutex);
        mSubRcfClientPtr = clientStubPtr;
    }

    RcfClientPtr ClientStub::getSubRcfClientPtr()
    {
        Lock lock(mSubRcfClientMutex);
        return mSubRcfClientPtr;
    }

    void ClientStub::call( 
        RCF::RemoteCallSemantics rcs)
    {
        mRetry = false;
        mRcs = rcs;
        mPingBackTimeStamp = 0;
        mPingBackCount = 0;

        RCF_ASSERT_EQ(mAsyncTimerReason , None);

        // Set the progress timer timeouts.
        mTimerIntervalMs = 0;
        mNextTimerCallbackMs = 0;

        if (    mClientProgressPtr.get()
            &&  mClientProgressPtr->mTriggerMask & ClientProgress::Timer)
        {            
            mTimerIntervalMs = mClientProgressPtr->mTimerIntervalMs;

            mNextTimerCallbackMs = 
                Platform::OS::getCurrentTimeMs() + mTimerIntervalMs;

            // So we avoid the special value 0.
            mNextTimerCallbackMs |= 1;
        }

        // We don't set ping back timeouts until we are about to receive.
        mPingBackCheckIntervalMs = 0;
        mNextPingBackCheckMs = 0;

        mSignalled = false;
        
        send();
    }

    void ClientStub::setAsync(bool async)
    {
        mAsync = async;

        if (mAsync && !mSignalledMutex)
        {
            mSignalledMutex.reset( new Mutex() );
            mSignalledCondition.reset( new Condition() );
        }
    }

    bool ClientStub::getAsync()
    {
        return mAsync;
    }

    void ClientStub::setAsyncCallback(boost::function0<void> callback)
    {
        mAsyncCallback = callback;
    }

    std::auto_ptr<Exception> ClientStub::getAsyncException()
    {
        Lock lock(*mSignalledMutex);
        return mAsyncException;
    }

    void ClientStub::setAsyncException(std::auto_ptr<Exception> asyncException)
    {
        Lock lock(*mSignalledMutex);
        mAsyncException = asyncException;
    }

    bool ClientStub::hasAsyncException()
    {
        Lock lock(*mSignalledMutex);
        return mAsyncException.get() != NULL;
    }

    typedef boost::shared_ptr< ClientTransportAutoPtr > ClientTransportAutoPtrPtr;

    void vc6_helper(
        boost::function1<void, ClientTransportAutoPtr> func,
        ClientTransportAutoPtrPtr clientTransportAutoPtrPtr)
    {
        func(*clientTransportAutoPtrPtr);
    }

    void convertRcfSessionToRcfClient(
        boost::function1<void, ClientTransportAutoPtr> func,
        RemoteCallSemantics rcs)
    {
        RcfSession & rcfSession = getCurrentRcfSession();

        I_ServerTransportEx & serverTransport =
            dynamic_cast<I_ServerTransportEx &>(
                rcfSession.getSessionState().getServerTransport());

        ClientTransportAutoPtrPtr clientTransportAutoPtrPtr(
            new ClientTransportAutoPtr(
                serverTransport.createClientTransport(rcfSession.shared_from_this())));

        rcfSession.addOnWriteCompletedCallback(
            boost::bind(
                vc6_helper,
                func,
                clientTransportAutoPtrPtr) );

        bool closeSession = (rcs == RCF::Twoway);

        rcfSession.setCloseSessionAfterWrite(closeSession);
    }

    RcfSessionPtr convertRcfClientToRcfSession(
        ClientStub & clientStub, 
        I_ServerTransport & serverTransport,
        bool keepClientConnection)
    {
        I_ServerTransportEx &serverTransportEx =
            dynamic_cast<RCF::I_ServerTransportEx &>(serverTransport);

        ClientTransportAutoPtr clientTransportAutoPtr(
            clientStub.releaseTransport());

        SessionPtr sessionPtr = serverTransportEx.createServerSession(
            clientTransportAutoPtr,
            StubEntryPtr(),
            keepClientConnection);

        clientStub.setTransport(clientTransportAutoPtr);

        return sessionPtr;
    }

} // namespace RCF
