
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

#include <RCF/AsioServerTransport.hpp>

#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <RCF/Asio.hpp>
#include <RCF/AsioDeadlineTimer.hpp>
#include <RCF/AsyncFilter.hpp>
#include <RCF/ConnectionOrientedClientTransport.hpp>
#include <RCF/CurrentSession.hpp>
#include <RCF/MethodInvocation.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/TimedBsdSockets.hpp>

namespace RCF {    

    // FilterAdapter

    class FilterAdapter : public RCF::IdentityFilter
    {
    public:
        FilterAdapter(AsioSessionState &sessionState) :
            mSessionState(sessionState)
        {}

    private:
        void read(
            const ByteBuffer &byteBuffer,
            std::size_t bytesRequested)
        {
            mSessionState.read(byteBuffer, bytesRequested);
        }

        void write(
            const std::vector<ByteBuffer> &byteBuffers)
        {
            mSessionState.write(byteBuffers);
        }

        void onReadCompleted(
            const ByteBuffer &byteBuffer)
        {
            mSessionState.onReadWrite(byteBuffer.getLength());
        }

        void onWriteCompleted(
            std::size_t bytesTransferred)
        {
            mSessionState.onReadWrite(bytesTransferred);
        }

        const FilterDescription &getFilterDescription() const
        {
            RCF_ASSERT(0);
            return * (const FilterDescription *) NULL;
        }

        AsioSessionState &mSessionState;
    };


    ReadHandler::ReadHandler(const ReadHandler & rhs) : 
        mSessionState(rhs.mSessionState)
    {
    }

    ReadHandler::ReadHandler(AsioSessionState & sessionState) : 
        mSessionState(sessionState)
    {
    }

    void ReadHandler::operator()(boost::system::error_code err, std::size_t bytes)
    {
        AsioSessionStatePtr sessionStatePtr = mSessionState.mThisPtr;
        mSessionState.mThisPtr.reset();
        mSessionState.onReadCompletion(err, bytes);
    }

    void * ReadHandler::allocate(std::size_t size)
    {
        if (mSessionState.mReadHandlerBuffer.size() < size)
        {
            mSessionState.mReadHandlerBuffer.resize(size);
        }
        return & mSessionState.mReadHandlerBuffer[0];
    }

    WriteHandler::WriteHandler(const WriteHandler & rhs) : 
        mSessionState(rhs.mSessionState)
    {
    }

    WriteHandler::WriteHandler(AsioSessionState & sessionState) : 
        mSessionState(sessionState)
    {
    }

    void WriteHandler::operator()(boost::system::error_code err, std::size_t bytes)
    {
        AsioSessionStatePtr sessionStatePtr = mSessionState.mThisPtr;
        mSessionState.mThisPtr.reset();
        mSessionState.onWriteCompletion(err, bytes);
    }

    void * WriteHandler::allocate(std::size_t size)
    {
        if (mSessionState.mWriteHandlerBuffer.size() < size)
        {
            mSessionState.mWriteHandlerBuffer.resize(size);
        }
        return & mSessionState.mWriteHandlerBuffer[0];
    }

    void * asio_handler_allocate(std::size_t size, ReadHandler * pHandler)
    {
        return pHandler->allocate(size);
    }

    void asio_handler_deallocate(void * pointer, std::size_t size, ReadHandler * pHandler)
    {
    }

    void * asio_handler_allocate(std::size_t size, WriteHandler * pHandler)
    {
        return pHandler->allocate(size);
    }

    void asio_handler_deallocate(void * pointer, std::size_t size, WriteHandler * pHandler)
    {
    }

    void AsioSessionState::postRead()
    {
        mState = AsioSessionState::ReadingDataCount;
        mReadByteBuffer.clear();
        mTempByteBuffer.clear();
        getUniqueReadBuffer().resize(0);
        mReadBufferRemaining = 0;
        mIssueZeroByteRead = true;
        invokeAsyncRead();
    }

    ByteBuffer AsioSessionState::getReadByteBuffer()
    {
        if (mReadBufferPtr->empty())
        {
            return ByteBuffer();
        }

        return ByteBuffer(
            &(*mReadBufferPtr)[0],
            (*mReadBufferPtr).size(),
            mReadBufferPtr);
    }

    void AsioSessionState::postWrite(
        std::vector<ByteBuffer> &byteBuffers)
    {
        RCF_ASSERT_EQ(mState , Ready);

        BOOST_STATIC_ASSERT(sizeof(unsigned int) == 4);

        mSlicedWriteByteBuffers.resize(0);
        mWriteByteBuffers.resize(0);

        std::copy(
            byteBuffers.begin(),
            byteBuffers.end(),
            std::back_inserter(mWriteByteBuffers));

        byteBuffers.resize(0);

        int messageSize = 
            static_cast<int>(RCF::lengthByteBuffers(mWriteByteBuffers));
        
        ByteBuffer &byteBuffer = mWriteByteBuffers.front();

        RCF_ASSERT_GTEQ(byteBuffer.getLeftMargin() , 4);

        byteBuffer.expandIntoLeftMargin(4);
        memcpy(byteBuffer.getPtr(), &messageSize, 4);
        RCF::machineToNetworkOrder(byteBuffer.getPtr(), 4, 1);

        mState = AsioSessionState::WritingData;
        
        mWriteBufferRemaining = RCF::lengthByteBuffers(mWriteByteBuffers);
        
        invokeAsyncWrite();
    }

    void AsioSessionState::postClose()
    {
        close();
    }

    I_ServerTransport & AsioSessionState::getServerTransport()
    {
        return mTransport;
    }

    const I_RemoteAddress &
        AsioSessionState::getRemoteAddress()
    {
        return implGetRemoteAddress();
    }

    int AsioSessionState::getNativeHandle() const
    {
        return implGetNative();
    }

    // SessionState

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4355 )  // warning C4355: 'this' : used in base member initializer list
#endif

    AsioSessionState::AsioSessionState(
        AsioServerTransport & transport,
        AsioIoService & ioService) :
            mState(Ready),
            mIssueZeroByteRead(false),
            mReadBufferRemaining(RCF_DEFAULT_INIT),
            mWriteBufferRemaining(RCF_DEFAULT_INIT),
            mTransport(transport),
            mFilterAdapterPtr(new FilterAdapter(*this)),
            mHasBeenClosed(RCF_DEFAULT_INIT),
            mCloseAfterWrite(RCF_DEFAULT_INIT),
            mReflecting(RCF_DEFAULT_INIT),
            mIoService(ioService)
    {
    }

#ifdef _MSC_VER
#pragma warning( pop )
#endif

    AsioSessionState::~AsioSessionState()
    {
        RCF_DTOR_BEGIN

        // TODO: invoke accept if appropriate
        // TODO: need a proper acceptex strategy in the first place
        //RCF_ASSERT(mState != Accepting);

        mTransport.unregisterSession(mWeakThisPtr);

        RCF_LOG_4()(mState)(mSessionPtr.get())(mHasBeenClosed) 
            << "AsioSessionState - destructor.";

        // close reflecting session if appropriate
        if (mReflecting)
        {
            AsioSessionStatePtr sessionStatePtr(mReflecteeWeakPtr.lock());
            if (sessionStatePtr)
            {
                sessionStatePtr->close();
            }
        }

        RCF_DTOR_END;
    }

    AsioSessionStatePtr AsioSessionState::sharedFromThis()
    {
        return boost::static_pointer_cast<AsioSessionState>(shared_from_this());
    }

    ReallocBuffer & AsioSessionState::getReadBuffer()
    {
        if (!mReadBufferPtr)
        {
            mReadBufferPtr.reset( new ReallocBuffer() );
        }
        return *mReadBufferPtr;
    }

    ReallocBuffer & AsioSessionState::getUniqueReadBuffer()
    {
        if (!mReadBufferPtr || !mReadBufferPtr.unique())
        {
            mReadBufferPtr.reset( new ReallocBuffer() );
        }
        return *mReadBufferPtr;
    }

    ByteBuffer AsioSessionState::getReadByteBuffer() const
    {
        return ByteBuffer(
            &(*mReadBufferPtr)[0],
            (*mReadBufferPtr).size(),
            mReadBufferPtr);
    }

    ReallocBuffer & AsioSessionState::getReadBufferSecondary()
    {
        if (!mReadBufferSecondaryPtr)
        {
            mReadBufferSecondaryPtr.reset( new ReallocBuffer() );
        }
        return *mReadBufferSecondaryPtr;
    }

    ReallocBuffer & AsioSessionState::getUniqueReadBufferSecondary()
    {
        if (!mReadBufferSecondaryPtr || !mReadBufferSecondaryPtr.unique())
        {
            mReadBufferSecondaryPtr.reset( new ReallocBuffer() );
        }
        return *mReadBufferSecondaryPtr;
    }

    ByteBuffer 
        AsioSessionState::getReadByteBufferSecondary() const
    {
        if (mReadBufferSecondaryPtr->empty())
        {
            return ByteBuffer();
        }

        return ByteBuffer(
            &(*mReadBufferSecondaryPtr)[0],
            (*mReadBufferSecondaryPtr).size(),
            mReadBufferSecondaryPtr);
    }

    void AsioSessionState::read(
        const ByteBuffer &byteBuffer,
        std::size_t bytesRequested)
    {

        if (byteBuffer.getLength() == 0)
        {
            ReallocBuffer &vec = getUniqueReadBufferSecondary();
            vec.resize(bytesRequested);
            mTempByteBuffer = getReadByteBufferSecondary();
        }
        else
        {
            mTempByteBuffer = ByteBuffer(byteBuffer, 0, bytesRequested);
        }

        RCF_ASSERT_LTEQ(bytesRequested , mTempByteBuffer.getLength());

        char *buffer = mTempByteBuffer.getPtr();
        std::size_t bufferLen = mTempByteBuffer.getLength();

        Lock lock(mMutex);
        if (!mHasBeenClosed)
        {
            implRead(buffer, bufferLen);
        }
    }

    void AsioSessionState::write(
        const std::vector<ByteBuffer> &byteBuffers)
    {
        RCF_ASSERT(!byteBuffers.empty());

        Lock lock(mMutex);
        if (!mHasBeenClosed)
        {
            implWrite(byteBuffers);
        }
    }

    // TODO: merge onReadCompletion/onWriteCompletion into one function

    void AsioSessionState::onReadCompletion(
        boost::system::error_code error, size_t bytesTransferred)
    {
        RCF_LOG_4()(this)(bytesTransferred) << "AsioSessionState - read from socket completed.";

        ThreadTouchGuard threadTouchGuard;

#ifdef BOOST_WINDOWS

        if (error.value() == ERROR_OPERATION_ABORTED)
        {
            error.clear();
        }

#endif

        if (!error && !mTransport.mStopFlag)
        {
            if (bytesTransferred == 0 && mIssueZeroByteRead)
            {
                getUniqueReadBuffer().resize(4);
                mReadBufferRemaining = 4;
                mIssueZeroByteRead = false;
                invokeAsyncRead();
            }
            else if (mReflecting)
            {
                boost::system::error_code ec;
                onReflectedReadWrite(ec, bytesTransferred);
            }
            else
            {
                CurrentRcfSessionSentry guard(*mSessionPtr);

                mTempByteBuffer = ByteBuffer(
                    mTempByteBuffer, 
                    0, 
                    bytesTransferred);

                mTransportFilters.empty() ?
                    onReadWrite(bytesTransferred) : 
                    mTransportFilters.back()->onReadCompleted(mTempByteBuffer);
            }
        }
    }

    void AsioSessionState::onWriteCompletion(
        boost::system::error_code error, 
        size_t bytesTransferred)
    {
        RCF_LOG_4()(this)(bytesTransferred) << "AsioSessionState - write to socket completed.";

        ThreadTouchGuard threadTouchGuard;

#ifdef BOOST_WINDOWS

        if (error.value() == ERROR_OPERATION_ABORTED)
        {
            error.clear();
        }

#endif

        if (!error && !mTransport.mStopFlag)
        {
            if (mReflecting)
            {
                if (mReflecteePtr)
                {
                    mReflecteePtr.reset();
                }
                boost::system::error_code ec;
                onReflectedReadWrite(ec, bytesTransferred);
            }
            else
            {
                CurrentRcfSessionSentry guard(*mSessionPtr);
                mTransportFilters.empty() ?
                    onReadWrite(bytesTransferred) :
                    mTransportFilters.back()->onWriteCompleted(bytesTransferred);
            }
        }
    }

    void AsioSessionState::setTransportFilters(
        const std::vector<FilterPtr> &filters)
    {

        mTransportFilters.assign(filters.begin(), filters.end());
        connectFilters(mTransportFilters);
        if (!mTransportFilters.empty())
        {
            mTransportFilters.front()->setPreFilter( *mFilterAdapterPtr );
            mTransportFilters.back()->setPostFilter( *mFilterAdapterPtr );
        }
    }

    void AsioSessionState::getTransportFilters(
        std::vector<FilterPtr> &filters)
    {
        filters = mTransportFilters;
    }

    void AsioSessionState::invokeAsyncRead()
    {
        mReadByteBuffer = ByteBuffer(
            getReadByteBuffer(),
            getReadByteBuffer().getLength()-mReadBufferRemaining);

        mTransportFilters.empty() ?
            read(mReadByteBuffer, mReadBufferRemaining) :
            mTransportFilters.front()->read(mReadByteBuffer, mReadBufferRemaining);
    }

    void AsioSessionState::invokeAsyncWrite()
    {
        mSlicedWriteByteBuffers.resize(0);

        sliceByteBuffers(
            mSlicedWriteByteBuffers,
            mWriteByteBuffers,
            lengthByteBuffers(mWriteByteBuffers)-mWriteBufferRemaining);

        mTransportFilters.empty() ?
            write(mSlicedWriteByteBuffers) :
            mTransportFilters.front()->write(mSlicedWriteByteBuffers);

    }

    void AsioSessionState::invokeAsyncAccept()
    {
        mState = Accepting;
        implAccept();
    }

    void AsioSessionState::onAccept(
        const boost::system::error_code& error)
    {
        RCF_LOG_4()(error.value())
            << "AsioSessionState - onAccept().";

        if (mTransport.mStopFlag)
        {
            RCF_LOG_4()(error.value())
                << "AsioSessionState - onAccept(). Returning early, stop flag is set.";

            return;
        }

        if (
            error == boost::asio::error::connection_aborted ||
            error == boost::asio::error::operation_aborted)
        {
            invokeAsyncAccept();
            return;
        }

        // create a new SessionState, and do an accept on that
        mTransport.createSessionState()->invokeAsyncAccept();

        if (!error)
        {
            // save the remote address in the SessionState object
            bool clientAddrAllowed = implOnAccept();
            mState = WritingData;

            // set current RCF session
            CurrentRcfSessionSentry guard(*mSessionPtr);

            if (clientAddrAllowed)
            {
                // Check the connection limit.
                bool allowConnect = true;
                std::size_t connectionLimit = mTransport.getConnectionLimit();
                if (connectionLimit)
                {
                    Lock lock(mTransport.mSessionsMutex);
                    
                    RCF_ASSERT_LTEQ(
                        mTransport.mSessions.size() , 1+1+connectionLimit);

                    if (mTransport.mSessions.size() == 1+1+connectionLimit)
                    {
                        allowConnect = false;
                    }
                }

                if (allowConnect)
                {
                    // start things rolling by faking a completed write operation
                    onReadWrite(0);
                }
                else
                {
                    sendServerError(RcfError_ConnectionLimitExceeded);
                }
            }
        }
    }

    void onError(
        boost::system::error_code &error1, 
        const boost::system::error_code &error2)
    {
        error1 = error2;
    }

    void AsioSessionState::sendServerError(int error)
    {
        mState = Ready;
        mCloseAfterWrite = true;
        std::vector<ByteBuffer> byteBuffers(1);
        encodeServerError(*mTransport.mpServer, byteBuffers.front(), error);
        mSessionPtr->getSessionState().postWrite(byteBuffers);
    }

    void AsioSessionState::onReadWrite(
        size_t bytesTransferred)
    {
        RCF_ASSERT(!mReflecting);
        {
            switch(mState)
            {
            case ReadingDataCount:
            case ReadingData:

                RCF_ASSERT_LTEQ(bytesTransferred , mReadBufferRemaining);

                mReadBufferRemaining -= bytesTransferred;
                if (    mReadBufferRemaining > 0 
                    ||  (mReadBufferRemaining == 0 && mIssueZeroByteRead))
                {
                    if (mReadBufferRemaining == 0 && mIssueZeroByteRead)
                    {
                        getUniqueReadBuffer().resize(4);
                        mReadBufferRemaining = 4;
                        mIssueZeroByteRead = false;
                    }
                    invokeAsyncRead();
                }
                else
                {
                    RCF_ASSERT_EQ(mReadBufferRemaining , 0);
                    if (mState == ReadingDataCount)
                    {
                        ReallocBuffer &readBuffer = getReadBuffer();
                        RCF_ASSERT_EQ(readBuffer.size() , 4);

                        unsigned int packetLength = 0;
                        memcpy(&packetLength, &readBuffer[0], 4);
                        networkToMachineOrder(&packetLength, 4, 1);
                        
                        if (    0 < packetLength 
                            &&  packetLength <= mTransport.getMaxMessageLength())
                        {
                            readBuffer.resize(packetLength);
                            mReadBufferRemaining = packetLength;
                            mState = ReadingData;
                            invokeAsyncRead();
                        }
                        else
                        {
                            sendServerError(RcfError_ServerMessageLength);
                        }

                    }
                    else if (mState == ReadingData)
                    {
                        mState = Ready;

                        mTransport.getSessionManager().onReadCompleted(
                            getSessionPtr());

                        if (mTransport.mInterrupt)
                        {
                            mTransport.mInterrupt = false;
                            mTransport.mpIoService->stop();
                        }
                    }
                }
                break;


            case WritingData:

                RCF_ASSERT_LTEQ(bytesTransferred , mWriteBufferRemaining);

                mWriteBufferRemaining -= bytesTransferred;
                if (mWriteBufferRemaining > 0)
                {
                    invokeAsyncWrite();
                }
                else
                {
                    if (mCloseAfterWrite)
                    {
                        // TODO: this code is only valid for a TCP connection, 
                        // so should really be in TcpAsioServerTransport.cpp.

                        int fd = implGetNative();
                        const int BufferSize = 8*1024;
                        char buffer[BufferSize];
                        while (recv(fd, buffer, BufferSize, 0) > 0);
#ifdef BOOST_WINDOWS
                        int ret = shutdown(fd, SD_BOTH);
#else
                        int ret = shutdown(fd, SHUT_RDWR);
#endif
                        RCF_UNUSED_VARIABLE(ret);
                        postRead();
                    }
                    else
                    {
                        mState = Ready;

                        mSlicedWriteByteBuffers.resize(0);
                        mWriteByteBuffers.resize(0);

                        mTransport.getSessionManager().onWriteCompleted(
                            getSessionPtr());

                        if (mTransport.mInterrupt)
                        {
                            mTransport.mInterrupt = false;
                            mTransport.mpIoService->stop();
                        }
                    }
                }
                break;

            default:
                RCF_ASSERT(0);
            }
        }
    }

    void AsioSessionState::onReflectedReadWrite(
        const boost::system::error_code& error,
        size_t bytesTransferred)
    {
        RCF_UNUSED_VARIABLE(error);

        RCF_ASSERT(
            mState == ReadingData ||
            mState == ReadingDataCount ||
            mState == WritingData)
            (mState);

        RCF_ASSERT(mReflecting);

        if (bytesTransferred == 0)
        {
            // Previous operation was aborted for some reason (probably because
            // of a thread exiting). Reissue the operation.

            mState = (mState == WritingData) ? ReadingData : WritingData;
        }

        if (mState == WritingData)
        {
            mState = ReadingData;
            ReallocBuffer &readBuffer = getReadBuffer();
            readBuffer.resize(8*1024);

            char *buffer = &readBuffer[0];
            std::size_t bufferLen = readBuffer.size();

            Lock lock(mMutex);
            if (!mHasBeenClosed)
            {
                implRead(buffer, bufferLen);
            }
        }
        else if (
            mState == ReadingData ||
            mState == ReadingDataCount)
        {
            mState = WritingData;
            ReallocBuffer &readBuffer = getReadBuffer();

            char *buffer = &readBuffer[0];
            std::size_t bufferLen = bytesTransferred;

            // mReflecteePtr will be nulled in onWriteCompletion(), otherwise 
            // we could easily end up with a cycle
            RCF_ASSERT(!mReflecteePtr);
            mReflecteePtr = mReflecteeWeakPtr.lock();
            if (mReflecteePtr)
            {
                RCF_ASSERT(mReflecteePtr->mReflecting);

                Lock lock(mReflecteePtr->mMutex);
                if (!mReflecteePtr->mHasBeenClosed)
                {
                    // TODO: if this can throw, then we need a scope_guard
                    // to reset mReflecteePtr
                    mReflecteePtr->implWrite(*this, buffer, bufferLen);
                }
            }
        }
    }

    // AsioServerTransport

    AsioSessionStatePtr AsioServerTransport::createSessionState()
    {
        AsioSessionStatePtr sessionStatePtr( implCreateSessionState() );
        SessionPtr sessionPtr( getSessionManager().createSession() );
        sessionPtr->setSessionState( *sessionStatePtr );
        sessionStatePtr->setSessionPtr(sessionPtr);
        sessionStatePtr->mWeakThisPtr = sessionStatePtr;
        registerSession(sessionStatePtr->mWeakThisPtr);
        return sessionStatePtr;
    }

    // I_ServerTransportEx implementation

    ClientTransportAutoPtr AsioServerTransport::createClientTransport(
        const I_Endpoint &endpoint)
    {
        return implCreateClientTransport(endpoint);
    }

    SessionPtr AsioServerTransport::createServerSession(
        ClientTransportAutoPtr & clientTransportAutoPtr,
        StubEntryPtr stubEntryPtr,
        bool keepClientConnection)
    {
        AsioSessionStatePtr sessionStatePtr(createSessionState());
        SessionPtr sessionPtr(sessionStatePtr->getSessionPtr());
        sessionStatePtr->implTransferNativeFrom(*clientTransportAutoPtr);

        if (stubEntryPtr)
        {
            sessionPtr->setDefaultStubEntryPtr(stubEntryPtr);
        }

        clientTransportAutoPtr.reset();
        if (keepClientConnection)
        {
            clientTransportAutoPtr = createClientTransport(sessionPtr);
        }

        sessionStatePtr->mState = AsioSessionState::WritingData;
        sessionStatePtr->onReadWrite(0);
        return sessionPtr;
    }

    ClientTransportAutoPtr AsioServerTransport::createClientTransport(
        SessionPtr sessionPtr)
    {
        AsioSessionState & sessionState = 
            dynamic_cast<AsioSessionState &>(sessionPtr->getSessionState());

        AsioSessionStatePtr sessionStatePtr = sessionState.sharedFromThis();

        ClientTransportAutoPtr clientTransportPtr =
            sessionStatePtr->implCreateClientTransport();

        ConnectionOrientedClientTransport & coClientTransport =
            static_cast<ConnectionOrientedClientTransport &>(
                *clientTransportPtr);

        coClientTransport.setNotifyCloseFunctor( boost::bind(
            &AsioServerTransport::notifyClose,
            this,
            AsioSessionStateWeakPtr(sessionStatePtr)));

        coClientTransport.setCloseFunctor( 
            sessionStatePtr->implGetCloseFunctor() );

        return clientTransportPtr;
    }

    bool AsioServerTransport::reflect(
        const SessionPtr &sessionPtr1, 
        const SessionPtr &sessionPtr2)
    {
        AsioSessionState & sessionState1 = 
            dynamic_cast<AsioSessionState &>(sessionPtr1->getSessionState());

        AsioSessionStatePtr sessionStatePtr1 = sessionState1.sharedFromThis();

        AsioSessionState & sessionState2 = 
            dynamic_cast<AsioSessionState &>(sessionPtr2->getSessionState());

        AsioSessionStatePtr sessionStatePtr2 = sessionState2.sharedFromThis();

        sessionStatePtr1->mReflecteeWeakPtr = sessionStatePtr2;
        sessionStatePtr2->mReflecteeWeakPtr = sessionStatePtr1;

        sessionStatePtr1->mReflecting = true;
        sessionStatePtr2->mReflecting = true;

        return true;
    }

    bool AsioServerTransport::isConnected(const SessionPtr &sessionPtr)
    {
        AsioSessionState & sessionState = 
            dynamic_cast<AsioSessionState &>(sessionPtr->getSessionState());

        AsioSessionStatePtr sessionStatePtr = sessionState.sharedFromThis();

        // TODO: what to do for non-TCP sockets
        return 
            sessionStatePtr.get() 
            && isFdConnected(sessionStatePtr->implGetNative());
    }

    // I_Service implementation

    void AsioServerTransport::open()
    {
        mInterrupt = false;
        mStopFlag = false;
        implOpen();
    }

    void AsioSessionState::setSessionPtr(
        SessionPtr sessionPtr)    
    { 
        mSessionPtr = sessionPtr; 
    }
    
    SessionPtr AsioSessionState::getSessionPtr()
    { 
        return mSessionPtr; 
    }

    void AsioSessionState::close()
    {
        Lock lock(mMutex);
        if (!mHasBeenClosed)
        {
            implClose();
            mHasBeenClosed = true;
        }
    }

    void AsioServerTransport::notifyClose(
        AsioSessionStateWeakPtr sessionStateWeakPtr)
    {
        AsioSessionStatePtr sessionStatePtr(sessionStateWeakPtr.lock());
        if (sessionStatePtr)
        {
            Lock lock(sessionStatePtr->mMutex);
            sessionStatePtr->mHasBeenClosed = true;
        }
    }

    void AsioServerTransport::close()
    {
        mAcceptorPtr.reset();
        mStopFlag = true;
        cancelOutstandingIo();
        mpIoService->reset();
        std::size_t items = mpIoService->poll();
        while (items)
        {
            mpIoService->reset();
            items = mpIoService->poll();
        }

        mpIoService = NULL;
        mpServer = NULL;
    }

    void AsioServerTransport::stop()
    {
        mpIoService->stop();
    }

    void AsioServerTransport::onServiceAdded(RcfServer &server)
    {
        setServer(server);
        mTaskEntries.clear();
        mTaskEntries.push_back(TaskEntry(Mt_Asio));
    }

    void AsioServerTransport::onServiceRemoved(RcfServer &)
    {}

    void AsioServerTransport::onServerStart(RcfServer & server)
    {
        open();

        mStopFlag = false;
        mpServer  = &server;
        mpIoService = mTaskEntries[0].getThreadPool().getIoService();        
    }

    void AsioServerTransport::startAccepting()
    {
        createSessionState()->invokeAsyncAccept();
    }

    void AsioServerTransport::onServerStop(RcfServer &)
    {
        close();
    }

    void AsioServerTransport::setServer(RcfServer &server)
    {
        mpServer = &server;
    }

    RcfServer & AsioServerTransport::getServer()
    {
        return *mpServer;
    }

    RcfServer & AsioServerTransport::getSessionManager()
    {
        return *mpServer;
    }

    AsioServerTransport::AsioServerTransport() :
        mpIoService(RCF_DEFAULT_INIT),
        mAcceptorPtr(),
        mInterrupt(RCF_DEFAULT_INIT),
        mStopFlag(RCF_DEFAULT_INIT),
        mpServer(RCF_DEFAULT_INIT)
    {
    }

    AsioServerTransport::~AsioServerTransport()
    {
    }

    void AsioServerTransport::registerSession(AsioSessionStateWeakPtr session)
    {
        Lock lock(mSessionsMutex);
        mSessions.insert(session);
    }

    void AsioServerTransport::unregisterSession(AsioSessionStateWeakPtr session)
    {
        Lock lock(mSessionsMutex);
        mSessions.erase(session);
    }

    void AsioServerTransport::cancelOutstandingIo()
    {
        Lock lock(mSessionsMutex);
        std::for_each( 
            mSessions.begin(), 
            mSessions.end(), 
            boost::bind(&AsioServerTransport::closeSessionState, this, _1));
    }

    void AsioServerTransport::closeSessionState(
        AsioSessionStateWeakPtr sessionStateWeakPtr)
    {
        AsioSessionStatePtr sessionStatePtr(sessionStateWeakPtr.lock());
        if (sessionStatePtr)
        {
            sessionStatePtr->close();
        }
    }

    AsioAcceptorPtr AsioServerTransport::getAcceptorPtr()
    {
        return mAcceptorPtr;
    }

    AsioIoService & AsioServerTransport::getIoService()
    {
        return *mpIoService;
    }

} // namespace RCF
