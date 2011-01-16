
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

#include <RCF/RcfServer.hpp>

#include <algorithm>

#include <boost/bind.hpp>

#include <RCF/AsyncFilter.hpp>
#include <RCF/CurrentSession.hpp>
#include <RCF/Endpoint.hpp>
#include <RCF/FilterService.hpp>
#include <RCF/IpServerTransport.hpp>
#include <RCF/Marshal.hpp>
#include <RCF/MethodInvocation.hpp>
#include <RCF/ObjectFactoryService.hpp>
#include <RCF/PingBackService.hpp>
#include <RCF/RcfClient.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerTask.hpp>
#include <RCF/SessionTimeoutService.hpp>
#include <RCF/Service.hpp>
#include <RCF/StubEntry.hpp>
#include <RCF/ThreadLocalCache.hpp>
#include <RCF/Token.hpp>
#include <RCF/Tools.hpp>
#include <RCF/Version.hpp>

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/memory.hpp>
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
#include <RCF/BsAutoPtr.hpp>
#endif

#ifdef RCF_USE_BOOST_FILESYSTEM
#include <RCF/FileTransferService.hpp>
#else
namespace RCF { class FileTransferService {}; }
#endif

namespace RCF {

    void repeatCycleServer(RcfServer &server, int timeoutMs)
    {
        while (!server.cycle(timeoutMs));
    }

    // RcfServer

    RcfServer::RcfServer() :
        mStubMapMutex(WriterPriority),
        mServicesMutex(WriterPriority),
        mServerThreadsStopFlag(RCF_DEFAULT_INIT),
        mStarted(RCF_DEFAULT_INIT),
        mThreadPoolPtr( new ThreadPool(1, "RCF Server") ),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion())
    {
    }

    RcfServer::RcfServer(const I_Endpoint &endpoint) :
        mStubMapMutex(WriterPriority),
        mServicesMutex(WriterPriority),
        mServerThreadsStopFlag(RCF_DEFAULT_INIT),
        mStarted(RCF_DEFAULT_INIT),
        mThreadPoolPtr( new ThreadPool(1, "RCF Server") ),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion())
    {
        addEndpoint(endpoint);
    }

    RcfServer::RcfServer(ServicePtr servicePtr) :
        mStubMapMutex(WriterPriority),
        mServicesMutex(WriterPriority),
        mServerThreadsStopFlag(RCF_DEFAULT_INIT),
        mStarted(RCF_DEFAULT_INIT),
        mThreadPoolPtr( new ThreadPool(1, "RCF Server") ),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion())
    {
        addService(servicePtr);
    }

    RcfServer::RcfServer(ServerTransportPtr serverTransportPtr) :
        mStubMapMutex(WriterPriority),
        mServicesMutex(WriterPriority),
        mServerThreadsStopFlag(RCF_DEFAULT_INIT),
        mStarted(RCF_DEFAULT_INIT),
        mThreadPoolPtr( new ThreadPool(1, "RCF Server") ),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion())
    {
        addService( boost::dynamic_pointer_cast<I_Service>(serverTransportPtr) );
    }

    RcfServer::~RcfServer()
    {
        RCF_DTOR_BEGIN
            stop();
        RCF_DTOR_END
    }

    bool RcfServer::addService(ServicePtr servicePtr)
    {
        RCF_LOG_2()(typeid(*servicePtr).name()) << "RcfServer - adding service.";

        WriteLock lock(mServicesMutex);

        if (
            std::find(
                mServices.begin(),
                mServices.end(),
                servicePtr) == mServices.end())
        {
            mServices.push_back(servicePtr);

            ObjectFactoryServicePtr objectFactoryServicePtr =
                boost::dynamic_pointer_cast<ObjectFactoryService>(servicePtr);

            if (objectFactoryServicePtr)
            {
                mObjectFactoryServicePtr = objectFactoryServicePtr;
            }

            FilterServicePtr filterServicePtr =
                boost::dynamic_pointer_cast<FilterService>(servicePtr);

            if (filterServicePtr)
            {
                mFilterServicePtr = filterServicePtr;
            }

            ServerTransportPtr serverTransportPtr =
                boost::dynamic_pointer_cast<I_ServerTransport>(servicePtr);

            if (serverTransportPtr)
            {
                mServerTransports.push_back(serverTransportPtr);
            }

            PingBackServicePtr pingBackServicePtr =
                boost::dynamic_pointer_cast<PingBackService>(servicePtr);

            if (pingBackServicePtr)
            {
                mPingBackServicePtr = pingBackServicePtr;
            }

            FileTransferServicePtr fileTransferServicePtr =
                boost::dynamic_pointer_cast<FileTransferService>(servicePtr);

            if (fileTransferServicePtr)
            {
                mFileTransferServicePtr = fileTransferServicePtr;
            }

            SessionTimeoutServicePtr sessionTimeoutServicePtr =
                boost::dynamic_pointer_cast<SessionTimeoutService>(servicePtr);

            if (sessionTimeoutServicePtr)
            {
                mSessionTimeoutServicePtr = sessionTimeoutServicePtr;
            }

            lock.unlock();
            servicePtr->onServiceAdded(*this);

            Lock lock(mStartStopMutex);
            if (mStarted)
            {
                resolveServiceThreadPools(servicePtr);
                startService(servicePtr);
            }

            return true;
        }
        return false;
    }

    bool RcfServer::removeService(ServicePtr servicePtr)
    {
        RCF_LOG_2()(typeid(*servicePtr).name()) << "Removing service.";

        WriteLock lock(mServicesMutex);

        std::vector<ServicePtr>::iterator iter =
            std::find(mServices.begin(), mServices.end(), servicePtr);

        if (iter != mServices.end())
        {
            mServices.erase(iter);

            ObjectFactoryServicePtr objectFactoryServicePtr =
                boost::dynamic_pointer_cast<ObjectFactoryService>(servicePtr);

            if (objectFactoryServicePtr)
            {
                mObjectFactoryServicePtr.reset();
            }

            FilterServicePtr filterServicePtr =
                boost::dynamic_pointer_cast<FilterService>(servicePtr);

            if (filterServicePtr)
            {
                mFilterServicePtr.reset();
            }

            ServerTransportPtr serverTransportPtr =
                boost::dynamic_pointer_cast<I_ServerTransport>(servicePtr);

            if (serverTransportPtr)
            {
                eraseRemove(mServerTransports, serverTransportPtr);
            }

            PingBackServicePtr pingBackServicePtr =
                boost::dynamic_pointer_cast<PingBackService>(servicePtr);

            if (pingBackServicePtr)
            {
                mPingBackServicePtr.reset();
            }

            FileTransferServicePtr fileTransferServicePtr =
                boost::dynamic_pointer_cast<FileTransferService>(servicePtr);

            if (fileTransferServicePtr)
            {
                mFileTransferServicePtr.reset();
            }

            SessionTimeoutServicePtr sessionTimeoutServicePtr =
                boost::dynamic_pointer_cast<SessionTimeoutService>(servicePtr);

            if (sessionTimeoutServicePtr)
            {
                mSessionTimeoutServicePtr.reset();
            }

            lock.unlock();

            stopService(servicePtr, true);
            servicePtr->onServiceRemoved(*this);
            return true;
        }
        return false;
    }

    bool RcfServer::addServerTransport(ServerTransportPtr serverTransportPtr)
    {
        return addService(
            boost::dynamic_pointer_cast<I_Service>(serverTransportPtr));
    }

    bool RcfServer::removeServerTransport(ServerTransportPtr serverTransportPtr)
    {
        return removeService(
            boost::dynamic_pointer_cast<I_Service>(serverTransportPtr));
    }

#ifdef RCF_MULTI_THREADED

    void RcfServer::start()
    {
        startImpl(true);
    }

#endif

    void RcfServer::startSt()
    {
        startImpl(false);
    }

    void RcfServer::resolveServiceThreadPools(ServicePtr servicePtr) const
    {
        I_Service & service = *servicePtr;
        TaskEntries & taskEntries = service.mTaskEntries;
        for (std::size_t j=0; j<taskEntries.size(); ++j)
        {
            TaskEntry & taskEntry = taskEntries[j];

            // Muxer type == 0 , means we have to have a local thread manager.
            if (taskEntry.mMuxerType == Mt_None && !taskEntry.mLocalThreadPoolPtr)
            {
                taskEntry.mLocalThreadPoolPtr.reset( new ThreadPool(1, "") );
            }

            if (taskEntry.mLocalThreadPoolPtr)
            {
                taskEntry.mLocalThreadPoolPtr->setTask(taskEntry.mTask);
                taskEntry.mLocalThreadPoolPtr->setStopFunctor(taskEntry.mStopFunctor);
                taskEntry.mLocalThreadPoolPtr->setThreadName(taskEntry.mThreadName);
                taskEntry.mWhichThreadPoolPtr = taskEntry.mLocalThreadPoolPtr;
            }
            else if (service.mThreadPoolPtr)
            {
                taskEntry.mWhichThreadPoolPtr = service.mThreadPoolPtr;
            }
            else
            {
                taskEntry.mWhichThreadPoolPtr = mThreadPoolPtr;
            }

            taskEntry.mWhichThreadPoolPtr->enableMuxerType(taskEntry.mMuxerType);
        }
    }

    void RcfServer::startImpl(bool spawnThreads)
    {
        Lock lock(mStartStopMutex);
        if (!mStarted)
        {
            mServerThreadsStopFlag = false;

            // open the server

            std::vector<ServicePtr> services;
            {
                ReadLock lock(mServicesMutex);
                services = mServices;
            }

            for (std::size_t i=0; i<services.size(); ++i)
            {
                resolveServiceThreadPools(services[i]);
            }

            // notify all services
            std::for_each(
                services.begin(),
                services.end(),
                boost::bind(&I_Service::onServerStart, _1, boost::ref(*this)));

            // spawn internal worker threads
            if (spawnThreads)
            {
                std::for_each(
                    services.begin(),
                    services.end(),
                    boost::bind(&RcfServer::startService, boost::cref(*this), _1));
            }

            mStarted = true;

            // call the start notification callback, if there is one
            invokeStartCallback();

            // notify anyone who was waiting on the stop event
            mStartEvent.notify_all();
        }
    }

    void RcfServer::addJoinFunctor(const JoinFunctor &joinFunctor)
    {
        if (joinFunctor)
        {
            mJoinFunctors.push_back(joinFunctor);
        }
    }

    void RcfServer::startInThisThread()
    {
        startInThisThread(JoinFunctor());
    }

    void RcfServer::startInThisThread(const JoinFunctor &joinFunctor)
    {
        startSt();

        // register the join functor
        if (joinFunctor)
        {
            mJoinFunctors.push_back(joinFunctor);
        }

        // run all tasks sequentially in this thread
        repeatCycleServer(*this, 500);

    }

    bool RcfServer::cycle(int timeoutMs)
    {
        // Cycle each task of each service, once.

        ReadLock lock(mServicesMutex);
        for (std::size_t i=0; i<mServices.size(); ++i)
        {
            TaskEntries & taskEntries = mServices[i]->mTaskEntries;
            for (std::size_t j=0; j<taskEntries.size(); ++j)
            {
                TaskEntry & taskEntry = taskEntries[j];
                int cycleTimeoutMs = 0;
                if (i==0 && j == 0)
                {
                    cycleTimeoutMs = timeoutMs;
                }
                ShouldStop shouldStop(mServerThreadsStopFlag, ThreadInfoPtr());
                taskEntry.mWhichThreadPoolPtr->cycle(cycleTimeoutMs, shouldStop);
            }
        }

        return mServerThreadsStopFlag;
    }

    void RcfServer::startService(ServicePtr servicePtr) const
    {
        RCF_LOG_2()(typeid(*servicePtr).name()) << "RcfServer - starting service.";

        TaskEntries &taskEntries = servicePtr->mTaskEntries;
        std::for_each(
            taskEntries.begin(),
            taskEntries.end(),
            boost::bind(
                &TaskEntry::start,
                _1,
                boost::ref(mServerThreadsStopFlag)));
    }

    void RcfServer::stopService(ServicePtr servicePtr, bool wait) const
    {
        RCF_LOG_2()(typeid(*servicePtr).name())(wait) << "RcfServer - stopping service.";

        typedef void (TaskEntry::*Pfn)(bool);

        TaskEntries &taskEntries = servicePtr->mTaskEntries;

#if defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)

        TaskEntries::reverse_iterator iter;
        for (iter=taskEntries.rbegin(); iter != taskEntries.rend(); ++iter)
        {
            (*iter).stop(wait);
        }

#else

        std::for_each(
            taskEntries.rbegin(),
            taskEntries.rend(),
            boost::bind( (Pfn) &TaskEntry::stop, _1, wait));

#endif

    }

    void RcfServer::stop()
    {
        bool wait = true;

        Lock lock(mStartStopMutex);
        if (mStarted)
        {
            mStarted = false;

            // set stop flag
            mServerThreadsStopFlag = true;

            // WriteLock here, so that we can flush out any threads in cycle().
            WriteLock lock(mServicesMutex);

            // notify and optionally join all internal worker threads
            std::for_each(
                mServices.rbegin(),
                mServices.rend(),
                boost::bind( &RcfServer::stopService, boost::cref(*this), _1, wait));

            if (wait)
            {
                // join all external worker threads
                std::for_each(
                    mJoinFunctors.rbegin(),
                    mJoinFunctors.rend(),
                    boost::bind(&JoinFunctor::operator(), _1));

                mJoinFunctors.clear();

                // notify all services
                std::for_each(
                    mServices.rbegin(),
                    mServices.rend(),
                    boost::bind(&I_Service::onServerStop, _1, boost::ref(*this)));

                // Reset all muxers.
                if (mThreadPoolPtr)
                {
                    mThreadPoolPtr->resetMuxers();
                }

                std::for_each(
                    mServices.rbegin(),
                    mServices.rend(),
                    boost::bind(&I_Service::resetMuxers, _1));

                // clear stop flag, since all the threads have been joined
                mServerThreadsStopFlag = false;

                // notify anyone who was waiting on the stop event
                mStopEvent.notify_all();
            }
        }
    }

#ifdef RCF_MULTI_THREADED

    void RcfServer::waitForStopEvent()
    {
        Lock lock(mStartStopMutex);
        if (mStarted)
        {
            mStopEvent.wait(lock);
        }
    }

    void RcfServer::waitForStartEvent()
    {
        Lock lock(mStartStopMutex);
        if (!mStarted)
        {
            mStartEvent.wait(lock);
        }
    }   

#endif

    bool RcfServer::isStarted()
    {
        return mStarted;
    }

    SessionPtr RcfServer::createSession()
    {
        RcfSessionPtr rcfSessionPtr(new RcfSession(*this));

        rcfSessionPtr->setWeakThisPtr();

        {
            Lock lock(mSessionsMutex);
            mSessions.insert(rcfSessionPtr);
        }

        return rcfSessionPtr;
    }

    void RcfServer::registerSession(
        RcfSessionPtr rcfSessionPtr)
    {
        Lock lock(mSessionsMutex);
        mSessions.insert( RcfSessionWeakPtr(rcfSessionPtr));
    }

    void RcfServer::unregisterSession(
        RcfSessionWeakPtr rcfSessionWeakPtr)
    {
        Lock lock(mSessionsMutex);

        std::set<RcfSessionWeakPtr>::iterator iter =
            mSessions.find(rcfSessionWeakPtr);

        RCF_ASSERT(iter != mSessions.end());

        mSessions.erase(iter);
    }

    void RcfServer::onReadCompleted(SessionPtr sessionPtr)
    {
        RcfSessionPtr rcfSessionPtr = sessionPtr;

        // Need a recursion limiter here. When processing many sequential oneway
        // calls, over a caching transport filter such as the zlib filter, we 
        // would otherwise be at risk of encountering unlimited recursion and 
        // eventual stack overflow.

        RecursionState<int, int> & recursionState = 
            getCurrentRcfSessionRecursionState();

        applyRecursionLimiter(
            recursionState, 
            &RcfSession::onReadCompleted, 
            *rcfSessionPtr);

        //rcfSessionPtr->onReadCompleted();
    }

    void RcfSession::onReadCompleted()
    {
        // 1. Deserialize request data
        // 2. Store request data in session
        // 3. Move session to corresponding queue

        Lock lock(mStopCallInProgressMutex);
        if (!mStopCallInProgress)
        {
            ByteBuffer readByteBuffer = getSessionState().getReadByteBuffer();

            RCF_LOG_3()(this)(readByteBuffer.getLength()) 
                << "RcfServer - received packet from transport.";

            ByteBuffer messageBody;

            bool ok = mRequest.decodeRequest(
                readByteBuffer,
                messageBody,
                shared_from_this(),
                mRcfServer);

            RCF_LOG_3()(this)(mRequest) 
                << "RcfServer - received request.";

            // Setup the in stream for this remote call.
            mIn.reset(
                messageBody, 
                mRequest.mSerializationProtocol, 
                mRuntimeVersion, 
                mArchiveVersion);

            messageBody.clear();
            
            readByteBuffer.clear();

            if (!ok)
            {
                // version mismatch (client is newer than we are)
                // send a control message back to client, with our runtime version

                if (mRequest.mOneway)
                {
                    mIn.clearByteBuffer();
                    onWriteCompleted();
                }
                else
                {
                    std::vector<ByteBuffer> byteBuffers(1);

                    encodeServerError(
                        mRcfServer,
                        byteBuffers.front(),
                        RcfError_VersionMismatch, 
                        mRcfServer.getRuntimeVersion(),
                        mRcfServer.getArchiveVersion());

                    getSessionState().postWrite(byteBuffers);
                }
            }
            else
            {
                if (mRequest.getClose()) 
                {
                    getSessionState().postClose();
                }
                else
                {
                    // TODO: downside of calling processRequest() now is that
                    // the stack might already be quite deep. Might be better
                    // to unwind the stack first and then call handleSession().
                    processRequest();
                }
            }
        }
    }

    void RcfServer::onWriteCompleted(SessionPtr sessionPtr)
    {
        RcfSessionPtr rcfSessionPtr = sessionPtr;
        rcfSessionPtr->onWriteCompleted();
    }

    void RcfSession::onWriteCompleted()
    {
        RCF_LOG_3()(this) << "RcfServer - completed sending of response.";

        {
            Lock lock(mIoStateMutex);

            if (mWritingPingBack)
            {
                mWritingPingBack = false;

                typedef std::vector<ByteBuffer> ByteBuffers;
                ThreadLocalCached< ByteBuffers > tlcQueuedBuffers;
                ByteBuffers & queuedBuffers = tlcQueuedBuffers.get();

                queuedBuffers = mQueuedSendBuffers;
                mQueuedSendBuffers.clear();
                if (!queuedBuffers.empty())
                {
                    lock.unlock();
                    getSessionState().postWrite(queuedBuffers);
                    RCF_ASSERT(queuedBuffers.empty());
                }

                return;
            }
        }

        typedef std::vector<RcfSession::OnWriteCompletedCallback> OnWriteCompletedCallbacks;
        ThreadLocalCached< OnWriteCompletedCallbacks > tlcOwcc;
        OnWriteCompletedCallbacks &onWriteCompletedCallbacks = tlcOwcc.get();
        
        extractOnWriteCompletedCallbacks(onWriteCompletedCallbacks);

        std::for_each(
            onWriteCompletedCallbacks.begin(),
            onWriteCompletedCallbacks.end(),
            boost::bind(
                &RcfSession::OnWriteCompletedCallback::operator(),
                _1,
                boost::ref(*this)));

        onWriteCompletedCallbacks.resize(0);

        mIn.clear();
        mOut.clear();

        if (!mCloseSessionAfterWrite)
        {
            getSessionState().postRead();
        }        
    }

    void RcfSession::sendSessionResponse()
    {
        mIn.clearByteBuffer();

        ThreadLocalCached< std::vector<ByteBuffer> > tlcByteBuffers;
        std::vector<ByteBuffer> &byteBuffers = tlcByteBuffers.get();

        mOut.extractByteBuffers(byteBuffers);
        const std::vector<FilterPtr> &filters = mFilters;
        ThreadLocalCached< std::vector<ByteBuffer> > tlcEncodedByteBuffers;
        std::vector<ByteBuffer> &encodedByteBuffers = tlcEncodedByteBuffers.get();

        ThreadLocalCached< std::vector<FilterPtr> > tlcNoFilters;
        std::vector<FilterPtr> &noFilters = tlcNoFilters.get();

        mRequest.encodeToMessage(
            encodedByteBuffers, 
            byteBuffers, 
            mFiltered ? filters : noFilters);

        RCF_LOG_3()(this)(lengthByteBuffers(byteBuffers))(lengthByteBuffers(encodedByteBuffers))
            << "RcfServer - sending response.";

        byteBuffers.resize(0);

        bool okToWrite = false;
        {
            Lock lock(mIoStateMutex);
            unregisterForPingBacks();
            if (mWritingPingBack)
            {
                mQueuedSendBuffers = encodedByteBuffers;
                encodedByteBuffers.resize(0);
                byteBuffers.resize(0);
            }
            else
            {
                okToWrite = true;
            }
        }

        if (okToWrite)
        {
            getSessionState().postWrite(encodedByteBuffers);
            RCF_ASSERT(encodedByteBuffers.empty());
            RCF_ASSERT(byteBuffers.empty());
        }

        setCurrentRcfSessionPtr();
    }

    void RcfSession::sendResponseUncaughtException()
    {
        RCF_LOG_3() << "RcfServer - non-std::exception-derived exception was thrown. Sending an error response.";
        sendResponseException( RemoteException(_RcfError_NonStdException()));
    }

    void RcfSession::encodeRemoteException(
        SerializationProtocolOut & out, 
        const RemoteException & e)
    {
        ByteBuffer buffer;
        bool shouldSerializeException = mRequest.encodeResponse(&e, buffer);

        mOut.reset(
            mRequest.mSerializationProtocol, 
            32, 
            buffer, 
            mRuntimeVersion, 
            mArchiveVersion);

        if (shouldSerializeException)
        {
            if (
                out.getSerializationProtocol() == Sp_BsBinary 
                || out.getSerializationProtocol() == Sp_BsText 
                || out.getSerializationProtocol() == Sp_BsXml)
            {
                int runtimeVersion = mRequest.mRuntimeVersion;
                if (runtimeVersion < 8)
                {
                    // Boost serialization is very picky about serializing pointers 
                    // vs values. Since the client will deserialize an auto_ptr, we 
                    // are forced to create an auto_ptr here as well.

                    std::auto_ptr<RemoteException> apRe( 
                        static_cast<RemoteException *>(e.clone().release()) );

                    serialize(out, apRe);
                }
                else
                {
                    const RCF::RemoteException * pRe = &e;
                    serialize(out, pRe);
                }
            }
            else
            {
                // SF is a bit more flexible.
                serialize(out, e);
            }
        }
    }

    void RcfSession::sendResponse()
    {
        bool exceptionalResponse = false;
        try
        {
            ByteBuffer buffer;
            mRequest.encodeResponse(NULL, buffer);

            mOut.reset(
                mRequest.mSerializationProtocol, 
                32, 
                buffer, 
                mRuntimeVersion, 
                mArchiveVersion);

            mpParameters->write(mOut);
            clearParameters();
        }
        catch(const std::exception &e)
        {
            sendResponseException(e);
            exceptionalResponse = true;
        }

        if (!exceptionalResponse)
        {
            sendSessionResponse();
        }
    }

    void RcfSession::sendResponseException(
        const std::exception &e)
    {
        clearParameters();

        const SerializationException *pSE =
            dynamic_cast<const SerializationException *>(&e);

        const RemoteException *pRE =
            dynamic_cast<const RemoteException *>(&e);

        const Exception *pE =
            dynamic_cast<const Exception *>(&e);

        if (pSE)
        {
            RCF_LOG_1()(typeid(*pSE))(*pSE) << "Encoding SerializationException.";
            encodeRemoteException(
                mOut,
                RemoteException(
                    Error( pSE->getError() ),
                    pSE->what(),
                    pSE->getContext(),
                    typeid(*pSE).name()));
        }
        else if (pRE)
        {
            RCF_LOG_1()(typeid(*pRE))(*pRE) << "Encoding RemoteException.";
            try
            {
                encodeRemoteException(mOut, *pRE);
            }
            catch(const RCF::Exception &e)
            {
                encodeRemoteException(
                    mOut,
                    RemoteException(
                        _RcfError_Serialization(typeid(*pRE).name(), typeid(e).name(), e.getError().getErrorString()),
                        e.getWhat(),
                        e.getContext(),
                        typeid(e).name()));
            }
            catch(const std::exception &e)
            {
                encodeRemoteException(
                    mOut,
                    RemoteException(
                        _RcfError_Serialization(typeid(*pRE).name(), typeid(e).name(), e.what()),
                        e.what(),
                        "",
                        typeid(e).name()));
            }
        }
        else if (pE)
        {
            RCF_LOG_1()(typeid(*pE))(*pE) << "Encoding Exception.";
            encodeRemoteException(
                mOut,
                RemoteException(
                    Error( pE->getError() ),
                    pE->getSubSystemError(),
                    pE->getSubSystem(),
                    pE->what(),
                    pE->getContext(),
                    typeid(*pE).name()));
        }
        else
        {
            RCF_LOG_1()(typeid(e))(e) << "Encoding std::exception.";
            encodeRemoteException(
                mOut,
                RemoteException(
                    _RcfError_UserModeException(typeid(e).name(), e.what()),
                    e.what(),
                    "",
                    typeid(e).name()));
        }

        sendSessionResponse();
    }

    class SessionTouch 
    {
    public:
        SessionTouch(RcfSession &rcfSession) : mRcfSession(rcfSession)
        {
            mRcfSession.touch();
        }
        ~SessionTouch()
        {
            mRcfSession.touch();
        }

    private:
        RcfSession & mRcfSession;
    };

    class StubEntryTouch 
    {
    public:
        StubEntryTouch(StubEntryPtr stubEntry) : mStubEntry(stubEntry)
        {
            if (mStubEntry)
            {
                mStubEntry->touch();
            }
        }
        ~StubEntryTouch()
        {
            if (mStubEntry)
            {
                mStubEntry->touch();
            }
        }

    private:
        StubEntryPtr mStubEntry;
    };

    void RcfSession::processRequest()
    {
        MethodInvocationRequest &request = mRequest;

        CurrentRcfSessionSentry guard(*this);

        StubEntryPtr stubEntryPtr = request.locateStubEntryPtr(mRcfServer);

        // NB: the following scopeguard is apparently not triggered by 
        // Borland C++, when throwing non std::exception derived exceptions.

        using namespace boost::multi_index::detail;

        scope_guard sendResponseUncaughtExceptionGuard =
            make_obj_guard(
                *this,
                &RcfSession::sendResponseUncaughtException);

        try
        {
            mAutoSend = true;

            if (NULL == stubEntryPtr.get() && request.getFnId() != -1)
            {
                Exception e( _RcfError_NoServerStub(
                    request.getService(), 
                    request.getSubInterface(),
                    request.getFnId()));

                RCF_THROW(e)(request.getFnId());
            }
            else
            {
                setCachedStubEntryPtr(stubEntryPtr);

                SessionTouch sessionTouch(*this);

                StubEntryTouch stubEntryTouch(stubEntryPtr);

                if (request.getFnId() == -1)
                {
                    // Function id -1 is a canned ping request. We set a
                    // timestamp on the current session and return immediately.

                    AllocateServerParameters<Void>()(*this);

                    setPingTimestamp();
                }
                else
                {
                    registerForPingBacks();

                    ThreadInfoPtr threadInfoPtr = getThreadInfoPtr();
                    if (threadInfoPtr)
                    {
                        threadInfoPtr->notifyBusy();
                    }

                    stubEntryPtr->getRcfClientPtr()->getServerStub().invoke(
                        request.getSubInterface(),
                        request.getFnId(),
                        *this);
                }
                
                sendResponseUncaughtExceptionGuard.dismiss();
                if (mAutoSend && !mRequest.mOneway)
                {
                    sendResponse();
                }
                else if (mRequest.mOneway)
                {
                    RCF_ASSERT(mAutoSend);
                    RCF_LOG_3()(this) << "RcfServer - suppressing response to oneway call.";
                    mIn.clearByteBuffer();
                    clearParameters();
                    setCurrentRcfSessionPtr();
                    onWriteCompleted();
                }
            }
        }
        catch(const std::exception &e)
        {
            sendResponseUncaughtExceptionGuard.dismiss();
            if (mAutoSend && !mRequest.mOneway)
            {
                sendResponseException(e);
            }
            else
            {
                mIn.clearByteBuffer();
                clearParameters();
                setCurrentRcfSessionPtr();
                onWriteCompleted();
            }
        }
    }

    void RcfServer::cycleSessions(
        int,                       // timeoutMs, 
        const volatile bool &)     // stopFlag
    {
        //if (mThreadSpecificSessionQueuePtr.get() == NULL)
        //{
        //    mThreadSpecificSessionQueuePtr.reset(new SessionQueue);
        //}

        //while (!stopFlag && !mThreadSpecificSessionQueuePtr->empty())
        //{
        //    RcfSessionPtr sessionPtr = mThreadSpecificSessionQueuePtr->back();
        //    mThreadSpecificSessionQueuePtr->pop_back();
        //    handleSession(sessionPtr);
        //}
    }

    I_ServerTransport &RcfServer::getServerTransport()
    {
        return *getServerTransportPtr();
    }

    I_Service &RcfServer::getServerTransportService()
    {
        return dynamic_cast<I_Service &>(*getServerTransportPtr());
    }

    ServerTransportPtr RcfServer::getServerTransportPtr()
    {
        RCF_ASSERT( ! mServerTransports.empty() );
        return mServerTransports[0];
    }

    I_IpServerTransport &RcfServer::getIpServerTransport()
    {
        return dynamic_cast<RCF::I_IpServerTransport &>(getServerTransport());
    }    

    bool RcfServer::bindShared(
        const std::string &name,
        RcfClientPtr rcfClientPtr)
    {
        RCF_ASSERT(rcfClientPtr.get());
        RCF_LOG_2()(name) << "RcfServer - exposing static binding.";

        WriteLock writeLock(mStubMapMutex);
        mStubMap[name] = StubEntryPtr( new StubEntry(rcfClientPtr));
        return true;
    }

    FilterPtr RcfServer::createFilter(int filterId)
    {
        ReadLock lock(mServicesMutex);
        if (mFilterServicePtr)
        {
            FilterFactoryPtr filterFactoryPtr = 
                mFilterServicePtr->getFilterFactoryPtr(filterId);

            if (filterFactoryPtr)
            {
                return filterFactoryPtr->createFilter();
            }
        }
        return FilterPtr();
    }

    void RcfServer::setStartCallback(const StartCallback &startCallback)
    {
        mStartCallback = startCallback;
    }

    void RcfServer::invokeStartCallback()
    {
        if (mStartCallback)
        {
            mStartCallback(*this);
        }
    }

    bool RcfServer::getStopFlag() const
    {
        return mServerThreadsStopFlag;
    }

    boost::uint32_t RcfServer::getRuntimeVersion()
    {
        return mRuntimeVersion;
    }

    void RcfServer::setRuntimeVersion(boost::uint32_t version)
    {
        mRuntimeVersion = version;
    }

    boost::uint32_t RcfServer::getArchiveVersion()
    {
        return mArchiveVersion;
    }

    void RcfServer::setArchiveVersion(boost::uint32_t version)
    {
        mArchiveVersion = version;
    }

    PingBackServicePtr RcfServer::getPingBackServicePtr()
    {
        ReadLock lock(mServicesMutex);
        return mPingBackServicePtr;
    }

    FileTransferServicePtr RcfServer::getFileTransferServicePtr()
    {
        ReadLock lock(mServicesMutex);
        return mFileTransferServicePtr;
    }

    ObjectFactoryServicePtr RcfServer::getObjectFactoryServicePtr()
    {
        ReadLock lock(mServicesMutex);
        return mObjectFactoryServicePtr;
    }

    SessionTimeoutServicePtr RcfServer::getSessionTimeoutServicePtr()
    {
        ReadLock lock(mServicesMutex);
        return mSessionTimeoutServicePtr;
    }

    void RcfServer::setThreadPool(ThreadPoolPtr threadPoolPtr)
    {
        if (threadPoolPtr->getThreadName().empty())
        {
            threadPoolPtr->setThreadName("RCF Server");
        }
        mThreadPoolPtr = threadPoolPtr;
    }

    ThreadPoolPtr RcfServer::getThreadPool()
    {
        return mThreadPoolPtr;
    }

    I_ServerTransport & RcfServer::addEndpoint(const RCF::I_Endpoint & endpoint)
    {
        ServerTransportPtr transportPtr(endpoint.createServerTransport().release());
        addServerTransport(transportPtr);
        return *transportPtr;
    }

} // namespace RCF
