
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

#include <RCF/ClientStub.hpp>

#include <boost/bind.hpp>

#include <RCF/ClientProgress.hpp>
#include <RCF/ClientTransport.hpp>
#include <RCF/FileIoThreadPool.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/IpClientTransport.hpp>
#include <RCF/Marshal.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/Version.hpp>

RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::ClientStub)
RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::RemoteCallSemantics)
RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::SerializationProtocol)
RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::EndpointPtr)

namespace RCF {

    //****************************************
    // ClientStub

    // 2s default connect timeout
    static unsigned int gClientConnectTimeoutMs = 1000*2;

    // 10s default call timeout
    static unsigned int gClientRemoteCallTimeoutMs = 1000*10;
    
    void setDefaultConnectTimeoutMs(unsigned int connectTimeoutMs)
    {
        gClientConnectTimeoutMs = connectTimeoutMs;
    }

    unsigned int getDefaultConnectTimeoutMs()
    {
        return gClientConnectTimeoutMs;
    }

    void setDefaultRemoteCallTimeoutMs(unsigned int remoteCallTimeoutMs)
    {
        gClientRemoteCallTimeoutMs = remoteCallTimeoutMs;
    }

    unsigned int getDefaultRemoteCallTimeoutMs()
    {
        return gClientRemoteCallTimeoutMs;
    }

    // Default wstring serialization is UTF-8 encoded.
    static bool gUseNativeWstringSerialization = false;

    void setDefaultNativeWstringSerialization(bool enable)
    {
        gUseNativeWstringSerialization = enable;
    }

    bool getDefaultNativeWstringSerialization()
    {
        return gUseNativeWstringSerialization;
    }

    void ClientStub::setAutoReconnect(bool autoReconnect)
    {
        mAutoReconnect = autoReconnect;
    }

    bool ClientStub::getAutoReconnect() const
    {
        return mAutoReconnect;
    }

    void ClientStub::setClientProgressPtr(ClientProgressPtr ClientProgressPtr)
    {
        mClientProgressPtr = ClientProgressPtr;
    }

    ClientProgressPtr ClientStub::getClientProgressPtr() const
    {
        return mClientProgressPtr;
    }

    static const boost::uint32_t DefaultBatchMaxMessageLimit = 1024*1024;

    ClientStub::ClientStub(const std::string &interfaceName) :
        mToken(),
        mDefaultCallingSemantics(Twoway),
        mProtocol(DefaultSerializationProtocol),
        mMarshalingProtocol(DefaultMarshalingProtocol),
        mEndpointName(),
        mObjectName(),
        mInterfaceName(interfaceName),
        mRemoteCallTimeoutMs(gClientRemoteCallTimeoutMs),
        mConnectTimeoutMs(gClientConnectTimeoutMs),
        mAutoReconnect(true),
        mConnected(RCF_DEFAULT_INIT),
        mTries(RCF_DEFAULT_INIT),
        mAutoVersioning(true),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion()),
        mUseNativeWstringSerialization(RCF::getDefaultNativeWstringSerialization()),

        mAsync(RCF_DEFAULT_INIT),
        mAsyncTimerReason(None),
        mEndTimeMs(RCF_DEFAULT_INIT),
        mRetry(RCF_DEFAULT_INIT),
        mRcs(Twoway),
        mEncodedByteBuffer(),
        mEncodedByteBuffers(),
        mpParameters(RCF_DEFAULT_INIT),
        mPingBackIntervalMs(RCF_DEFAULT_INIT),
        mPingBackTimeStamp(RCF_DEFAULT_INIT),
        mPingBackCount(RCF_DEFAULT_INIT),
        mNextTimerCallbackMs(RCF_DEFAULT_INIT),
        mNextPingBackCheckMs(RCF_DEFAULT_INIT),
        mPingBackCheckIntervalMs(RCF_DEFAULT_INIT),
        mTimerIntervalMs(RCF_DEFAULT_INIT),

        mSignalled(RCF_DEFAULT_INIT),

        mBatchMode(false),
        mBatchMaxMessageLength(DefaultBatchMaxMessageLimit),
        mBatchCount(0),
        mBatchMessageCount(0),

        mTransferWindowS(5)
    {
    }

    ClientStub::ClientStub(const std::string &interfaceName, const std::string &objectName) :
        mToken(),
        mDefaultCallingSemantics(Twoway),
        mProtocol(DefaultSerializationProtocol),
        mMarshalingProtocol(DefaultMarshalingProtocol),
        mEndpointName(),
        mObjectName(objectName),
        mInterfaceName(interfaceName),
        mRemoteCallTimeoutMs(gClientRemoteCallTimeoutMs),
        mConnectTimeoutMs(gClientConnectTimeoutMs),
        mAutoReconnect(true),
        mConnected(RCF_DEFAULT_INIT),
        mTries(RCF_DEFAULT_INIT),
        mAutoVersioning(true),
        mRuntimeVersion(RCF::getDefaultRuntimeVersion()),
        mArchiveVersion(RCF::getDefaultArchiveVersion()),
        mUseNativeWstringSerialization(RCF::getDefaultNativeWstringSerialization()),
        
        mAsync(RCF_DEFAULT_INIT),
        mAsyncTimerReason(None),
        mEndTimeMs(RCF_DEFAULT_INIT),
        mRetry(RCF_DEFAULT_INIT),
        mRcs(Twoway),
        mEncodedByteBuffer(),
        mEncodedByteBuffers(),
        mpParameters(RCF_DEFAULT_INIT),
        mPingBackIntervalMs(RCF_DEFAULT_INIT),
        mPingBackTimeStamp(RCF_DEFAULT_INIT),
        mPingBackCount(RCF_DEFAULT_INIT),
        mNextTimerCallbackMs(RCF_DEFAULT_INIT),
        mNextPingBackCheckMs(RCF_DEFAULT_INIT),
        mPingBackCheckIntervalMs(RCF_DEFAULT_INIT),
        mTimerIntervalMs(RCF_DEFAULT_INIT),

        mSignalled(RCF_DEFAULT_INIT),

        mBatchMode(false),
        mBatchMaxMessageLength(DefaultBatchMaxMessageLimit),
        mBatchCount(0),
        mBatchMessageCount(0),

        mTransferWindowS(5)
    {
    }

    ClientStub::ClientStub(const ClientStub &rhs) :
        mToken(rhs.mToken),
        mDefaultCallingSemantics(rhs.mDefaultCallingSemantics),
        mProtocol(rhs.mProtocol),
        mMarshalingProtocol(DefaultMarshalingProtocol),
        mEndpointName(rhs.mEndpointName),
        mObjectName(rhs.mObjectName),
        mInterfaceName(rhs.mInterfaceName),
        mRemoteCallTimeoutMs(rhs.mRemoteCallTimeoutMs),
        mConnectTimeoutMs(rhs.mConnectTimeoutMs),
        mAutoReconnect(rhs.mAutoReconnect),
        mConnected(RCF_DEFAULT_INIT),
        mTries(RCF_DEFAULT_INIT),
        mAutoVersioning(rhs.mAutoVersioning),
        mRuntimeVersion(rhs.mRuntimeVersion),
        mArchiveVersion(rhs.mArchiveVersion),
        mUseNativeWstringSerialization(rhs.mUseNativeWstringSerialization),
        mUserData(rhs.mUserData),
        
        mAsync(RCF_DEFAULT_INIT),
        mAsyncTimerReason(None),
        mEndTimeMs(RCF_DEFAULT_INIT),
        mRetry(RCF_DEFAULT_INIT),
        mRcs(Twoway),
        mEncodedByteBuffer(),
        mEncodedByteBuffers(),
        mpParameters(RCF_DEFAULT_INIT),
        mPingBackIntervalMs(rhs.mPingBackIntervalMs),
        mPingBackTimeStamp(RCF_DEFAULT_INIT),
        mPingBackCount(RCF_DEFAULT_INIT),
        mNextTimerCallbackMs(RCF_DEFAULT_INIT),
        mNextPingBackCheckMs(RCF_DEFAULT_INIT),
        mPingBackCheckIntervalMs(RCF_DEFAULT_INIT),
        mTimerIntervalMs(RCF_DEFAULT_INIT),

        mSignalled(RCF_DEFAULT_INIT),

        mBatchMode(false),
        mBatchMaxMessageLength(DefaultBatchMaxMessageLimit),
        mBatchCount(0),
        mBatchMessageCount(0)
        

#ifdef RCF_USE_BOOST_FILESYSTEM
        ,mFileProgressCb(rhs.mFileProgressCb)
        ,mTransferWindowS(rhs.mTransferWindowS)
#endif

    {
        setEndpoint( rhs.getEndpoint() );
        if (rhs.mClientProgressPtr)
        {
            mClientProgressPtr.reset(
                new ClientProgress(*rhs.mClientProgressPtr));
        }
    }

    ClientStub::~ClientStub()
    {
        disconnect();
        clearParameters();        
    }

    void ClientStub::clearParameters()
    {
        if (mpParameters)
        {
            CurrentClientStubSentry sentry(*this);

            //mpParameters->~I_Parameters();

            // need to be elaborate here, for borland compiler
            typedef I_Parameters P;
            P &p = *mpParameters;
            p.~P();
            mpParameters = NULL;
        }
    }

    ClientStub &ClientStub::operator=( const ClientStub &rhs )
    {
        if (&rhs != this)
        {
            mInterfaceName              = rhs.mInterfaceName;
            mToken                      = rhs.mToken;
            mDefaultCallingSemantics    = rhs.mDefaultCallingSemantics;
            mProtocol                   = rhs.mProtocol;
            mMarshalingProtocol         = rhs.mMarshalingProtocol;
            mEndpointName               = rhs.mEndpointName;
            mObjectName                 = rhs.mObjectName;
            mRemoteCallTimeoutMs        = rhs.mRemoteCallTimeoutMs;
            mConnectTimeoutMs           = rhs.mConnectTimeoutMs;
            mAutoReconnect              = rhs.mAutoReconnect;
            mConnected                  = false;
            mAutoVersioning             = rhs.mAutoVersioning;
            mRuntimeVersion             = rhs.mRuntimeVersion;
            mArchiveVersion             = rhs.mArchiveVersion;
            mUseNativeWstringSerialization = rhs.mUseNativeWstringSerialization;
            mUserData                   = rhs.mUserData;
            mPingBackIntervalMs         = rhs.mPingBackIntervalMs;
            mSignalled                  = false;

            setEndpoint( rhs.getEndpoint());

            if (rhs.mClientProgressPtr)
            {
                mClientProgressPtr.reset(
                    new ClientProgress(*rhs.mClientProgressPtr));
            }

#ifdef RCF_USE_BOOST_FILESYSTEM
            mFileProgressCb = rhs.mFileProgressCb;
            mTransferWindowS = rhs.mTransferWindowS;
#endif

        }
        return *this;
    }

    Token ClientStub::getTargetToken() const
    {
        return mToken;
    }

    void ClientStub::setTargetToken(Token token)
    {
        mToken = token;
    }

    std::string ClientStub::getTargetName() const
    {
        return mObjectName;
    }

    void ClientStub::setTargetName(const std::string &objectName)
    {
        mObjectName = objectName;
    }

    RemoteCallSemantics ClientStub::getDefaultCallingSemantics() const
    {
        return mDefaultCallingSemantics;
    }

    void ClientStub::setDefaultCallingSemantics(
        RemoteCallSemantics defaultCallingSemantics)
    {
        mDefaultCallingSemantics = defaultCallingSemantics;
    }

    void ClientStub::setSerializationProtocol(SerializationProtocol  protocol)
    {
        mProtocol = protocol;
    }

    SerializationProtocol ClientStub::getSerializationProtocol() const
    {
        return mProtocol;
    }



    void ClientStub::setMarshalingProtocol(MarshalingProtocol  protocol)
    {
        mMarshalingProtocol = protocol;
    }

    MarshalingProtocol ClientStub::getMarshalingProtocol() const
    {
        return mMarshalingProtocol;
    }

    bool ClientStub::getNativeWstringSerialization()    
    {
        return mUseNativeWstringSerialization;
    }

    void ClientStub::setNativeWstringSerialization(bool useNativeWstringSerialization)
    {
        mUseNativeWstringSerialization = useNativeWstringSerialization;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void ClientStub::enableSfSerializationPointerTracking(bool enable)
    {
        mOut.mOutProtocol1.setCustomizationCallback(
            boost::bind(enableSfPointerTracking_1, _1, enable) );

        //mOut.mOutProtocol2.setCustomizationCallback(
        //    boost::bind(enableSfPointerTracking_2, _1, enable) );
    }

#else

    void ClientStub::enableSfSerializationPointerTracking(bool enable)
    {}

#endif

    void ClientStub::setEndpoint(const I_Endpoint &endpoint)
    {
        mEndpoint = endpoint.clone();
    }

    void ClientStub::setEndpoint(EndpointPtr endpointPtr)
    {
        mEndpoint = endpointPtr;
    }

    EndpointPtr ClientStub::getEndpoint() const
    {
        return mEndpoint;
    }

    void ClientStub::setTransport(std::auto_ptr<I_ClientTransport> transport)
    {
        mTransport = transport;
        mConnected = mTransport.get() && mTransport->isConnected();
    }

    std::auto_ptr<I_ClientTransport> ClientStub::releaseTransport()
    {
        instantiateTransport();
        return mTransport;
    }

    I_ClientTransport& ClientStub::getTransport()
    {
        instantiateTransport();
        return *mTransport;
    }

    I_IpClientTransport &ClientStub::getIpTransport()
    {
        return dynamic_cast<I_IpClientTransport &>(getTransport());
    }

    void ClientStub::instantiateTransport()
    {
        if (!mTransport.get())
        {
            RCF_VERIFY(mEndpoint.get(), Exception(_RcfError_NoEndpoint()));
            mTransport.reset( mEndpoint->createClientTransport().release() );
            RCF_VERIFY(mTransport.get(), Exception(_RcfError_TransportCreation()));
        }
    }

    void ClientStub::disconnect()
    {
        std::string endpoint;
        if (mEndpoint.get())
        {
            endpoint = mEndpoint->asString();
        }

        RcfClientPtr subRcfClientPtr = getSubRcfClientPtr();
        setSubRcfClientPtr( RcfClientPtr() );
        if (subRcfClientPtr)
        {
            subRcfClientPtr->getClientStub().disconnect();
            subRcfClientPtr.reset();
        }

        if (mTransport.get())
        {
            RCF_LOG_2()(this)(endpoint)
                << "RcfClient - disconnecting from server.";

            mTransport->disconnect(mConnectTimeoutMs);
            mConnected = false;
        }

        if (mBatchBufferPtr)
        {
            mBatchBufferPtr->resize(0);
        }

        mAsyncCallback = boost::function0<void>();
    }

    bool ClientStub::isConnected()
    {
        return mTransport.get() && mTransport->isConnected();
    }

    void ClientStub::setConnected(bool connected)
    {
        mConnected = connected;
    }

    void ClientStub::setMessageFilters()
    {
        setMessageFilters( std::vector<FilterPtr>());
    }

    void ClientStub::setMessageFilters(const std::vector<FilterPtr> &filters)
    {
        mMessageFilters.assign(filters.begin(), filters.end());
        RCF::connectFilters(mMessageFilters);
    }

    void ClientStub::setMessageFilters(FilterPtr filterPtr)
    {
        std::vector<FilterPtr> filters;
        filters.push_back(filterPtr);
        setMessageFilters(filters);
    }

    const std::vector<FilterPtr> &ClientStub::getMessageFilters()
    {
        return mMessageFilters;
    }

    void ClientStub::setRemoteCallTimeoutMs(unsigned int remoteCallTimeoutMs)
    {
        mRemoteCallTimeoutMs = remoteCallTimeoutMs;
    }

    unsigned int ClientStub::getRemoteCallTimeoutMs() const
    {
        return mRemoteCallTimeoutMs;
    }

    void ClientStub::setConnectTimeoutMs(unsigned int connectTimeoutMs)
    {
        mConnectTimeoutMs = connectTimeoutMs;
    }

    unsigned int ClientStub::getConnectTimeoutMs() const
    {
        return mConnectTimeoutMs;
    }

    void ClientStub::setAutoVersioning(bool autoVersioning)
    {
        mAutoVersioning = autoVersioning;
    }

    bool ClientStub::getAutoVersioning() const
    {
        return mAutoVersioning;
    }

    void ClientStub::setRuntimeVersion(boost::uint32_t version)
    {
        mRuntimeVersion = version;
    }

    boost::uint32_t ClientStub::getRuntimeVersion() const
    {
        return mRuntimeVersion;
    }

    void ClientStub::setArchiveVersion(boost::uint32_t version)
    {
        mArchiveVersion = version;
    }

    boost::uint32_t ClientStub::getArchiveVersion() const
    {
        return mArchiveVersion;
    }

    void ClientStub::setTries(std::size_t tries)
    {
        mTries = tries;
    }

    std::size_t ClientStub::getTries() const
    {
        return mTries;
    }

    CurrentClientStubSentry::CurrentClientStubSentry(ClientStub & clientStub)
    {
        pushCurrentClientStub(&clientStub);
    }

    CurrentClientStubSentry::~CurrentClientStubSentry()
    {
        popCurrentClientStub();
    }

    void ClientStub::onError(const std::exception &e)
    {
        if (mAsync)
        {
            prepareAmiNotification();
        }

        if (mTransport.get() && mAsyncTimerEntry != TimerEntry())
        {
            mTransport->killTimer(mAsyncTimerEntry);
            mAsyncTimerEntry = TimerEntry();
            mAsyncTimerReason = None;
        }

        const RemoteException *pRcfRE = 
            dynamic_cast<const RemoteException *>(&e);

        const Exception *pRcfE = 
            dynamic_cast<const Exception *>(&e);

        if (pRcfRE)
        {
            mEncodedByteBuffers.resize(0);
            if (shouldDisconnectOnRemoteError( pRcfRE->getError() ))
            {
                disconnect();
            }
            setAsyncException(pRcfRE->clone());
        }
        else if (pRcfE)
        {
            mEncodedByteBuffers.resize(0);
            disconnect();
            setAsyncException(pRcfE->clone());
        }
        else
        {
            mEncodedByteBuffers.resize(0);
            disconnect();

            setAsyncException( std::auto_ptr<Exception>(
                new Exception(e.what())));
        }
    }

    void ClientStub::onTimerExpired()
    {
        TimerReason timerReason = mAsyncTimerReason;
        mAsyncTimerReason = None;

        if (timerReason == Wait)
        {
            prepareAmiNotification();

            if (mTransport.get() && mAsyncTimerEntry != TimerEntry())
            {
                mTransport->killTimer(mAsyncTimerEntry);
                mAsyncTimerEntry = TimerEntry();
            }
        }
        else
        {
            switch(timerReason)
            {
            case Connect:
                RCF_ASSERT(mEndpoint.get());
                
                onError(RCF::Exception(_RcfError_ClientConnectTimeout(
                    mConnectTimeoutMs, 
                    mEndpoint->asString())));

                break;

            case Write:
                onError(RCF::Exception(_RcfError_ClientWriteTimeout()));
                break;

            case Read: 
                onError(RCF::Exception(_RcfError_ClientReadTimeout()));
                break;

            default:
                RCF_ASSERT(0)(timerReason);
            };
        }        
    }

    void ClientStub::setUserData(boost::any userData)
    {
        mUserData = userData;
    }

    boost::any ClientStub::getUserData()
    {
        return mUserData;
    }

    void ClientStub::setInterfaceName(const std::string & interfaceName)
    {
        mInterfaceName = interfaceName;
    }

    std::string ClientStub::getInterfaceName()
    {
        return mInterfaceName;
    }

    SerializationProtocolIn & ClientStub::getSpIn()
    {
        return mIn;
    }

    SerializationProtocolOut & ClientStub::getSpOut()
    {
        return mOut;
    }

    void ClientStub::setPingBackIntervalMs(int pingBackIntervalMs)
    {
        mPingBackIntervalMs = pingBackIntervalMs;
    }
    
    int ClientStub::getPingBackIntervalMs()
    {
        return mPingBackIntervalMs;
    }

    std::size_t ClientStub::getPingBackCount()
    {
        return mPingBackCount;
    }

    boost::uint32_t ClientStub::getPingBackTimeStamp()
    {
        return mPingBackTimeStamp;
    }

    void ClientStub::ping()
    {
        ping( getDefaultCallingSemantics() );
    }

    void ClientStub::ping(RemoteCallSemantics rcs)
    {
        typedef Void V;

        CurrentClientStubSentry sentry(*this);

        AllocateClientParameters<V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V >::ParametersT & parms = 
            AllocateClientParameters<V,V,V,V,V,V,V,V,V,V,V,V,V,V,V,V >()(
                *this, V(), V(), V(), V(), V(), V(), V(), V(), V(), V(), V(), V(), V(), V(), V());

        FutureImpl<V>(
            parms.r.get(),
            *this,
            mInterfaceName,
            -1,
            CallOptions(rcs).apply(*this));
    }   

    // Take the proposed timeout and cut it down to accommodate client progress 
    // callbacks and checking of ping back interval.

    boost::uint32_t ClientStub::generatePollingTimeout(boost::uint32_t timeoutMs)
    {
        boost::uint32_t timeNowMs = Platform::OS::getCurrentTimeMs();

        boost::uint32_t timeToNextTimerCallbackMs = mNextTimerCallbackMs ?
            mNextTimerCallbackMs - timeNowMs:
            -1;

        boost::uint32_t timeToNextPingBackCheckMs = mNextPingBackCheckMs ?
            mNextPingBackCheckMs - timeNowMs:
            -1;

        return 
            RCF_MIN( 
                RCF_MIN(timeToNextTimerCallbackMs, timeToNextPingBackCheckMs), 
                timeoutMs);
    }

    void ClientStub::onPollingTimeout()
    {
        // Check whether we need to fire a client progress timer callback.
        if (mNextTimerCallbackMs && 0 == generateTimeoutMs(mNextTimerCallbackMs))
        {
            ClientProgress::Action action = ClientProgress::Continue;

            mClientProgressPtr->mProgressCallback(
                0,
                0,
                ClientProgress::Timer,
                ClientProgress::Receive,
                action);

            RCF_VERIFY(
                action == ClientProgress::Continue,
                Exception(_RcfError_ClientCancel()))
                (mTimerIntervalMs);

            mNextTimerCallbackMs = 
                Platform::OS::getCurrentTimeMs() + mTimerIntervalMs;

            mNextTimerCallbackMs |= 1;
        }

        // Check that pingbacks have been received.
        if (mNextPingBackCheckMs && 0 == generateTimeoutMs(mNextPingBackCheckMs))
        {
            boost::uint32_t timeNowMs = Platform::OS::getCurrentTimeMs();

            boost::uint32_t timeSinceLastPingBackMs = 
                timeNowMs - mPingBackTimeStamp;

            RCF_VERIFY(
                timeSinceLastPingBackMs < mPingBackCheckIntervalMs,
                Exception(_RcfError_PingBackTimeout(mPingBackCheckIntervalMs))) // TODO: special error for pingbacks
                (mPingBackCheckIntervalMs);

            mNextPingBackCheckMs = 
                Platform::OS::getCurrentTimeMs() + mPingBackCheckIntervalMs;

            mNextPingBackCheckMs |= 1;
        }

    }

    void ClientStub::onUiMessage()
    {
        ClientProgress::Action action = ClientProgress::Continue;

        mClientProgressPtr->mProgressCallback(
            0,
            0,
            ClientProgress::UiMessage,
            ClientProgress::Receive,
            action);

        RCF_VERIFY(
            action != ClientProgress::Cancel,
            Exception(_RcfError_ClientCancel()))
            (mClientProgressPtr->mUiMessageFilter);

        // a sample message filter

        //MSG msg = {0};
        //while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        //{
        //    if (msg.message == WM_QUIT)
        //    {
        //
        //    }
        //    else if (msg.message == WM_PAINT)
        //    {
        //        TranslateMessage(&msg);
        //        DispatchMessage(&msg);
        //    }
        //}

    }

#ifdef RCF_USE_SF_SERIALIZATION

    void ClientStub::serialize(SF::Archive & ar)
    {
        ar  & mToken
            & mDefaultCallingSemantics
            & mProtocol
            & mEndpointName
            & mObjectName
            & mInterfaceName
            & mRemoteCallTimeoutMs
            & mAutoReconnect
            & mEndpoint;
    }

#endif

} // namespace RCF

#ifdef RCF_USE_BOOST_FILESYSTEM

#include <sys/stat.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/operations.hpp>
#include <RCF/FileTransferService.hpp>
#include <RCF/util/Platform/OS/Sleep.hpp>
namespace fs = boost::filesystem;

namespace RCF {

    // FileStream

    FileStream::FileStream() : mImplPtr( new FileStreamImpl() )
    {
    }

    FileStream::FileStream(FileStreamImplPtr implPtr) : 
        mImplPtr(implPtr)
    {
    }

    FileStream::FileStream(const std::string & filePath) : 
        mImplPtr(new FileStreamImpl(filePath) )
    {
    }

    FileStream::FileStream(const FileManifest & manifest) : 
        mImplPtr(new FileStreamImpl(manifest) )
    {
    }

    std::string FileStream::getLocalPath() const
    {
        RCF_ASSERT(mImplPtr);
        FileManifest & manifest = mImplPtr->mManifest;
        fs::path localPath = manifest.mManifestBase;
        RCF_ASSERT(!manifest.mFiles.empty());
        localPath /= (*manifest.mFiles[0].mFilePath.begin());
        return localPath.file_string();
    }

    FileManifest & FileStream::getManifest() const
    {
        RCF_ASSERT(mImplPtr);
        return mImplPtr->mManifest;
    }

    void FileStream::setDownloadPath(const std::string & downloadPath)
    {
        RCF_ASSERT(mImplPtr);
        mImplPtr->mDownloadPath = downloadPath;
    }

    std::string FileStream::getDownloadPath() const
    {
        RCF_ASSERT(mImplPtr);
        return mImplPtr->mDownloadPath.file_string();
    }

    void FileStream::setTransferRateBps(boost::uint32_t transferRateBps)
    {
        RCF_ASSERT(mImplPtr);
        mImplPtr->mTransferRateBps = transferRateBps;
    }

    boost::uint32_t FileStream::getTransferRateBps()
    {
        RCF_ASSERT(mImplPtr);
        return mImplPtr->mTransferRateBps;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void FileStream::serialize(SF::Archive & ar)
    {
        ar & *mImplPtr;
    }

    void FileStreamImpl::serializeImplSf(
        SF::Archive & ar, 
        boost::uint32_t & transferId, 
        Direction & dir)
    {
        ar & transferId & dir;
    }

    void FileStreamImpl::serialize(SF::Archive & ar)
    {
        bool isSaving = ar.isWrite(); 

        serializeGeneric( 
            isSaving,
            boost::bind( 
                &FileStreamImpl::serializeImplSf, 
                this, 
                boost::ref(ar),
                _1, 
                _2) );
    }

#endif

    void FileStream::upload(RCF::ClientStub & clientStub)
    {
        boost::uint32_t chunkSize = 1024*1024;
        clientStub.uploadFiles(mImplPtr->mManifest, mImplPtr->mUploadId, chunkSize, mImplPtr->mSessionLocalId);
        mImplPtr->mSessionLocalId = 0;
    }

    fs::path makeTempDir(const fs::path & basePath, const std::string & prefix);

    void FileStream::download(RCF::ClientStub & clientStub)
    {
        boost::uint32_t chunkSize = 1024*1024;
        
        if (mImplPtr->mDownloadPath.empty())
        {
            mImplPtr->mDownloadPath = makeTempDir("RCF-Downloads", "");
        }

        clientStub.downloadFiles(
            mImplPtr->mDownloadPath.file_string(), 
            mImplPtr->mManifest, 
            chunkSize, 
            mImplPtr->mTransferRateBps,
            mImplPtr->mSessionLocalId);

        mImplPtr->mManifest.mManifestBase = mImplPtr->mDownloadPath;
        mImplPtr->mSessionLocalId = 0;
    }

    // FileStreamImpl

    FileStreamImpl::FileStreamImpl() : 
        mTransferRateBps(0),
        mSessionLocalId(0), 
        mUploadId(0),
        mDirection(Unspecified)
    {
    }

    FileStreamImpl::FileStreamImpl(const std::string & filePath) : 
        mManifest(filePath),
            mTransferRateBps(0),
        mSessionLocalId(0),
        mUploadId(0),
        mDirection(Unspecified)
    {
    }

    FileStreamImpl::FileStreamImpl(const FileManifest & manifest) :
        mManifest(manifest),
        mTransferRateBps(0),
        mSessionLocalId(0),
        mUploadId(0),
        mDirection(Unspecified)
    {
    }

    FileStreamImpl::~FileStreamImpl()
    {
    }

    void FileStreamImpl::serializeGeneric(
        bool isWriting,
        boost::function2<void, boost::uint32_t &, Direction &> serializeImpl)
    {
        // Determine if we are client side or server side.
        // TODO: what if both of these are non-zero?
        RCF::ClientStub * pClientStub = RCF::getCurrentClientStubPtr();
        RCF::RcfSession * pSession = RCF::getCurrentRcfSessionPtr();

        // Client side.
        // Append a ref to ourselves the the current ClientStub.
        // After serializing regular parameters, we'll regain control.
        if (pClientStub)
        {
            if (isWriting)
            {
                if (mDirection == Upload)
                {
                    mSessionLocalId = pClientStub->addUploadStream(
                        FileUpload(shared_from_this()));
                }
                else if (mDirection == Download)
                {
                    mSessionLocalId = pClientStub->addDownloadStream(
                        FileDownload(shared_from_this()));
                }

                serializeImpl(mSessionLocalId, mDirection);
            }
            else
            {
                // Shouldn't really be in here.
                boost::uint32_t sessionLocalId = 0;
                Direction dir = Unspecified;
                serializeImpl(sessionLocalId, dir);
            }
        }
        else if (pSession)
        {
            if (isWriting)
            {
                // Shouldn't really be in here.
                boost::uint32_t sessionLocalId = 0;
                Direction dir = Unspecified;
                serializeImpl(sessionLocalId, dir);
            }
            else
            {
                serializeImpl(mSessionLocalId, mDirection);

                if (mDirection == Upload && mSessionLocalId)
                {
                    RcfSession::SessionUploads::iterator iter = 
                        pSession->mSessionUploads.find(mSessionLocalId);

                    if (iter != pSession->mSessionUploads.end())
                    {
                        FileUploadInfoPtr uploadPtr = iter->second;
                        mManifest = uploadPtr->mManifest;
                        mManifest.mManifestBase = uploadPtr->mUploadPath;

                        pSession->mSessionUploads.erase(iter);
                    }
                    else
                    {
                        // Couldn't find the upload.
                    }
                }
                else if (mDirection == Download && mSessionLocalId)
                {
                    pSession->addDownloadStream(
                        mSessionLocalId,
                        FileDownload(shared_from_this()));
                }
            }
        }
    }

    // FileUpload

    FileUpload::FileUpload()
    {
        mImplPtr->mDirection = FileStreamImpl::Upload;
    }

    FileUpload::FileUpload(const std::string & filePath) : FileStream(filePath)
    {
        mImplPtr->mDirection = FileStreamImpl::Upload;
    }

    FileUpload::FileUpload(const FileManifest & manifest) : FileStream(manifest)
    {
        mImplPtr->mDirection = FileStreamImpl::Upload;
    }

    FileUpload::FileUpload(FileStreamImplPtr implPtr) : FileStream(implPtr)
    {
        mImplPtr->mDirection = FileStreamImpl::Upload;
    }

    // FileDownload

    FileDownload::FileDownload()
    {
        mImplPtr->mDirection = FileStreamImpl::Download;
    }

    FileDownload::FileDownload(const std::string & filePath) : FileStream(filePath)
    {
        mImplPtr->mDirection = FileStreamImpl::Download;
    }

    FileDownload::FileDownload(const FileManifest & manifest) : FileStream(manifest)
    {
        mImplPtr->mDirection = FileStreamImpl::Download;
    }

    FileDownload::FileDownload(FileStreamImplPtr implPtr) : FileStream(implPtr)
    {
        mImplPtr->mDirection = FileStreamImpl::Download;
    }

    // ClientStub

    boost::uint32_t ClientStub::addUploadStream(FileUpload fileStream)
    {
        mUploadStreams.push_back(fileStream);
        return static_cast<boost::uint32_t>(mUploadStreams.size());
    }

    void ClientStub::processUploadStreams()
    {
        std::vector<FileUpload> fileStreams;
        fileStreams.swap(mUploadStreams);
        for (std::size_t i=0; i<fileStreams.size(); ++i)
        {
            fileStreams[i].upload(*this);
        }
    }

    boost::uint32_t ClientStub::addDownloadStream(FileDownload fileStream)
    {
        mDownloadStreams.push_back(fileStream);
        return static_cast<boost::uint32_t>(mDownloadStreams.size());
    }

    void ClientStub::setFileProgressCallback(FileProgressCb fileProgressCb)
    {
        mFileProgressCb = fileProgressCb;
    }

    void ClientStub::setTransferWindowS(boost::uint32_t transferWindowS)
    {
        mTransferWindowS = transferWindowS;
    }

    boost::uint32_t ClientStub::getTransferWindowS()
    {
        return mTransferWindowS;
    }

    // TODO: resuming of failed transfers, in either direction.

    void ClientStub::uploadFiles(
        const std::string & whichFile,
        boost::uint32_t & uploadId,
        boost::uint32_t chunkSize,
        boost::uint32_t sessionLocalId)
    {
        RCF::FileManifest manifest(whichFile);
        uploadFiles(manifest, chunkSize, sessionLocalId);
    }

    void ClientStub::uploadFiles(
        const FileManifest & manifest,
        boost::uint32_t & uploadId,
        boost::uint32_t chunkSize,
        boost::uint32_t sessionLocalId)
    {
        RCF_LOG_3()(manifest.mFiles.size())(chunkSize)(sessionLocalId) 
            << "ClientStub::uploadFiles() - entry.";

        ClientStub & clientStub = *this;

        if (! clientStub.isConnected())
        {
            clientStub.connect();
        }

        RCF::RcfClient<RCF::I_FileTransferService> ftsClient(clientStub);
        ftsClient.getClientStub().setTransport( clientStub.releaseTransport() );
        ftsClient.getClientStub().setTargetToken( Token());

        RestoreClientTransportGuard guard(clientStub, ftsClient.getClientStub());
        RCF_UNUSED_VARIABLE(guard);

        // 1) Send manifest to server, and an optimistic first chunk. 
        // 2) Server replies, with index, pos and CRC of next chunk to transfer.
        // --> CRC only passed if pos != chunk length
        // 3) Client goes into a loop, sending chunks until all files are transferred.

        namespace fs = boost::filesystem;

        fs::path manifestBase = manifest.mManifestBase;

        boost::uint32_t err                 = RcfError_Ok;
        FileChunk startPos;
        boost::uint32_t maxMessageLength    = 0;
        boost::uint32_t bps                 = 0;

        RCF_LOG_3()(manifest.mFiles.size())(chunkSize)(sessionLocalId) 
            << "ClientStub::uploadFiles() - calling BeginUpload().";

        ftsClient.BeginUpload(
            manifest, 
            std::vector<FileChunk>(), 
            startPos, 
            maxMessageLength,
            uploadId,
            bps,
            sessionLocalId);

        RCF_LOG_3()(startPos.mFileIndex)(startPos.mOffset)(maxMessageLength)(uploadId)(bps) 
            << "ClientStub::uploadFiles() - BeginUpload() returned.";
        
        // TODO: error handling
        // ..

        boost::uint64_t totalByteSize = manifest.getTotalByteSize();

        boost::uint64_t totalBytesUploadedSoFar = 0;
        for (std::size_t i=0; i<startPos.mFileIndex; ++i)
        {
            totalBytesUploadedSoFar += manifest.mFiles[i].mFileSize;
        }
        totalBytesUploadedSoFar += startPos.mOffset;

        // Progress callback.
        if (mFileProgressCb)
        {
            mFileProgressCb(totalBytesUploadedSoFar, totalByteSize);
        }
        
        boost::uint32_t firstFile = startPos.mFileIndex;
        boost::uint64_t firstPos = startPos.mOffset;

        // Limit the chunk size to 80 % of max message length.
        chunkSize = RCF_MIN(chunkSize, maxMessageLength*8/10);

        Timer windowTimer;
        boost::uint32_t windowBytesTotal    = mTransferWindowS*bps;
        boost::uint32_t windowBytesSoFar    = 0;

        std::vector<FileChunk> chunks;

        // Async file reading.
        FileIoRequestPtr readOp( new FileIoRequest() );
        ByteBuffer bufferRead(chunkSize);
        ByteBuffer bufferSend(chunkSize);

        FileManifest::Files::iterator iter;
        for (std::size_t i=firstFile; i<manifest.mFiles.size(); ++i)
        {            
            const FileInfo & info = manifest.mFiles[i];
            fs::path filePath = manifestBase / info.mFilePath;

            // Upload chunks to the server until we're done.
            const boost::uint64_t FileSize = fs::file_size(filePath);
            
            RCF_LOG_3()(filePath)
                << "ClientStub::uploadFiles() - opening file.";

            IfstreamPtr fin( new std::ifstream(
                filePath.string().c_str(), 
                std::ios::binary));

            RCF_VERIFY(*fin, Exception(_RcfError_FileOpen(filePath.file_string())));

            boost::uint64_t pos = 0;        
            if (i == firstFile)
            {
                pos = firstPos;

                RCF_LOG_3()(pos)
                    << "ClientStub::uploadFiles() - seeking in file.";

                fin->seekg( static_cast<std::streamoff>(pos) );
            }

            readOp->complete();

            while (pos < FileSize)
            {
                std::size_t bytesRead = 0;

                // Wait for current read to complete.
                if (readOp->initiated())
                {
                    readOp->complete();
                    
                    bytesRead = static_cast<std::size_t>(
                        readOp->getBytesTransferred());

                    RCF_LOG_3()(bytesRead)
                        << "ClientStub::uploadFiles() - completing read from file.";
                    
                    RCF_VERIFY(
                        bytesRead > 0, 
                        Exception(_RcfError_FileRead(filePath.file_string(), pos)));

                    bufferSend.swap(bufferRead);

                    pos += bytesRead;
                }
                
                // Initiate next read.

                RCF_LOG_3()(bufferRead.getLength())
                    << "ClientStub::uploadFiles() - initiating read from file.";

                std::size_t bytesToRead = 
                    static_cast<std::size_t>(bufferRead.getLength());

                // Trim to throttle settings.
                if (bps && windowBytesSoFar < windowBytesTotal)
                {
                    boost::uint32_t windowBytesRemaining = 
                        windowBytesTotal - windowBytesSoFar;

                    RCF_LOG_3()(bytesToRead)(windowBytesRemaining)
                        << "ClientStub::uploadFiles() - trimming chunk size to throttle setting.";

                    bytesToRead = RCF_MIN(
                        static_cast<boost::uint32_t>(bytesToRead), 
                        windowBytesRemaining);
                }

                readOp->read(fin, ByteBuffer(bufferRead, 0, bytesToRead));

                // Upload current buffer.
                if (bytesRead)
                {
                    boost::uint32_t err = RcfError_Ok;

                    FileChunk chunk;
                    chunk.mFileIndex = i;
                    chunk.mOffset = pos - bytesRead;
                    chunk.mData = ByteBuffer(bufferSend, 0, bytesRead);

                    RCF_LOG_3()(chunk.mFileIndex)(chunk.mOffset)(chunk.mData.getLength())
                        << "ClientStub::uploadFiles() - adding chunk.";

                    chunks.clear();
                    chunks.push_back( chunk );

                    // TODO: upload many chunks in one go, to speed up transfer
                    // of many small files.

                    RCF_LOG_3()(chunks.size())
                        << "ClientStub::uploadFiles() - calling UploadChunks().";

                    ftsClient.UploadChunks(chunks, bps);

                    RCF_LOG_3()(bps)
                        << "ClientStub::uploadFiles() - UploadChunks() returned.";

                    totalBytesUploadedSoFar += bytesRead;

                    RCF_VERIFY(static_cast<int>(err) == RcfError_Ok, RemoteException(err));

                    // Progress callback.
                    if (mFileProgressCb)
                    {
                        mFileProgressCb(totalBytesUploadedSoFar, totalByteSize);
                    }

                    if (bps)
                    {
                        // Recalculate window size based on possibly updated bps setting.
                        windowBytesTotal = mTransferWindowS * bps;
                        windowBytesSoFar += bytesRead;
                        if (windowBytesSoFar >= windowBytesTotal)
                        {
                            RCF_LOG_3()(windowBytesSoFar)(windowBytesTotal)
                                << "ClientStub::uploadFiles() - window capacity reached.";

                            // Exceeded window capacity. Wait for window to expire.
                            boost::uint32_t windowMsSoFar = windowTimer.getDurationMs();
                            if (windowMsSoFar < mTransferWindowS*1000)
                            {
                                boost::uint32_t waitMs = mTransferWindowS*1000 - windowMsSoFar;

                                RCF_LOG_3()(waitMs)
                                    << "ClientStub::uploadFiles() - waiting for next window.";

                                Sleep(waitMs);
                                while (!windowTimer.elapsed(mTransferWindowS*1000))
                                {
                                    Sleep(100);
                                }
                            }
                        }

                        // If window has expired, open a new one.
                        if (windowTimer.elapsed(mTransferWindowS*1000))
                        {
                            windowTimer.restart();

                            // Carry over balance from previous window.
                            if (windowBytesSoFar > windowBytesTotal)
                            {
                                windowBytesSoFar = windowBytesSoFar - windowBytesTotal;
                            }
                            else
                            {
                                windowBytesSoFar = 0;
                            }

                            RCF_LOG_3()(mTransferWindowS)(windowBytesSoFar)
                                << "ClientStub::uploadFiles() - new transfer window.";
                        }
                    }
                }
            }
        }

        RCF_LOG_3()(totalByteSize)(totalBytesUploadedSoFar) 
            << "ClientStub::uploadFiles() - exit.";
    }

    void trimManifest(
        const FileManifest & manifest, 
        boost::uint64_t & bytesAlreadyTransferred,
        FileChunk & startPos)
    {
        fs::path manifestBase = manifest.mManifestBase;
        RCF_ASSERT(!manifestBase.empty());

        bytesAlreadyTransferred = 0;
        startPos = FileChunk();

        std::size_t whichFile=0;
        boost::uint64_t offset = 0;

        for (whichFile=0; whichFile<manifest.mFiles.size(); ++whichFile)
        {
            const FileInfo & fileInfo = manifest.mFiles[whichFile];
            fs::path p = fileInfo.mFilePath;
            p = manifestBase / p;

            if (!fs::exists(p) || fs::is_directory(p))
            {
                break;
            }

            boost::uint64_t fileSize = fs::file_size(p);
            
            if (fileSize < fileInfo.mFileSize)
            {
                bytesAlreadyTransferred += fileSize;
                offset = fileSize;
                break;
            }
            
            if (fileSize > fileInfo.mFileSize)
            {
                break;
            }

            RCF_ASSERT_EQ(fileSize , fileInfo.mFileSize);

            bytesAlreadyTransferred += fileSize;
        }

        if (whichFile <= manifest.mFiles.size())
        {
            startPos.mFileIndex = whichFile;
            startPos.mOffset = offset;
        }
    }

    void ClientStub::downloadFiles(
        const std::string & downloadLocation,
        FileManifest & totalManifest,
        boost::uint32_t chunkSize,
        boost::uint32_t transferRateBps,
        boost::uint32_t sessionLocalId)
    {
        RCF_LOG_3()(downloadLocation)(chunkSize)(transferRateBps)(sessionLocalId) 
            << "ClientStub::downloadFiles() - entry.";

        ClientStub & clientStub = *this;

        if (! clientStub.isConnected())
        {
            clientStub.connect();
        }

        RCF::RcfClient<RCF::I_FileTransferService> ftsClient(clientStub);
        ftsClient.getClientStub().setTransport( clientStub.releaseTransport() );
        ftsClient.getClientStub().setTargetToken( Token());

        RestoreClientTransportGuard guard(clientStub, ftsClient.getClientStub());
        RCF_UNUSED_VARIABLE(guard);

        // Download chunks from the server until we're done.

        // TODO: optional first chunks.
        FileManifest manifest;
        FileTransferRequest request;
        std::vector<FileChunk> chunks;
        boost::uint32_t maxMessageLength = 0;

        RCF_LOG_3()(downloadLocation)(chunkSize)(transferRateBps)(sessionLocalId) 
            << "ClientStub::downloadFiles() - calling BeginDownload().";

        ftsClient.BeginDownload(
            manifest, 
            request, 
            chunks, 
            maxMessageLength, 
            sessionLocalId);

        RCF_LOG_3()(manifest.mFiles.size())(maxMessageLength)
            << "ClientStub::downloadFiles() - BeginDownload() returned.";

        chunkSize = RCF_MIN(chunkSize, maxMessageLength*8/10);

        fs::path manifestBase = downloadLocation;

        boost::uint32_t currentFile = 0;
        boost::uint64_t currentPos = 0;

        // See if we have any fragments already downloaded.
        bool resume = false;
        manifest.mManifestBase = downloadLocation;
        boost::uint64_t bytesAlreadyTransferred = 0;
        FileChunk startPos;
        trimManifest(manifest, bytesAlreadyTransferred, startPos);
        if (bytesAlreadyTransferred)
        {
            RCF_LOG_3()(startPos.mFileIndex)(startPos.mOffset)
                << "ClientStub::downloadFiles() - calling TrimDownload().";

            ftsClient.TrimDownload(startPos);

            RCF_LOG_3()(startPos.mFileIndex)(startPos.mOffset)
                << "ClientStub::downloadFiles() - TrimDownload() returned.";

            currentFile = startPos.mFileIndex;
            currentPos = startPos.mOffset;
            resume = true;
        }

        OfstreamPtr fout( new std::ofstream() );
        FileIoRequestPtr writeOp( new FileIoRequest() );
        
        boost::uint32_t adviseWaitMs = 0;
        
        // Calculate total byte count of the manifest.
        boost::uint64_t totalByteSize = manifest.getTotalByteSize();

        // Did we get any chunks on the BeginDownload() call?
        boost::uint64_t totalBytesReadSoFar = bytesAlreadyTransferred;
        RCF_ASSERT(chunks.empty());
        for (std::size_t i=0; i<chunks.size(); ++i)
        {
            totalBytesReadSoFar += chunks[i].mData.getLength();
        }

        // Progress callback.
        if (mFileProgressCb)
        {
            mFileProgressCb(totalBytesReadSoFar, totalByteSize);
        }

        const boost::uint32_t TransferWindowS = 5;
        RCF::Timer transferWindowTimer;
        boost::uint32_t transferWindowBytes = 0;
        bool localWait = false;

        while (currentFile != manifest.mFiles.size())
        {
            RCF_ASSERT(chunks.empty() || (currentFile == 0 && currentPos == 0));

            // Round trip to the server for more chunks.
            if (chunks.empty())
            {
                FileTransferRequest request;
                request.mFile = currentFile;
                request.mPos = currentPos;
                request.mChunkSize = chunkSize;

                // Respect server throttle settings.
                if (adviseWaitMs)
                {
                    RCF_LOG_3()(adviseWaitMs)
                        << "ClientStub::downloadFiles() - waiting on server throttle.";

                    // This needs to be a sub-second accurate sleep, or the timing tests will
                    // be thrown.
                    //Platform::OS::Sleep(1 + adviseWaitMs / 1000);
                    //Platform::OS::SleepMs(adviseWaitMs);
                    Sleep(adviseWaitMs);
                    adviseWaitMs = 0;
                }

                // Respect local throttle setting.
                if (localWait)
                {
                    boost::uint32_t startTimeMs = transferWindowTimer.getStartTimeMs();
                    boost::uint32_t nowMs = getCurrentTimeMs();
                    if (nowMs < startTimeMs + 1000*TransferWindowS)
                    {
                        boost::uint32_t waitMs = startTimeMs + 1000*TransferWindowS - nowMs;
                        Platform::OS::Sleep(1 + waitMs / 1000);
                        localWait = false;
                    }
                }

                // Trim chunk size according to transfer rate.                
                if (transferRateBps)
                {
                    if (transferWindowTimer.elapsed(TransferWindowS*1000))
                    {
                        transferWindowTimer.restart();
                        transferWindowBytes = 0;
                    }

                    boost::uint32_t bytesTotal = transferRateBps * TransferWindowS;
                    boost::uint32_t bytesRemaining =  bytesTotal - transferWindowBytes;

                    if (bytesRemaining < request.mChunkSize)
                    {
                        localWait = true;
                    }

                    chunkSize = RCF_MIN(chunkSize, bytesRemaining);
                }

                RCF_LOG_3()(request.mFile)(request.mPos)(startPos.mOffset)
                    << "ClientStub::downloadFiles() - calling DownloadChunks().";

                ftsClient.DownloadChunks(request, chunks, adviseWaitMs);

                RCF_LOG_3()(chunks.size())(adviseWaitMs)
                    << "ClientStub::downloadFiles() - DownloadChunks() returned.";

                // Update byte totals.
                for (std::size_t i=0; i<chunks.size(); ++i)
                {
                    totalBytesReadSoFar += chunks[i].mData.getLength();
                    transferWindowBytes += chunks[i].mData.getLength();
                }

                // Progress callback.
                if (mFileProgressCb)
                {
                    mFileProgressCb(totalBytesReadSoFar, totalByteSize);
                }
            }

            // Write to disk.
            for (std::size_t i=0; i<chunks.size(); ++i)
            {
                const FileChunk & chunk = chunks[i];

                if (chunk.mFileIndex != currentFile || chunk.mOffset != currentPos)
                {
                    // TODO: error
                    RCF_ASSERT(0);
                }

                FileInfo & info = manifest.mFiles[currentFile];

                fs::path filePath = manifestBase / info.mFilePath;
                if (info.mRenameFile.size() > 0)
                {
                    filePath = filePath.branch_path() / info.mRenameFile;
                }

                if (currentPos == 0 || resume)
                {
                    RCF_ASSERT(writeOp->completed());

                    if (currentPos == 0)
                    {
                        // Create new file.
                        RCF_LOG_3()(filePath)
                            << "ClientStub::downloadFiles() - opening file (truncating).";

                        fs::create_directories(filePath.branch_path());
                        fout->clear();
                        fout->open( filePath.string().c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
                        RCF_VERIFY(fout->good(), Exception(_RcfError_FileOpen(filePath.file_string())));
                    }
                    else
                    {
                        // Create new file.
                        RCF_LOG_3()(filePath)
                            << "ClientStub::downloadFiles() - opening file (appending).";

                        // Append to existing file.
                        fout->clear();
                        fout->open( filePath.string().c_str(), std::ios::out | std::ios::binary | std::ios::app | std::ios::ate);
                        RCF_VERIFY(fout->good(), Exception(_RcfError_FileOpen(filePath.file_string())));
                        resume = false;

                        // TODO: verify file position against currentPos.
                        // ...
                    }
                }

                // Wait for previous write to complete.
                if (writeOp->initiated())
                {
                    writeOp->complete();

                    boost::uint64_t bytesWritten = writeOp->getBytesTransferred();
                    if (bytesWritten == 0)
                    {
                        fout->close();
                        Exception e(_RcfError_FileWrite(filePath.file_string(), currentPos));
                        RCF_THROW(e);
                    }
                }

                if (currentPos + chunk.mData.getLength() > manifest.mFiles[chunk.mFileIndex].mFileSize)
                {
                    // TODO: error
                    RCF_ASSERT(0);
                }

                // Check stream state.
                if (!fout->good())
                {
                    // TODO: more info.
                    Exception e(_RcfError_FileWrite(filePath.file_string(), currentPos));
                    RCF_THROW(e);
                }

                RCF_ASSERT_EQ(currentPos , fout->tellp());

                RCF_LOG_3()(chunk.mData.getLength())
                    << "ClientStub::downloadFiles() - writing to file.";

                writeOp->write(fout, chunk.mData);

                currentPos += chunk.mData.getLength();

                if (currentPos == manifest.mFiles[currentFile].mFileSize)
                {
                    RCF_LOG_3()(currentFile)
                        << "ClientStub::downloadFiles() - closing file.";

                    writeOp->complete();
                    boost::uint64_t bytesWritten = writeOp->getBytesTransferred();
                    if (bytesWritten == 0)
                    {
                        fout->close();
                        Exception e(_RcfError_FileWrite(filePath.file_string(), currentPos));
                        RCF_THROW(e);
                    }
                    fout->close();
                    currentPos = 0;
                    ++currentFile;

                    if (currentFile < manifest.mFiles.size())
                    {
                        currentPos = manifest.mFiles[currentFile].mFileStartPos;
                    }
                }
            }

            chunks.clear();
        }

        totalManifest = manifest;

        RCF_LOG_3()(totalManifest.mFiles.size())
            << "ClientStub::downloadFiles() - exit.";

    }
} // namespace RCF

#endif // RCF_USE_BOOST_FILESYSTEM

namespace RCF {

    //**************************************************************************
    // Synchronous create object calls.

    namespace {

        void reinstateClientTransport(
            ClientStub &clientStub,
            I_RcfClient &factory)
        {
            clientStub.setTransport(factory.getClientStub().releaseTransport());
        }

    }

    void ClientStub::createRemoteObject(
        const std::string &objectName_)
    {
        const std::string &objectName = objectName_.empty() ? mInterfaceName : objectName_;
        unsigned int timeoutMs = getRemoteCallTimeoutMs();
        connect();
        RcfClient<I_ObjectFactory> factory(*this);
        factory.getClientStub().setTransport( releaseTransport());
        factory.getClientStub().setTargetToken( Token());
        // TODO: should only be using the remainder of the timeout
        factory.getClientStub().setRemoteCallTimeoutMs(timeoutMs);
        using namespace boost::multi_index::detail;
        scope_guard guard = make_guard(
            reinstateClientTransport,
            boost::ref(*this),
            boost::ref(factory));
        RCF_UNUSED_VARIABLE(guard);
        RCF::Token token;
        boost::int32_t ret = factory.CreateObject(RCF::Twoway, objectName, token);
        if (ret == RcfError_Ok)
        {
            setTargetToken(token);
        }
        else
        {
            setTargetToken(Token());

            // dtor issues with borland
#ifdef __BORLANDC__
            setTransport(factory.getClientStub().releaseTransport());
            guard.dismiss();
#endif

            Error err(ret);
            RemoteException e(err);
            RCF_THROW(e);
        }
    }

    // ObjectFactoryClient is an abstraction of RcfClient<I_ObjectFactoryService>,
    // and RcfClient<I_SessionObjectFactoryService>. We need to use either one,
    // depending on what the RCF runtime version is.

    class ObjectFactoryClient
    {
    public:
        ObjectFactoryClient(ClientStub & clientStub) :
            mRuntimeVersion(clientStub.getRuntimeVersion()),
            mCutoffVersion(2)
        {
            mRuntimeVersion <= mCutoffVersion ?
                client1.reset( new RcfClient<I_ObjectFactory>(clientStub)) :
                client2.reset( new RcfClient<I_SessionObjectFactory>(clientStub));
        }

        ClientStub &getClientStub()
        {
            return mRuntimeVersion <= mCutoffVersion ?
                client1->getClientStub() :
                client2->getClientStub();
        }

        RcfClientPtr getRcfClientPtr()
        {
            return mRuntimeVersion <= mCutoffVersion ?
                RcfClientPtr(client1) :
                RcfClientPtr(client2);
        }

        FutureImpl<boost::int32_t> CreateSessionObject(
            const ::RCF::CallOptions &callOptions,
            const std::string & objectName)
        {
            return mRuntimeVersion <= mCutoffVersion ?
                client1->CreateSessionObject(callOptions, objectName) :
                client2->CreateSessionObject(callOptions, objectName);
        }

        FutureImpl<boost::int32_t> DeleteSessionObject(
            const ::RCF::CallOptions &callOptions)
        {
            return mRuntimeVersion <= mCutoffVersion ?
                client1->DeleteSessionObject(callOptions) :
                client2->DeleteSessionObject(callOptions);
        }

        void reinstateClientTransport(ClientStub & clientStub)
        {
            ClientTransportAutoPtr clientTransportAutoPtr = 
                mRuntimeVersion <= mCutoffVersion ?
                    client1->getClientStub().releaseTransport() :
                    client2->getClientStub().releaseTransport();

            clientStub.setTransport(clientTransportAutoPtr);
        }

    private:
        boost::shared_ptr<RcfClient<I_ObjectFactory> >          client1;
        boost::shared_ptr<RcfClient<I_SessionObjectFactory> >   client2;

        const int                                               mRuntimeVersion;
        const int                                               mCutoffVersion;
    };

    void ClientStub::createRemoteSessionObject(
        const std::string &objectName_)
    {
        const std::string &objectName = objectName_.empty() ? mInterfaceName : objectName_;
        unsigned int timeoutMs = getRemoteCallTimeoutMs();
        connect();
        
        ObjectFactoryClient factory(*this);
        
        factory.getClientStub().setTransport( releaseTransport());
        factory.getClientStub().setTargetToken( Token());
        // TODO: should only be using the remainder of the timeout
        factory.getClientStub().setRemoteCallTimeoutMs(timeoutMs);

        using namespace boost::multi_index::detail;
        scope_guard guard = make_obj_guard(
            factory,
            &ObjectFactoryClient::reinstateClientTransport,
            boost::ref(*this));
        RCF_UNUSED_VARIABLE(guard);

        boost::int32_t ret = factory.CreateSessionObject(RCF::Twoway, objectName);
        if (ret == RcfError_Ok)
        {
            setTargetName("");
            setTargetToken(Token());
        }
        else
        {
            Error err(ret);
            RemoteException e(err);
            RCF_THROW(e);
        }
    }

    void ClientStub::deleteRemoteSessionObject()
    {
        ObjectFactoryClient factory(*this);
        factory.getClientStub().setTransport( releaseTransport());
        factory.getClientStub().setTargetToken( Token());

        using namespace boost::multi_index::detail;
        scope_guard guard = make_obj_guard(
            factory,
            &ObjectFactoryClient::reinstateClientTransport,
            boost::ref(*this));
        RCF_UNUSED_VARIABLE(guard);

        boost::int32_t ret = factory.DeleteSessionObject(RCF::Twoway);
        RCF_VERIFY(ret == RcfError_Ok, RCF::RemoteException( Error(ret) ));
    }

    void ClientStub::deleteRemoteObject()
    {
        Token token = getTargetToken();
        if (token != Token())
        {
            RcfClient<I_ObjectFactory> factory(*this);
            factory.getClientStub().setTransport( releaseTransport());
            factory.getClientStub().setTargetToken( Token());
            using namespace boost::multi_index::detail;
            scope_guard guard = make_guard(
                reinstateClientTransport,
                boost::ref(*this),
                boost::ref(factory));
            RCF_UNUSED_VARIABLE(guard);

            boost::int32_t ret = factory.DeleteObject(RCF::Twoway, token);
            RCF_VERIFY(ret == RcfError_Ok, RCF::RemoteException( Error(ret) ));
        }
    }

    //**************************************************************************
    // Synchronous transport filter requests.

    void pushBackFilterId(std::vector<boost::int32_t> &filterIds, FilterPtr filterPtr)
    {
        filterIds.push_back( filterPtr->getFilterDescription().getId());
    }

    // TODO: merge common code with queryTransportFilters()
    void ClientStub::requestTransportFilters(const std::vector<FilterPtr> &filters)
    {
        // TODO: the current message filter sequence is not being used,
        // when making the filter request call to the server.

        using namespace boost::multi_index::detail; // for scope_guard

        std::vector<boost::int32_t> filterIds;
        std::for_each(filters.begin(), filters.end(),
            boost::bind(pushBackFilterId, boost::ref(filterIds), _1));

        if (!isConnected())
        {
            connect();
        }
        RCF::RcfClient<RCF::I_RequestTransportFilters> client(*this);
        client.getClientStub().setTransport( releaseTransport());
        client.getClientStub().setTargetToken( Token());

        RestoreClientTransportGuard guard(*this, client.getClientStub());
        RCF_UNUSED_VARIABLE(guard);

        client.getClientStub().setRemoteCallTimeoutMs( getRemoteCallTimeoutMs() );
        int ret = client.RequestTransportFilters(RCF::Twoway, filterIds);
        RCF_VERIFY(ret == RcfError_Ok, RemoteException( Error(ret) ))(filterIds);
        client.getClientStub().getTransport().setTransportFilters(filters);
    }

    void ClientStub::requestTransportFilters(FilterPtr filterPtr)
    {
        std::vector<FilterPtr> filters;
        if (filterPtr.get())
        {
            filters.push_back(filterPtr);
        }
        requestTransportFilters(filters);
    }

    void ClientStub::requestTransportFilters()
    {
        requestTransportFilters( std::vector<FilterPtr>());
    }

    void ClientStub::clearTransportFilters()
    {
        disconnect();
        if (mTransport.get())
        {
            mTransport->setTransportFilters( std::vector<FilterPtr>());
        }
    }

    bool ClientStub::queryForTransportFilters(const std::vector<FilterPtr> &filters)
    {
        using namespace boost::multi_index::detail; // for scope_guard

        std::vector<boost::int32_t> filterIds;
        std::for_each(filters.begin(), filters.end(),
            boost::bind(pushBackFilterId, boost::ref(filterIds), _1));

        if (!isConnected())
        {
            connect();
        }
        RCF::RcfClient<RCF::I_RequestTransportFilters> client(*this);
        client.getClientStub().setTransport( releaseTransport());
        client.getClientStub().setTargetToken( Token());

        RestoreClientTransportGuard guard(*this, client.getClientStub());
        RCF_UNUSED_VARIABLE(guard);

        client.getClientStub().setRemoteCallTimeoutMs( getRemoteCallTimeoutMs() );
        int ret = client.QueryForTransportFilters(RCF::Twoway, filterIds);
        return ret == RcfError_Ok;
    }

    bool ClientStub::queryForTransportFilters(FilterPtr filterPtr)
    {
        std::vector<FilterPtr> filters;
        if (filterPtr.get())
        {
            filters.push_back(filterPtr);
        }
        return queryForTransportFilters(filters);
    }

    //**************************************************************************
    // Asynchronous object creation/destruction.

    class Handler
    {
    public:

        virtual ~Handler()
        {
        }

        void handle(
            Future<boost::int32_t>      fRet,
            I_RcfClient &               rcfClient,
            ClientStub &                clientStubOrig,
            boost::function0<void>      onCompletion)
        {
            ClientStubPtr clientStubPtr = 
                rcfClient.getClientStub().shared_from_this();

            ClientStubPtr clientStubOrigPtr = clientStubOrig.shared_from_this();

            clientStubOrigPtr->setTransport( 
                clientStubPtr->releaseTransport() );

            clientStubOrigPtr->setSubRcfClientPtr( RcfClientPtr() );

            std::auto_ptr<Exception> ape(clientStubPtr->getAsyncException());

            bool failed = (ape.get() != NULL);

            clientStubOrigPtr->setAsyncException(ape);

            if (failed)
            {
                onCompletion();
            }
            else
            {
                mClientStubPtr = clientStubOrigPtr;

                boost::int32_t ret = fRet;
                if (ret == RcfError_Ok)
                {
                    handleOk();
                    onCompletion();
                }
                else
                {
                    std::auto_ptr<Exception> apException(
                        new RemoteException( Error(ret) ));

                    clientStubOrigPtr->setAsyncException(apException);

                    handleFail();

                    onCompletion();
                }
            }
        }

        virtual void handleOk()
        {
        }

        virtual void handleFail()
        {
        }

    protected:
        ClientStubPtr mClientStubPtr;
    };

    typedef boost::shared_ptr<Handler> HandlerPtr;

    class CreateSessionObjectHandler : public Handler
    {
    private:
        void handleOk()
        {
            mClientStubPtr->setTargetName("");
            mClientStubPtr->setTargetToken(Token());
        }
    };

    class CreateObjectHandler : public Handler
    {
    public :
        CreateObjectHandler(Future<Token> fToken) :
            mfToken(fToken)
        {
        }

    private:
        void handleOk()
        {
            Token token = mfToken;
            mClientStubPtr->setTargetToken(token);
        }

        void handleFail()
        {
            mClientStubPtr->setTargetToken(Token());
        }

        Future<Token> mfToken;
    };

    class DeleteSessionObjectHandler : public Handler
    {};

    class DeleteObjectHandler : public Handler
    {};

    void ClientStub::createRemoteSessionObjectAsync(
        boost::function0<void>      onCompletion,
        const std::string &         objectName_)
    {
        const std::string &objectName = objectName_.empty() ? mInterfaceName : objectName_;
        unsigned int timeoutMs = getRemoteCallTimeoutMs();

        ObjectFactoryClient factory(*this);

        factory.getClientStub().setTransport( releaseTransport());
        factory.getClientStub().setTargetToken( Token());
        // TODO: should only be using the remainder of the timeout
        factory.getClientStub().setRemoteCallTimeoutMs(timeoutMs);

        setSubRcfClientPtr(factory.getRcfClientPtr());

        setAsync(true);

        Future<boost::int32_t> fRet;

        HandlerPtr handlerPtr( new CreateSessionObjectHandler());

        fRet = factory.CreateSessionObject(

            RCF::AsyncTwoway( boost::bind(
                &Handler::handle, 
                handlerPtr,
                fRet,
                boost::ref(*factory.getRcfClientPtr()),
                boost::ref(*this),
                onCompletion)),
                
            objectName);
    }

    void ClientStub::deleteRemoteSessionObjectAsync(
        boost::function0<void> onCompletion)
    {
        ObjectFactoryClient factory(*this);
        factory.getClientStub().setTransport( releaseTransport());
        factory.getClientStub().setTargetToken( Token());

        setSubRcfClientPtr(factory.getRcfClientPtr());

        setAsync(true);

        Future<boost::int32_t> fRet;

        HandlerPtr handlerPtr( new DeleteSessionObjectHandler());

        fRet = factory.DeleteSessionObject(

            RCF::AsyncTwoway( boost::bind(
                &Handler::handle, 
                handlerPtr,
                fRet,
                boost::ref(*factory.getRcfClientPtr()),
                boost::ref(*this),
                onCompletion))
                
                );
    }

    void ClientStub::createRemoteObjectAsync(
        boost::function0<void>  onCompletion,
        const std::string &     objectName_)
    {
        const std::string &objectName = objectName_.empty() ? mInterfaceName : objectName_;
        unsigned int timeoutMs = getRemoteCallTimeoutMs();
        //connect();

        typedef RcfClient<I_ObjectFactory> OfClient;
        typedef boost::shared_ptr<OfClient> OfClientPtr;

        OfClientPtr ofClientPtr( new OfClient(*this) );
        ofClientPtr->getClientStub().setTransport( releaseTransport());
        ofClientPtr->getClientStub().setTargetToken( Token());
        // TODO: should only be using the remainder of the timeout
        ofClientPtr->getClientStub().setRemoteCallTimeoutMs(timeoutMs);

        setSubRcfClientPtr(ofClientPtr);

        setAsync(true);

        Future<boost::int32_t> fRet;
        Future<Token> fToken;

        HandlerPtr handlerPtr( new CreateObjectHandler(fToken));

        fRet = ofClientPtr->CreateObject(

            RCF::AsyncTwoway( boost::bind(
                &Handler::handle, 
                handlerPtr,
                fRet,
                boost::ref(*ofClientPtr),
                boost::ref(*this),
                onCompletion)),
                
            objectName,
            fToken);
    }

    void ClientStub::deleteRemoteObjectAsync(
        boost::function0<void> onCompletion)
    {
        Token token = getTargetToken();
        if (token != Token())
        {
            typedef RcfClient<I_ObjectFactory> OfClient;
            typedef boost::shared_ptr<OfClient> OfClientPtr;

            OfClientPtr ofClientPtr( new OfClient(*this) );
            ofClientPtr->getClientStub().setTransport( releaseTransport());
            ofClientPtr->getClientStub().setTargetToken( Token());

            setSubRcfClientPtr(ofClientPtr);

            setAsync(true);

            Future<boost::int32_t> fRet;

            HandlerPtr handlerPtr( new DeleteObjectHandler());

            fRet = ofClientPtr->DeleteObject(

                RCF::AsyncTwoway( boost::bind(
                    &Handler::handle, 
                    handlerPtr,
                    fRet,
                    boost::ref(*ofClientPtr),
                    boost::ref(*this),
                    onCompletion)),
                    
                token);
        }
    }

    //**************************************************************************
    // Asynchronous transport filter requests.

    class RequestTransportFiltersHandler : public Handler
    {
    public :
        RequestTransportFiltersHandler(
            boost::shared_ptr< std::vector<FilterPtr> > filtersPtr) :
            mFiltersPtr(filtersPtr)
        {
        }

    private:
        void handleOk()
        {
            mClientStubPtr->getTransport().setTransportFilters(*mFiltersPtr);
        }

        boost::shared_ptr< std::vector<FilterPtr> > mFiltersPtr;
    };

    class QueryForTransportFiltersHandler : public Handler
    {
    };

    void ClientStub::requestTransportFiltersAsync(
        const std::vector<FilterPtr> &filters,
        boost::function0<void> onCompletion)
    {

        std::vector<boost::int32_t> filterIds;

        std::for_each(
            filters.begin(), 
            filters.end(),
            boost::bind(pushBackFilterId, boost::ref(filterIds), _1));

        boost::shared_ptr<std::vector<FilterPtr> > filtersPtr(
            new std::vector<FilterPtr>(filters) );

        typedef RcfClient<I_RequestTransportFilters> RtfClient;
        typedef boost::shared_ptr<RtfClient> RtfClientPtr;

        RtfClientPtr rtfClientPtr( new RtfClient(*this) );

        rtfClientPtr->getClientStub().setTransport( releaseTransport());
        rtfClientPtr->getClientStub().setTargetToken( Token());

        setSubRcfClientPtr(rtfClientPtr);

        setAsync(true);

        Future<boost::int32_t> fRet;

        HandlerPtr handlerPtr( new RequestTransportFiltersHandler(filtersPtr));

        fRet = rtfClientPtr->RequestTransportFilters(
            
            RCF::AsyncTwoway( boost::bind(
                &Handler::handle, 
                handlerPtr,
                fRet,
                boost::ref(*rtfClientPtr),
                boost::ref(*this),
                onCompletion)),

            filterIds);

    }

    void ClientStub::queryForTransportFiltersAsync(
        const std::vector<FilterPtr> &filters,
        boost::function0<void> onCompletion)
    {

        std::vector<boost::int32_t> filterIds;

        std::for_each(filters.begin(), filters.end(),
            boost::bind(pushBackFilterId, boost::ref(filterIds), _1));

        typedef RcfClient<I_RequestTransportFilters> RtfClient;
        typedef boost::shared_ptr<RtfClient> RtfClientPtr;

        RtfClientPtr rtfClientPtr( new RtfClient(*this) );
        rtfClientPtr->getClientStub().setTransport( releaseTransport());
        rtfClientPtr->getClientStub().setTargetToken( Token());

        setSubRcfClientPtr(rtfClientPtr);

        setAsync(true);

        Future<boost::int32_t> fRet;

        HandlerPtr handlerPtr( new QueryForTransportFiltersHandler());

        fRet = rtfClientPtr->QueryForTransportFilters(
            
            RCF::AsyncTwoway( boost::bind(
                &Handler::handle, 
                handlerPtr,
                fRet,
                boost::ref(*rtfClientPtr),
                boost::ref(*this),
                onCompletion)),

            filterIds);
    }

    void ClientStub::requestTransportFiltersAsync(
        FilterPtr filterPtr,
        boost::function0<void> onCompletion)
    {
        std::vector<FilterPtr> filters;
        if (filterPtr.get())
        {
            filters.push_back(filterPtr);
        }
        requestTransportFiltersAsync(filters, onCompletion);
    }

    void ClientStub::queryForTransportFiltersAsync(
        FilterPtr filterPtr,
        boost::function0<void> onCompletion)
    {
        std::vector<FilterPtr> filters;
        if (filterPtr.get())
        {
            filters.push_back(filterPtr);
        }
        queryForTransportFiltersAsync(filters, onCompletion);
    }

    // Batching

    void ClientStub::enableBatching()
    {
        mBatchMode = true;
        if (!mBatchBufferPtr)
        {
            mBatchBufferPtr.reset( new ReallocBuffer() );
        }
        mBatchBufferPtr->resize(0);
        mBatchCount = 0;
        mBatchMessageCount = 0;
    }

    void ClientStub::disableBatching(bool flush)
    {
        if (flush)
        {
            flushBatch();
        }
        mBatchMode = false;
        mBatchBufferPtr->resize(0);
        mBatchMessageCount = 0;
    }

    void ClientStub::flushBatch(unsigned int timeoutMs)
    {
        CurrentClientStubSentry sentry(*this);

        if (timeoutMs == 0)
        {
            timeoutMs = getRemoteCallTimeoutMs();
        }

        try
        {
            std::vector<ByteBuffer> buffers;
            buffers.push_back( ByteBuffer(mBatchBufferPtr) );
            int err = getTransport().send(*this, buffers, timeoutMs);
            RCF_UNUSED_VARIABLE(err);

            mBatchBufferPtr->resize(0);

            ++mBatchCount;
            mBatchMessageCount = 0;
        }
        catch(const RemoteException & e)
        {
            RCF_UNUSED_VARIABLE(e);
            mEncodedByteBuffers.resize(0);
            throw;
        }
        catch(...)
        {
            mEncodedByteBuffers.resize(0);
            disconnect();
            throw;
        }
    }

    void ClientStub::setMaxBatchMessageLength(boost::uint32_t maxBatchMessageLength)
    {
        mBatchMaxMessageLength = maxBatchMessageLength;
    }

    boost::uint32_t ClientStub::getMaxBatchMessageLength()
    {
        return mBatchMaxMessageLength;
    }

    boost::uint32_t ClientStub::getBatchesSent()
    {
        return mBatchCount;
    }

    boost::uint32_t ClientStub::getMessagesInCurrentBatch()
    {
        return mBatchMessageCount;
    }

    void ClientStub::setRequestUserData(const std::string & userData)
    {
        mRequest.mRequestUserData = ByteBuffer(userData);
    }

    std::string ClientStub::getRequestUserData()
    {
        if ( mRequest.mRequestUserData.isEmpty() )
        {
            return std::string();
        }

        return std::string(
            mRequest.mRequestUserData.getPtr(), 
            mRequest.mRequestUserData.getLength());
    }

    void ClientStub::setResponseUserData(const std::string & userData)
    {
        mRequest.mResponseUserData = ByteBuffer(userData);
    }

    std::string ClientStub::getResponseUserData()
    {
        if ( mRequest.mResponseUserData.isEmpty() )
        {
            return std::string();
        }

        return std::string(
            mRequest.mResponseUserData.getPtr(), 
            mRequest.mResponseUserData.getLength());
    }

} // namespace RCF
