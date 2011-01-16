
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

#ifndef INCLUDE_RCF_CLIENTSTUB_HPP
#define INCLUDE_RCF_CLIENTSTUB_HPP

#include <string>
#include <vector>
#include <memory>

#include <boost/scoped_ptr.hpp>
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <RCF/AsyncFilter.hpp>
#include <RCF/ClientTransport.hpp>
#include <RCF/Endpoint.hpp>
#include <RCF/Export.hpp>
#include <RCF/GetInterfaceName.hpp>
#include <RCF/MethodInvocation.hpp>
#include <RCF/Protocol/Protocol.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/Token.hpp>

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/SfNew.hpp>
#include <SF/shared_ptr.hpp>
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
#include <boost/serialization/shared_ptr.hpp>
#endif

#ifdef RCF_USE_BOOST_FILESYSTEM
#include <RCF/FileStream.hpp>
#endif

namespace RCF {

    class ConnectionResetGuard;

    class I_Parameters;

    /// Indicates whether a client should use oneway or twoway semantics for remote calls.
    enum RemoteCallSemantics
    {
        Oneway,
        Twoway
    };

    RCF_EXPORT void setDefaultConnectTimeoutMs(unsigned int connectTimeoutMs);
    RCF_EXPORT unsigned int getDefaultConnectTimeoutMs();

    RCF_EXPORT void setDefaultRemoteCallTimeoutMs(unsigned int remoteCallTimeoutMs);
    RCF_EXPORT unsigned int getDefaultRemoteCallTimeoutMs();

    RCF_EXPORT void setDefaultNativeWstringSerialization(bool enable);
    RCF_EXPORT bool getDefaultNativeWstringSerialization();

    class ClientStub;

    typedef boost::shared_ptr<ClientStub> ClientStubPtr;

    typedef Token FileTransferToken;

    class ClientProgress;
    typedef boost::shared_ptr<ClientProgress> ClientProgressPtr;

    class I_ClientTransport;
    typedef std::auto_ptr<I_ClientTransport> ClientTransportAutoPtr;

    class I_RcfClient;
    typedef boost::shared_ptr<I_RcfClient> RcfClientPtr;

    class I_Future;
    class I_IpClientTransport;

    template<typename T>
    class FutureImpl;

    typedef boost::function2<void, boost::uint64_t, boost::uint64_t> FileProgressCb;

    template<
        typename R, 
        typename A1,
        typename A2,
        typename A3,
        typename A4,
        typename A5,
        typename A6,
        typename A7,
        typename A8,
        typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15>
    class AllocateClientParameters;

    template<
        typename R, 
        typename A1,
        typename A2,
        typename A3,
        typename A4,
        typename A5,
        typename A6,
        typename A7,
        typename A8,
        typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15>
    class ClientParameters;

    class OverlappedAmi;
    typedef boost::shared_ptr<OverlappedAmi> OverlappedAmiPtr;

    class RCF_EXPORT CurrentClientStubSentry
    {
    public:
        CurrentClientStubSentry(ClientStub & clientStub);
        ~CurrentClientStubSentry();
    };

    class RCF_EXPORT ClientStub : 
        public I_ClientTransportCallback, 
        public boost::enable_shared_from_this<ClientStub>
    {
    public:
        ClientStub(const std::string &interfaceName);
        ClientStub(const std::string &interfaceName, const std::string &objectName);
        ClientStub(const ClientStub &rhs);
        ~ClientStub();

        ClientStub &operator=(const ClientStub &rhs);

        void                setEndpoint(const I_Endpoint &endpoint);
        void                setEndpoint(EndpointPtr endpointPtr);
        EndpointPtr         getEndpoint() const;
        Token               getTargetToken() const;
        void                setTargetToken(Token token);
        std::string         getTargetName() const;
        void                setTargetName(const std::string &targetName);
        void                setInterfaceName(const std::string & interfaceName);
        std::string         getInterfaceName();
        RemoteCallSemantics getDefaultCallingSemantics() const;
        void                setDefaultCallingSemantics(RemoteCallSemantics defaultCallingSemantics);

        void                    setSerializationProtocol(SerializationProtocol protocol);
        SerializationProtocol   getSerializationProtocol() const;

        void                    setMarshalingProtocol(MarshalingProtocol protocol);
        MarshalingProtocol      getMarshalingProtocol() const;

        bool                    getNativeWstringSerialization();
        void                    setNativeWstringSerialization(bool enable);

        void                    enableSfSerializationPointerTracking(bool enable);
        void                    setTransport(ClientTransportAutoPtr transport);

        I_ClientTransport&      getTransport();
        I_IpClientTransport &   getIpTransport();

        ClientTransportAutoPtr  releaseTransport();

        void        instantiateTransport();
        void        connect();
        void        connectAsync(boost::function0<void> onCompletion);
        void        waitAsync(boost::function0<void> onCompletion, boost::uint32_t timeoutMs);
        void        disconnect();
        bool        isConnected();
        void        setConnected(bool connected);

        void        setMessageFilters(const std::vector<FilterPtr> &filters);
        void        setMessageFilters();
        void        setMessageFilters(FilterPtr filterPtr);

        const std::vector<FilterPtr> &
                    getMessageFilters();


        // Synchronous transport filter requests.
        void        requestTransportFilters(const std::vector<FilterPtr> &filters);
        void        requestTransportFilters(FilterPtr filterPtr);
        void        requestTransportFilters();

        bool        queryForTransportFilters(const std::vector<FilterPtr> &filters);
        bool        queryForTransportFilters(FilterPtr filterPtr);
        void        clearTransportFilters();

        // Asynchronous transport filter requests.
        void        requestTransportFiltersAsync(
                        const std::vector<FilterPtr> &filters,
                        boost::function0<void> onCompletion);

        void        requestTransportFiltersAsync(
                        FilterPtr filterPtr,
                        boost::function0<void> onCompletion);

        void        queryForTransportFiltersAsync(
                        const std::vector<FilterPtr> &filters,
                        boost::function0<void> onCompletion);

        void        queryForTransportFiltersAsync(
                        FilterPtr filterPtr,
                        boost::function0<void> onCompletion);


        void            setRemoteCallTimeoutMs(unsigned int remoteCallTimeoutMs);
        unsigned int    getRemoteCallTimeoutMs() const;

        void            setConnectTimeoutMs(unsigned int connectTimeoutMs);
        unsigned int    getConnectTimeoutMs() const;

        void        setAutoReconnect(bool autoReconnect);
        bool        getAutoReconnect() const;

        void        setAutoVersioning(bool autoVersioning);
        bool        getAutoVersioning() const;

        void            setRuntimeVersion(boost::uint32_t version);
        boost::uint32_t getRuntimeVersion() const;

        void            setArchiveVersion(boost::uint32_t version);
        boost::uint32_t getArchiveVersion() const;

        void        setClientProgressPtr(ClientProgressPtr ClientProgressPtr);

        ClientProgressPtr 
                    getClientProgressPtr() const;

        void        setTries(std::size_t tries);
        std::size_t getTries() const;

        void        setUserData(boost::any userData);
        boost::any  getUserData();

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

        //**********************************************************************
        // These functions involve network calls.

        // Synchronous versions.

        void createRemoteObject(const std::string &objectName = "");
        void deleteRemoteObject();

        void createRemoteSessionObject(const std::string &objectName = "");
        void deleteRemoteSessionObject();

#ifdef RCF_USE_BOOST_FILESYSTEM
        void setFileProgressCallback(FileProgressCb fileProgressCb);
        void setFileProgressCallback() { setFileProgressCallback( FileProgressCb() ); }

        void uploadFiles(
            const std::string & whichFile, 
            boost::uint32_t & uploadId,
            boost::uint32_t chunkSize = 8*1024,
            boost::uint32_t sessionLocalId = 0);

        void uploadFiles(
            const FileManifest & whichFile, 
            boost::uint32_t & uploadId,
            boost::uint32_t chunkSize = 8*1024,
            boost::uint32_t sessionLocalId = 0);

        void downloadFiles(
            const std::string & downloadLocation, 
            FileManifest & manifest,
            boost::uint32_t chunkSize = 8*1024, 
            boost::uint32_t transferRateBps = 0,
            boost::uint32_t sessionLocalId = 0);

        boost::uint32_t addUploadStream(FileUpload fileStream);
        void processUploadStreams();

        boost::uint32_t addDownloadStream(FileDownload fileStream);

        // For testing.
        void                setTransferWindowS(boost::uint32_t transferWindowS);
        boost::uint32_t     getTransferWindowS();
#endif

        void ping();
        void ping(RemoteCallSemantics rcs);

        // Asynchronous versions.

        void createRemoteObjectAsync(
            boost::function0<void> onCompletion,
            const std::string &objectName = "");

        void deleteRemoteObjectAsync(
            boost::function0<void> onCompletion);

        void createRemoteSessionObjectAsync(
            boost::function0<void> onCompletion,
            const std::string &objectName = "");

        void deleteRemoteSessionObjectAsync(
            boost::function0<void> onCompletion);

        //void ping(
        //    boost::function1<void, ExceptionPtr> onCompletion);

        //void ping(
        //    RemoteCallSemantics rcs, 
        //    boost::function1<void, ExceptionPtr> onCompletion);



        //**********************************************************************

        void setPingBackIntervalMs(int pingBackIntervalMs);
        int getPingBackIntervalMs();

        std::size_t     getPingBackCount();
        boost::uint32_t getPingBackTimeStamp();
        
        void clearParameters();

        SerializationProtocolIn &   getSpIn();
        SerializationProtocolOut &  getSpOut();

        std::auto_ptr<Exception>    getAsyncException();
        void                        setAsyncException(std::auto_ptr<Exception>);
        bool                        hasAsyncException();

        boost::uint32_t     generatePollingTimeout(boost::uint32_t timeoutMs);
        void                onPollingTimeout();
        void                onUiMessage();

        friend class CallOptions;

#if defined(_MSC_VER) && _MSC_VER < 1310

        // vc6: Can't seem to get declare FutureImpl<>, ClientParameters<> and 
        // AllocateClientParameters<> as friends.
    public:    

#else
    private:

        template<typename T>
        friend class FutureImpl;

        template<
            typename R, 
            typename A1,
            typename A2,
            typename A3,
            typename A4,
            typename A5,
            typename A6,
            typename A7,
            typename A8,
            typename A9,
            typename A10,
            typename A11,
            typename A12,
            typename A13,
            typename A14,
            typename A15>
        friend class AllocateClientParameters;

        template<
            typename R, 
            typename A1,
            typename A2,
            typename A3,
            typename A4,
            typename A5,
            typename A6,
            typename A7,
            typename A8,
            typename A9,
            typename A10,
            typename A11,
            typename A12,
            typename A13,
            typename A14,
            typename A15>
        friend class ClientParameters;

#endif

        Token                       mToken;
        RemoteCallSemantics         mDefaultCallingSemantics;
        SerializationProtocol       mProtocol;
        MarshalingProtocol          mMarshalingProtocol; 
        std::string                 mEndpointName;
        std::string                 mObjectName;
        std::string                 mInterfaceName;

        unsigned int                mRemoteCallTimeoutMs;
        unsigned int                mConnectTimeoutMs;
        bool                        mAutoReconnect;
        bool                        mConnected;
        std::size_t                 mTries;

        EndpointPtr                 mEndpoint;
        ClientTransportAutoPtr      mTransport;

        VectorFilter                mMessageFilters;

        ClientProgressPtr           mClientProgressPtr;

        bool                        mAutoVersioning;
        boost::uint32_t             mRuntimeVersion;
        boost::uint32_t             mArchiveVersion;

        bool                        mUseNativeWstringSerialization;

        std::vector<I_Future *>     mFutures;
        boost::any                  mUserData;        

        MethodInvocationRequest     mRequest;
        SerializationProtocolIn     mIn;
        SerializationProtocolOut    mOut;

        typedef std::pair<boost::uint32_t, OverlappedAmiPtr> TimerEntry;

        enum TimerReason
        {
            None,
            Wait,
            Connect,
            Write,
            Read
        };

        bool                        mAsync;
        TimerEntry                  mAsyncTimerEntry;
        TimerReason                 mAsyncTimerReason;
        boost::function0<void>      mAsyncCallback;
        std::auto_ptr<Exception>    mAsyncException;
        unsigned int                mEndTimeMs;
        bool                        mRetry;
        RemoteCallSemantics         mRcs;
        ByteBuffer                  mEncodedByteBuffer;
        std::vector<ByteBuffer>     mEncodedByteBuffers;

        std::vector<char>           mRetValVec;
        std::vector<char>           mParametersVec;
        I_Parameters *              mpParameters;

        int                         mPingBackIntervalMs;
        boost::uint32_t             mPingBackTimeStamp;
        std::size_t                 mPingBackCount;

        boost::uint32_t             mNextTimerCallbackMs;
        boost::uint32_t             mNextPingBackCheckMs;
        boost::uint32_t             mPingBackCheckIntervalMs;
        boost::uint32_t             mTimerIntervalMs;

        boost::shared_ptr<Mutex>        mSignalledMutex;
        boost::shared_ptr<Condition>    mSignalledCondition;
        bool                            mSignalled;

        Mutex                           mSubRcfClientMutex;
        RcfClientPtr                    mSubRcfClientPtr;

        bool                            mBatchMode;
        ReallocBufferPtr                mBatchBufferPtr;
        ReallocBuffer                   mBatchBufferTemp;
        boost::uint32_t                 mBatchMaxMessageLength;
        boost::uint32_t                 mBatchCount;
        boost::uint32_t                 mBatchMessageCount;

        void                enrol(
                                I_Future *pFuture);

        void                init( 
                                const std::string &subInterface, 
                                int fnId, 
                                RCF::RemoteCallSemantics rcs);

        void                send();

        void                receive();

        void                call( 
                                RCF::RemoteCallSemantics rcs);

        void                onConnectCompleted(
                                bool alreadyConnected = false);

        void                onSendCompleted();

        void                onReceiveCompleted();

        void                onTimerExpired();

        void                onError(
                                const std::exception &e);
   
        void                setAsyncCallback(
                                boost::function0<void> callback);        

        void                onException(
                                const Exception & e);

        void                prepareAmiNotification();

    public:

        std::vector<char> & getRetValVec() { return mRetValVec; }

        // Batching

        void                enableBatching();
        void                disableBatching(bool flush = true);
        void                flushBatch(unsigned int timeoutMs = 0);

        void                setMaxBatchMessageLength(boost::uint32_t maxBatchMessageLength);
        boost::uint32_t     getMaxBatchMessageLength();

        boost::uint32_t     getBatchesSent();
        boost::uint32_t     getMessagesInCurrentBatch();

        // Async

        void                setAsync(bool async);
        bool                getAsync();

        bool                ready();
        void                wait(boost::uint32_t timeoutMs = 0);
        void                cancel();

        void                setSubRcfClientPtr(RcfClientPtr clientStubPtr);
        RcfClientPtr        getSubRcfClientPtr();

        // User data

        void                setRequestUserData(const std::string & userData);
        std::string         getRequestUserData();

        void                setResponseUserData(const std::string & userData);
        std::string         getResponseUserData();

#ifdef RCF_USE_BOOST_FILESYSTEM
        FileProgressCb              mFileProgressCb;

        std::vector<FileUpload>     mUploadStreams;
        std::vector<FileDownload>   mDownloadStreams;       
#endif

        boost::uint32_t             mTransferWindowS;    
    };

    class CallOptions
    {
    public:
        CallOptions() :
            mAsync(false),
            mRcsSpecified(false), 
            mRcs(Twoway),
            mCallback()
        {}

        CallOptions(RemoteCallSemantics rcs) : 
            mAsync(false),
            mRcsSpecified(true), 
            mRcs(rcs),
            mCallback()
        {}

        CallOptions(RemoteCallSemantics rcs, boost::function0<void> callback) : 
            mAsync(true),
            mRcsSpecified(true),
            mRcs(rcs),
            mCallback(callback)
        {}

        CallOptions(boost::function0<void> callback) : 
            mAsync(true),
            mRcsSpecified(false),
            mRcs(Twoway), 
            mCallback(callback)
        {}

        RemoteCallSemantics apply(ClientStub &clientStub) const
        {
            clientStub.setAsync(mAsync);
            clientStub.setAsyncCallback(mCallback);
            return mRcsSpecified ? mRcs : clientStub.getDefaultCallingSemantics();
        }

    private:
        bool mAsync;
        bool mRcsSpecified;
        RemoteCallSemantics mRcs;
        boost::function0<void> mCallback;
    };

    class RCF_EXPORT AsyncTwoway : public CallOptions
    {
    public:
        AsyncTwoway(boost::function0<void> callback) :
            CallOptions(RCF::Twoway, callback)
        {}
    };

    class AsyncOneway : public CallOptions
    {
    public:
        AsyncOneway(boost::function0<void> callback) :
            CallOptions(RCF::Oneway, callback)
        {}
    };

    class RestoreClientTransportGuard
    {
    public:

        RestoreClientTransportGuard(ClientStub &client, ClientStub &clientTemp) :
            mClient(client),
            mClientTemp(clientTemp)
        {}

        ~RestoreClientTransportGuard()
        {
            RCF_DTOR_BEGIN
            mClient.setTransport(mClientTemp.releaseTransport());
            RCF_DTOR_END
        }

    private:
        ClientStub &mClient;
        ClientStub &mClientTemp;
    };


} // namespace RCF

#ifdef RCF_USE_SF_SERIALIZATION

namespace SF {

    SF_CTOR(RCF::ClientStub, RCF::ClientStub("", ""))

} // namespace SF

#endif

#endif // ! INCLUDE_RCF_CLIENTSTUB_HPP
