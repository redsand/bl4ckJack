
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

#ifndef INCLUDE_RCF_RCFSESSION_HPP
#define INCLUDE_RCF_RCFSESSION_HPP

#include <vector>

#include <boost/any.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <RCF/Export.hpp>
#include <RCF/MethodInvocation.hpp>
#include <RCF/SerializationProtocol.hpp>
#include <RCF/ServerTransport.hpp>
#include <RCF/StubEntry.hpp>

#ifdef RCF_USE_BOOST_FILESYSTEM
#include <RCF/FileStream.hpp>
#endif

namespace RCF {

    class Filter;

    typedef boost::shared_ptr<Filter> FilterPtr;

    class RcfSession;

    typedef boost::shared_ptr<RcfSession> RcfSessionPtr;
    typedef boost::weak_ptr<RcfSession> RcfSessionWeakPtr;

    class I_Future;

    class I_Parameters;

    class UdpServerTransport;
    class UdpSessionState;

    class FileTransferService;
    class FileUploadInfo;
    class FileDownloadInfo;

    class FileStreamImpl;

    typedef boost::shared_ptr<FileUploadInfo>   FileUploadInfoPtr;
    typedef boost::shared_ptr<FileDownloadInfo> FileDownloadInfoPtr;

    typedef std::pair<boost::uint32_t, RcfSessionWeakPtr>   PingBackTimerEntry;

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
    class AllocateServerParameters;

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
    class ServerParameters;

    class PingBackService;

    class RCF_EXPORT RcfSession : 
        public boost::enable_shared_from_this<RcfSession>
    {
    public:
        RcfSession(RcfServer &server);
        ~RcfSession();

        typedef boost::function1<void, RcfSession&> OnWriteCompletedCallback;
        typedef boost::function1<void, RcfSession&> OnWriteInitiatedCallback;
        typedef boost::function1<void, RcfSession&> OnDestroyCallback;

        //*******************************
        // callback tables - synchronized

        // may well be called on a different thread than the one that executed the remote call
        void addOnWriteCompletedCallback(
            const OnWriteCompletedCallback &        onWriteCompletedCallback);

        void extractOnWriteCompletedCallbacks(
            std::vector<OnWriteCompletedCallback> & onWriteCompletedCallbacks);

        void setOnDestroyCallback(
            OnDestroyCallback                       onDestroyCallback);

        //*******************************

        const RCF::I_RemoteAddress &
                        getRemoteAddress();

        RcfServer &     getRcfServer();

        void            disconnect();

        bool            hasDefaultServerStub();
        StubEntryPtr    getDefaultStubEntryPtr();
        void            setDefaultStubEntryPtr(StubEntryPtr stubEntryPtr);
        void            setCachedStubEntryPtr(StubEntryPtr stubEntryPtr);

        void            enableSfSerializationPointerTracking(bool enable);

        boost::uint32_t getRuntimeVersion();
        void            setRuntimeVersion(boost::uint32_t version);

        boost::uint32_t getArchiveVersion();
        void            setArchiveVersion(boost::uint32_t version);

        bool            getNativeWstringSerialization();
        void            setNativeWstringSerialization(bool enable);

        void            setUserData(boost::any userData);
        boost::any      getUserData();

        void            getMessageFilters(std::vector<FilterPtr> &filters);
        void            getTransportFilters(std::vector<FilterPtr> &filters);

        void            lockTransportFilters();
        void            unlockTransportFilters();
        bool            transportFiltersLocked();

        SerializationProtocolIn &   getSpIn();
        SerializationProtocolOut &  getSpOut();

        bool                        getFiltered();
        void                        setFiltered(bool filtered);

        std::vector<FilterPtr> &    getFilters();

        void            setCloseSessionAfterWrite(bool close);

        boost::uint32_t getPingBackIntervalMs();

        boost::uint32_t getPingTimestamp();
        void            setPingTimestamp();

        boost::uint32_t getPingIntervalMs();
        void            setPingIntervalMs(boost::uint32_t pingIntervalMs);

        boost::uint32_t getTouchTimestamp();

        void            touch();

        void            sendPingBack();
        bool            getAutoSend();

        void            setWeakThisPtr();

        void            setRequestUserData(const std::string & userData);
        std::string     getRequestUserData();

        void            setResponseUserData(const std::string & userData);
        std::string     getResponseUserData();

        void            cancelDownload();

#ifdef RCF_USE_BOOST_FILESYSTEM

        void            addDownloadStream(
                            boost::uint32_t sessionLocalId, 
                            FileStream fileStream);

#endif

        Mutex                                   mStopCallInProgressMutex;
        bool                                    mStopCallInProgress;
        
#if defined(_MSC_VER) && _MSC_VER < 1310

        // vc6: Can't seem to get declare ServerParameters<> and 
        // AllocateServerParameters<> as friends.
    public:    

#else

    private:

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
        friend class AllocateServerParameters;

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
        friend class ServerParameters;

        friend class PingBackService;

#endif

        RcfServer &                             mRcfServer;

        Mutex                                   mMutex;
        std::vector<OnWriteCompletedCallback>   mOnWriteCompletedCallbacks;
        std::vector<OnWriteInitiatedCallback>   mOnWriteInitiatedCallbacks;
        OnDestroyCallback                       mOnDestroyCallback;

        boost::uint32_t                         mRuntimeVersion;
        boost::uint32_t                         mArchiveVersion;

        bool                                    mUseNativeWstringSerialization;
        
        bool                                    mTransportFiltersLocked;

        SerializationProtocolIn                 mIn;
        SerializationProtocolOut                mOut;

        // message filters
        std::vector<FilterPtr>                  mFilters;
        bool                                    mFiltered;

        MethodInvocationRequest                 mRequest;

        bool                                    mCloseSessionAfterWrite;
        boost::uint32_t                         mPingTimestamp;
        boost::uint32_t                         mPingIntervalMs;
        boost::uint32_t                         mTouchTimestamp;
        ByteBuffer                              mPingBackByteBuffer;
        PingBackTimerEntry                      mPingBackTimerEntry;

        Mutex                                   mIoStateMutex;
        bool                                    mWritingPingBack;
        std::vector<ByteBuffer>                 mQueuedSendBuffers;

        void clearParameters();

        void onReadCompleted();
        void onWriteCompleted();

        void processRequest();

        void sendResponse();
        void sendResponseException(const std::exception &e);
        void sendResponseUncaughtException();

        void encodeRemoteException(
            SerializationProtocolOut & out, 
            const RemoteException & e);

        void sendSessionResponse(); 

        void registerForPingBacks();
        void unregisterForPingBacks();

        friend class RcfServer;
        friend class AmdImpl;

        I_Parameters *                          mpParameters;
        std::vector<char>                       mParametersVec;

        // For individual parameters.
        std::vector< std::vector<char> >        mParmsVec;

        bool                                    mAutoSend;

        RcfSessionWeakPtr                       mWeakThisPtr;

    private:

        // UdpServerTransport needs to explicitly set mIoState to Reading,
        // since it doesn't use async I/O with callbacks to RcfServer.
        friend class UdpServerTransport;
        friend class UdpSessionState;
        friend class FileStreamImpl;

#ifdef RCF_USE_BOOST_FILESYSTEM

    private:

        friend class FileTransferService;
        friend class PingBackService;

        FileDownloadInfoPtr                     mDownloadInfoPtr;
        FileUploadInfoPtr                       mUploadInfoPtr;

        typedef std::map<boost::uint32_t, FileUploadInfoPtr> SessionUploads;
        typedef std::map<boost::uint32_t, FileDownload> SessionDownloads;

        SessionUploads                          mSessionUploads;
        SessionDownloads                        mSessionDownloads;

#endif

    private:

        boost::any                              mUserData;
        StubEntryPtr                            mDefaultStubEntryPtr;
        StubEntryPtr                            mCachedStubEntryPtr;

    public:
        I_SessionState & getSessionState() const;
        void setSessionState(I_SessionState & sessionState);

    private:
        I_SessionState * mpSessionState;
    };       

} // namespace RCF

#endif // ! INCLUDE_RCF_RCFSESSION_HPP
