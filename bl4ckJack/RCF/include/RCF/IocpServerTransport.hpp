
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

#ifndef INCLUDE_RCF_IOCPSERVERTRANSPORT_HPP
#define INCLUDE_RCF_IOCPSERVERTRANSPORT_HPP

#include <memory>
#include <set>
#include <vector>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#include <RCF/Export.hpp>
#include <RCF/Iocp.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/ServerTask.hpp>
#include <RCF/ServerTransport.hpp>
#include <RCF/Service.hpp>
#include <RCF/ThreadLibrary.hpp>

namespace RCF {

    class RcfServer;

    class                                           IocpSessionState;
    typedef boost::shared_ptr<IocpSessionState>     IocpSessionStatePtr;
    typedef boost::weak_ptr<IocpSessionState>       IocpSessionStateWeakPtr;
    class                                           IocpServerTransport;

    typedef int                                     Fd;
    typedef std::vector<ByteBuffer>                 ByteBuffers;
    typedef std::vector<FilterPtr>                  Filters;

    // SessionState

    class RCF_EXPORT IocpSessionState :
        public IocpOverlapped,
        public I_SessionState
    {
    public:

        enum State
        {
            Accepting,
            ReadingDataCount,
            ReadingData,
            WritingData,
            Ready
        };

        enum PostState
        {
            Reading,
            Writing
        };

        IocpSessionState(IocpServerTransport &transport);

        ~IocpSessionState();

        IocpSessionStatePtr sharedFromThis();

        void onCompletion(
            BOOL ret, 
            DWORD dwErr,
            ULONG_PTR completionKey, 
            DWORD dwNumBytes);

        void    setTransportFilters(
                    const std::vector<FilterPtr> &filters);

        void    getTransportFilters(
                    std::vector<FilterPtr> &filters);

        void    filteredRead(
                    ByteBuffer &byteBuffer,
                    std::size_t bufferLen);

        void    filteredWrite(
                    const std::vector<ByteBuffer> &byteBuffers);

        void    onReadWriteCompleted(
                    std::size_t bytesTransferred);

        void    onFilteredReadWriteCompleted(
                    std::size_t bytesTransferred);

        void    onAccept();

        void    transition();

        void    sendServerError(int error);

        void    postWrite(std::vector<ByteBuffer> &byteBuffers);

        void    postRead();

        void    postClose();

        I_ServerTransport & getServerTransport();
        
        static 
        void    notifyClose(
                    const IocpSessionStateWeakPtr &sessionStateWeakPtr);

        static 
        void    closeSession(
                    const IocpSessionStateWeakPtr &sessionStateWeakPtr);

        void    resetState();

        int     getLastError();
        
    protected:

        friend class IocpServerTransport;
        friend class FilterProxy;

        int                         mError;
        std::vector<WSABUF>         mWsabufs;

        IocpSessionStateWeakPtr     mReflectionSessionStateWeakPtr;

        State                       mPreState;
        PostState                   mPostState;
        SessionPtr                  mSessionPtr;
        bool                        mIssueZeroByteRead;

        ReallocBufferPtr            mReadBufferPtr;
        ReallocBufferPtr            mReadBufferSecondaryPtr;

        ByteBuffer                  mReadByteBuffer;
        ByteBuffer                  mReadByteBufferSecondary;

        std::vector<ByteBuffer>     mWriteByteBuffers;
        std::vector<ByteBuffer>     mSlicedWriteByteBuffers;
        std::size_t                 mReadBufferRemaining;
        std::size_t                 mWriteBufferRemaining;

        std::vector<FilterPtr>      mTransportFilters;
        IocpServerTransport &       mTransport;
        
        IocpSessionStateWeakPtr     mWeakThisPtr;
        IocpSessionStatePtr         mThisPtr;

        bool                        mCloseAfterWrite;
        bool                        mReflected;

        bool                        mOwnFd;

    public:
        bool                        mHasBeenClosed;
        Mutex                       mMutex;

    protected:

        void                    read(
                                    const ByteBuffer &,
                                    std::size_t);

        void                    write(
                                    const std::vector<ByteBuffer> &,
                                    IocpSessionState * pReflectee = NULL);

        void                    reflect(
                                    std::size_t bytesRead);

        ReallocBuffer &         getReadBuffer();
        ReallocBuffer &         getUniqueReadBuffer();

        ByteBuffer              getReadByteBuffer();

        ReallocBuffer &         getReadBufferSecondary();
        ReallocBuffer &         getUniqueReadBufferSecondary();

        ByteBuffer              getReadByteBufferSecondary() const;

    private:

        virtual 
        void                    implOnAccept() = 0;

        virtual 
        void                    implRead(
                                    const ByteBuffer &byteBuffer, 
                                    std::size_t bufferLen) = 0;
        virtual 
        void                    implWrite(
                                    const std::vector<ByteBuffer> &byteBuffers, 
                                    IocpSessionState * pReflectee = NULL) = 0;

        virtual 
        ClientTransportAutoPtr  implCreateClientTransport() = 0;

        virtual 
        void                    implDelayCloseAfterSend() = 0;

        virtual 
        void                    implClose() = 0;
        
        virtual 
        bool                    implIsConnected() = 0;

        virtual
        void                    implOnMessageLengthError();

    };

    // IocpServerTransport

    class RCF_EXPORT IocpServerTransport :
        public I_ServerTransport,
        public I_ServerTransportEx,
        public I_Service,
        boost::noncopyable
    {
    private:

        friend class IocpSessionState;

    public:
        IocpServerTransport();
        ~IocpServerTransport();

        void                open();

        void                close();

        void                cycle(
                                int timeoutMs,
                                const volatile bool &stopFlag);

        bool                cycleTransportAndServer(
                                RcfServer &server,
                                int timeoutMs,
                                const volatile bool &stopFlag);

        // this is the size going into wsasend()/wsarecv(), not the max message size!
        void                setMaxSendRecvSize(
                                std::size_t maxSendRecvSize);

        std::size_t         getMaxSendRecvSize() const;

        void                setSessionManager(
                                RcfServer & rcfServer);

        RcfServer &  getSessionManager();


        // I_ServerTransportEx implementation
    private:

        ClientTransportAutoPtr  createClientTransport(
                                    const I_Endpoint &endpoint);

        ClientTransportAutoPtr  createClientTransport(
                                    SessionPtr sessionPtr);

        SessionPtr          createServerSession(
                                ClientTransportAutoPtr & clientTransportAutoPtr,
                                StubEntryPtr stubEntryPtr,
                                bool keepClientConnection);

        bool                reflect(
                                const SessionPtr &sessionPtr1,
                                const SessionPtr &sessionPtr2);

        bool                reflect(
                                const IocpSessionStatePtr &sessionStatePtr1,
                                const IocpSessionStatePtr &sessionStatePtr2);

        bool                isConnected(
                                const SessionPtr &sessionPtr);

        // I_Service implementation
    protected:
        void                onServiceAdded(RcfServer &server);
        void                onServiceRemoved(RcfServer &server);
        void                onServerStart(RcfServer &server);
        void                onServerStop(RcfServer &server);
        bool                isStarted();

    private:

        virtual 
        IocpSessionStatePtr     implCreateServerSession(
                                    I_ClientTransport & clientTransport) = 0;
        
        virtual 
        ClientTransportAutoPtr  implCreateClientTransport(
                                    const I_Endpoint &endpoint) = 0;

        virtual 
        void                    implOpen() = 0;

        virtual 
        void                    implClose() = 0;

    protected:

        void                    registerSession(IocpSessionStateWeakPtr session);
        void                    unregisterSession(IocpSessionStateWeakPtr session);

        RcfServer *                         mpRcfServer;        
        Iocp *                              mpIocp;
        std::size_t                         mMaxSendRecvSize;
        
        volatile bool                       mStopFlag;
        Mutex                               mSessionsMutex;
        std::set<IocpSessionStateWeakPtr>   mSessions;

    public:

        void associateSocket(int fd);
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_IOCPSERVERTRANSPORT_HPP
