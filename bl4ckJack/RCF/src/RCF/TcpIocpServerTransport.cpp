
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

#include <RCF/TcpIocpServerTransport.hpp>

#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/TcpClientTransport.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/TimedBsdSockets.hpp>

namespace RCF {

    void resetSessionStatePtr(IocpSessionStatePtr &sessionStatePtr);

    // TcpIocpSessionState

    TcpIocpSessionState::TcpIocpSessionState(
        TcpIocpServerTransport & transport) : 
            IocpSessionState(transport),
            mTransport(transport),
            mFd(-1)
    {}

    TcpIocpSessionState::TcpIocpSessionStatePtr 
    TcpIocpSessionState::create(
        TcpIocpServerTransport & transport)
    {
        TcpIocpSessionStatePtr sessionStatePtr( 
            new TcpIocpSessionState(transport));

        SessionPtr sessionPtr = transport.getSessionManager().createSession();
        sessionPtr->setSessionState(*sessionStatePtr);
        sessionStatePtr->mSessionPtr = sessionPtr;
        sessionStatePtr->mWeakThisPtr = IocpSessionStateWeakPtr(sessionStatePtr);

        return sessionStatePtr;
    }

    TcpIocpSessionState::~TcpIocpSessionState()
    {
        RCF_DTOR_BEGIN

            // close the socket, if appropriate
            RCF_LOG_4()(mFd) << "TcpIocpSessionState - closing server socket.";

            RCF_ASSERT_NEQ(mFd , -1);
            postClose();

            // adjust number of queued accepts, if appropriate
            if (mPreState == IocpSessionState::Accepting && mTransport.isStarted())
            {
                TcpIocpSessionState::create(mTransport)->accept();
            }

        RCF_DTOR_END
    }

    const I_RemoteAddress & TcpIocpSessionState::getRemoteAddress()
    {
        return mRemoteAddress;
    }

    void TcpIocpSessionState::accept()
    {
        mTransport.registerSession(mWeakThisPtr);

        int family = mTransport.mIpAddress.getType() == RCF::IpAddress::V4 ?
            AF_INET :
            AF_INET6;


        Fd fd = static_cast<Fd>( socket(
            family,
            SOCK_STREAM,
            IPPROTO_TCP));

        int error = Platform::OS::BsdSockets::GetLastError();

        RCF_VERIFY(
            fd != -1,
            Exception(
                _RcfError_Socket(),
                error,
                RcfSubsystem_Os,
                "socket() failed"));

        Platform::OS::BsdSockets::setblocking(fd, false);

        mFd = fd;

        DWORD dwBytes = 0;

        for (std::size_t i=0; i<mAcceptExBuffer.size(); ++i)
        {
            mAcceptExBuffer[i] = 0;
        }

        clearOverlapped();

        mThisPtr = mWeakThisPtr.lock();

        BOOL ret = mTransport.mlpfnAcceptEx(
            mTransport.mAcceptorFd,
            mFd,
            &mAcceptExBuffer[0],
            0,
            AcceptExBufferSize/2,
            AcceptExBufferSize/2,
            &dwBytes,
            static_cast<OVERLAPPED *>(this));

        int err = WSAGetLastError();

        if (ret == FALSE && err == ERROR_IO_PENDING)
        {
            // async accept initiated successfully
            
        }
        else if (dwBytes > 0)
        {
            RCF_ASSERT(0);
            mThisPtr.reset();
            transition();
        }
        else
        {
            mThisPtr.reset();
            int err = Platform::OS::BsdSockets::GetLastError();
            
            Exception e(
                _RcfError_Socket(),
                err,
                RcfSubsystem_Os,
                "AcceptEx() failed");

            RCF_THROW(e);
        }
    }

    void TcpIocpSessionState::implOnAccept()
    {
        mTransport.mpIocp->AssociateSocket(mFd, 0);

        std::size_t sessionCount = 0;
        std::size_t connectionLimit = mTransport.getConnectionLimit();
        if (connectionLimit)
        {
            Lock lock(mTransport.mSessionsMutex);
            sessionCount = mTransport.mSessions.size();
        }

        TcpIocpSessionState::create(mTransport)->accept();

        // Check the connection limit.
        if (connectionLimit && sessionCount >= 1 + connectionLimit)
        {
            sendServerError(RcfError_ConnectionLimitExceeded);
            return;
        }

        // Parse local and remote address info.
        {
            SOCKADDR *pLocalAddr = NULL;
            SOCKADDR *pRemoteAddr = NULL;

            int localAddrLen = 0;
            int remoteAddrLen = 0;

            RCF_ASSERT(mTransport.mlpfnGetAcceptExSockAddrs);
            mTransport.mlpfnGetAcceptExSockAddrs(
                &mAcceptExBuffer[0],
                0,
                AcceptExBufferSize/2,
                AcceptExBufferSize/2,
                &pLocalAddr,
                &localAddrLen,
                &pRemoteAddr,
                &remoteAddrLen);

            IpAddress::Type addrType = mTransport.mIpAddress.getType();
            RCF_ASSERT(addrType == IpAddress::V4 || addrType == IpAddress::V6);
            mLocalAddress = IpAddress(*pLocalAddr, localAddrLen, addrType);
            mRemoteAddress = IpAddress(*pRemoteAddr, remoteAddrLen, addrType);
        }

        if (mTransport.isIpAllowed(mRemoteAddress))
        {
            // Simulate a completed write to kick things off.
            mPreState = IocpSessionState::WritingData;
            mWriteBufferRemaining = 0;
            transition();
        }
        else
        {
            RCF_LOG_2()(mRemoteAddress.getIp()) 
                << "Client IP does not match server's IP access rules. Closing connection.";

            // The server session will fall off the stack.
        }
    }

    void TcpIocpSessionState::implRead(
        const ByteBuffer &byteBuffer,
        std::size_t bufferLen)
    {
        WSAOVERLAPPED *pOverlapped = static_cast<WSAOVERLAPPED *>(this);

        bufferLen = RCF_MIN(mTransport.getMaxSendRecvSize(), bufferLen);
        WSABUF wsabuf = {0};
        wsabuf.buf = byteBuffer.getPtr();
        wsabuf.len = static_cast<u_long>(bufferLen);
        DWORD dwReceived = 0;
        DWORD dwFlags = 0;
        mError = 0;
        mPostState = Reading;

        // set self-reference
        RCF_ASSERT(!mThisPtr.get());
        mThisPtr = mWeakThisPtr.lock();
        RCF_ASSERT(mThisPtr.get());

        using namespace boost::multi_index::detail;
        scope_guard clearSelfReferenceGuard =
            make_guard(resetSessionStatePtr, boost::ref(mThisPtr));

        RCF_LOG_4()(wsabuf.len) << "TcpIocpSessionState - calling WSARecv().";

        Lock lock(mMutex);

        if (!mHasBeenClosed)
        {

            int ret = WSARecv(
                mFd,
                &wsabuf,
                1,
                &dwReceived,
                &dwFlags,
                pOverlapped,
                NULL);

            if (ret == -1)
            {
                mError = WSAGetLastError();
            }

            RCF_ASSERT(ret == -1 || ret == 0);
            if (mError == S_OK || mError == WSA_IO_PENDING)
            {
                mError = 0;
                clearSelfReferenceGuard.dismiss();
            }
        }
    }

    void TcpIocpSessionState::implWrite(
        const std::vector<ByteBuffer> &byteBuffers,
        IocpSessionState * pReflectee)
    {
        WSAOVERLAPPED *pOverlapped = 
            static_cast<WSAOVERLAPPED *>(this);

        TcpIocpSessionState * pTcpReflectee = 
            static_cast<TcpIocpSessionState *>(pReflectee);

        std::size_t bytesAdded = 0;

        mWsabufs.resize(0);
        for (std::size_t i=0; i<byteBuffers.size(); ++i)
        {
            std::size_t bytesToAdd = RCF_MIN(
                byteBuffers[i].getLength(),
                mTransport.getMaxSendRecvSize() - bytesAdded);

            if (bytesToAdd > 0)
            {
                WSABUF wsabuf = {0};
                wsabuf.buf = byteBuffers[i].getPtr();
                wsabuf.len = static_cast<u_long>(bytesToAdd);
                mWsabufs.push_back(wsabuf);
                bytesAdded += bytesToAdd;
            }
        }

        DWORD dwSent = 0;
        DWORD dwFlags = 0;
        mError = 0;
        mPostState = Writing;

        // set self-reference
        RCF_ASSERT(!mThisPtr.get());
        mThisPtr = mWeakThisPtr.lock();
        RCF_ASSERT(mThisPtr.get());

        using namespace boost::multi_index::detail;
        scope_guard clearSelfReferenceGuard =
            make_guard(resetSessionStatePtr, boost::ref(mThisPtr));

        RCF_LOG_4()(RCF::lengthByteBuffers(byteBuffers))(bytesAdded) 
            << "TcpIocpSessionState - calling WSASend().";

        Mutex & mutex = pReflectee ? pReflectee->mMutex : mMutex;
        bool & hasBeenClosed = pReflectee ? pReflectee->mHasBeenClosed : mHasBeenClosed;
        Fd & fd = pReflectee ? pTcpReflectee->mFd : mFd;

        Lock lock(mutex);

        if (!hasBeenClosed)
        {
            int ret = WSASend(
                fd,
                &mWsabufs[0],
                static_cast<DWORD>(mWsabufs.size()),
                &dwSent,
                dwFlags,
                pOverlapped,
                NULL);

            if (ret == -1)
            {
                mError = WSAGetLastError();
            }

            RCF_ASSERT(ret == -1 || ret == 0);
            if (mError == S_OK || mError == WSA_IO_PENDING)
            {
                clearSelfReferenceGuard.dismiss();
                mError = 0;
            }
        }
    }

    void TcpIocpSessionState::implDelayCloseAfterSend()
    {
        const int BufferSize = 8*1024;
        char buffer[BufferSize];
        while (recv(mFd, buffer, BufferSize, 0) > 0);
        int ret = shutdown(mFd, SD_BOTH);
        RCF_UNUSED_VARIABLE(ret);
        postRead();
    }

    void TcpIocpSessionState::implClose()
    {
        int ret = Platform::OS::BsdSockets::closesocket(mFd);
        int err = Platform::OS::BsdSockets::GetLastError();

        RCF_VERIFY(
            ret == 0,
            Exception(
                _RcfError_SocketClose(),
                err,
                RcfSubsystem_Os,
                "closesocket() failed"))
            (mFd);

        mHasBeenClosed = true;
    }

    Fd TcpIocpSessionState::getNativeHandle()
    {
        return mFd;
    }

    bool TcpIocpSessionState::implIsConnected()
    {
        return isFdConnected(mFd);
    }

    ClientTransportAutoPtr TcpIocpSessionState::implCreateClientTransport()
    {
        int fd = -1;
        {
            Lock lock(mMutex);
            if (mOwnFd && !mHasBeenClosed)
            {
                mOwnFd = false;
                fd = mFd;
            }
        }

        std::auto_ptr<TcpClientTransport> tcpClientTransport(
            new TcpClientTransport(fd));

        tcpClientTransport->setNotifyCloseFunctor( boost::bind(
            &IocpSessionState::notifyClose, 
            IocpSessionStateWeakPtr(sharedFromThis())));

        tcpClientTransport->setRemoteAddr(mRemoteAddress);

        tcpClientTransport->mRegisteredForAmi = true;

        return ClientTransportAutoPtr(tcpClientTransport.release());
    }

    // TcpIocpServerTransport

    TcpIocpServerTransport::TcpIocpServerTransport(
        const IpAddress & ipAddress) : 
            IocpServerTransport(),
            mIpAddress(ipAddress),
            mAcceptorFd(-1),
            mQueuedAccepts(0),
            mQueuedAcceptsThreshold(10),
            mQueuedAcceptsAugment(10),
            mlpfnAcceptEx(RCF_DEFAULT_INIT),
            mlpfnGetAcceptExSockAddrs(RCF_DEFAULT_INIT),
            mMaxPendingConnectionCount(100)
    {       
    }

    TcpIocpServerTransport::TcpIocpServerTransport(
        const std::string & ip, 
        int port) :
            IocpServerTransport(),
            mIpAddress( IpAddress(ip, port) ),
            mAcceptorFd(-1),
            mQueuedAccepts(0),
            mQueuedAcceptsThreshold(10),
            mQueuedAcceptsAugment(10),
            mlpfnAcceptEx(RCF_DEFAULT_INIT),
            mlpfnGetAcceptExSockAddrs(RCF_DEFAULT_INIT),
            mMaxPendingConnectionCount(100)
    {

    }

    ServerTransportPtr TcpIocpServerTransport::clone()
    {
        return ServerTransportPtr( new TcpIocpServerTransport(mIpAddress) );
    }

    int TcpIocpServerTransport::getPort() const
    {
        return mIpAddress.getPort();
    }

    void TcpIocpServerTransport::setMaxPendingConnectionCount(
        std::size_t maxPendingConnectionCount)
    {
        mMaxPendingConnectionCount = maxPendingConnectionCount;
    }

    std::size_t TcpIocpServerTransport::getMaxPendingConnectionCount() const
    {
        return mMaxPendingConnectionCount;
    }

    void TcpIocpServerTransport::implOpen()
    {
        // set up a listening socket, if we have a non-negative port number (>0)

        if (mAcceptorFd != -1)
        {
            // Listening socket is already set up.
            return;
        }

        mIpAddress.resolve();

        int mPort = getPort();
        
        RCF_ASSERT_GTEQ(mPort , -1);
        RCF_ASSERT_EQ(mAcceptorFd , -1);

        mQueuedAccepts = 0;
        
        if (mPort >= 0)
        {
            // create listener socket
            int ret = 0;
            int err = 0;

            mAcceptorFd = mIpAddress.createSocket(SOCK_STREAM, IPPROTO_TCP);

            sockaddr * pSockAddr = NULL;
            Platform::OS::BsdSockets::socklen_t sockAddrSize = 0;
            mIpAddress.getSockAddr(pSockAddr, sockAddrSize);

            ret = ::bind(
                mAcceptorFd, 
                pSockAddr, 
                sockAddrSize);

            if (ret < 0)
            {
                err = Platform::OS::BsdSockets::GetLastError();
                if (err == WSAEADDRINUSE)
                {
                    Exception e(_RcfError_PortInUse(mIpAddress.getIp(), mIpAddress.getPort()), err, RcfSubsystem_Os, "bind() failed");
                    RCF_THROW(e)(mAcceptorFd);
                }
                else
                {
                    Exception e(_RcfError_SocketBind(mIpAddress.getIp(), mIpAddress.getPort()), err, RcfSubsystem_Os, "bind() failed");
                    RCF_THROW(e)(mAcceptorFd);
                }
            }

            // listen on listener socket
            ret = listen(
                mAcceptorFd, 
                static_cast<int>(mMaxPendingConnectionCount));

            if (ret < 0)
            {
                err = Platform::OS::BsdSockets::GetLastError();
                Exception e(_RcfError_Socket(), err, RcfSubsystem_Os, "listen() failed");
                RCF_THROW(e);
            }
            RCF_ASSERT_NEQ( mAcceptorFd , -1 );

            // retrieve the port number, if it's generated by the system
            if (mPort == 0)
            {
                IpAddress ip(mAcceptorFd, mIpAddress.getType());
                mPort = ip.getPort();
                mIpAddress.setPort(mPort);
            }

            RCF_LOG_2() << "TcpIocpServerTransport - listening on port " << mPort << ".";

            // load AcceptEx() function
            GUID GuidAcceptEx = WSAID_ACCEPTEX;
            DWORD dwBytes;

            ret = WSAIoctl(
                mAcceptorFd,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                &GuidAcceptEx,
                sizeof(GuidAcceptEx),
                &mlpfnAcceptEx,
                sizeof(mlpfnAcceptEx),
                &dwBytes,
                NULL,
                NULL);

            err = Platform::OS::BsdSockets::GetLastError();

            RCF_VERIFY(
                ret == 0,
                Exception(_RcfError_Socket(), err, RcfSubsystem_Os,
                "WSAIoctl() failed"));

            // load GetAcceptExSockAddrs() function
            GUID GuidGetAcceptExSockAddrs = WSAID_GETACCEPTEXSOCKADDRS;

            ret = WSAIoctl(
                mAcceptorFd,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                &GuidGetAcceptExSockAddrs,
                sizeof(GuidGetAcceptExSockAddrs),
                &mlpfnGetAcceptExSockAddrs,
                sizeof(mlpfnGetAcceptExSockAddrs),
                &dwBytes,
                NULL,
                NULL);

            err = Platform::OS::BsdSockets::GetLastError();

            RCF_VERIFY(
                ret == 0,
                Exception(_RcfError_Socket(), err, RcfSubsystem_Os,
                "WsaIoctl() failed"));
            
        }        
    }

    void TcpIocpServerTransport::implClose()
    {
        // close listener socket
        if (mAcceptorFd != -1)
        {
            int ret = closesocket(mAcceptorFd);
            int err = Platform::OS::BsdSockets::GetLastError();

            RCF_VERIFY(
                ret == 0,
                Exception(
                _RcfError_SocketClose(),
                err,
                RcfSubsystem_Os,
                "closesocket() failed"))
                (mAcceptorFd);

            mAcceptorFd = -1;
        }

        // reset queued accepts count
        mQueuedAccepts = 0;
    }

    void TcpIocpSessionState::assignFd(Fd fd)
    {
        mFd = fd;
    }    

    void TcpIocpSessionState::assignRemoteAddress(
        const IpAddress & remoteAddress)
    {
        mRemoteAddress = remoteAddress;
    }

    IocpSessionStatePtr TcpIocpServerTransport::implCreateServerSession(
        I_ClientTransport & clientTransport)
    {
        TcpClientTransport &tcpClientTransport =
            dynamic_cast<TcpClientTransport &>(clientTransport);

        int fd = tcpClientTransport.releaseFd();
        RCF_ASSERT_GT(fd , 0);
        
        typedef boost::shared_ptr<TcpIocpSessionState> TcpIocpSessionStatePtr;
        TcpIocpSessionStatePtr sessionStatePtr( TcpIocpSessionState::create(*this));
        sessionStatePtr->assignFd(fd);

        sessionStatePtr->assignRemoteAddress(
            IpAddress(tcpClientTransport.getRemoteAddr()));

        // TODO: If the client transport *is* registered for AMI, then we 
        // really should check which iocp it is associated with, and assert 
        // that it is the iocp of this particular server transport.
        if (!tcpClientTransport.mRegisteredForAmi)
        {
            mpIocp->AssociateSocket(fd, 0);
        }

        return IocpSessionStatePtr(sessionStatePtr);
    }

    ClientTransportAutoPtr TcpIocpServerTransport::implCreateClientTransport(
        const I_Endpoint &endpoint)
    {
        const TcpEndpoint &tcpEndpoint =
            dynamic_cast<const TcpEndpoint &>(endpoint);

        std::auto_ptr<TcpClientTransport> tcpClientTransportAutoPtr(
            new TcpClientTransport( tcpEndpoint.getIpAddress() ));

        return ClientTransportAutoPtr(tcpClientTransportAutoPtr.release());
    }

    void TcpIocpServerTransport::onServiceAdded(RcfServer &server)
    {
        IocpServerTransport::onServiceAdded(server);
    }

    void TcpIocpServerTransport::onServerStart(RcfServer & server)
    {
        IocpServerTransport::onServerStart(server);

        // associate listener socket to iocp
        if (mAcceptorFd != -1)
        {
            mpIocp->AssociateSocket( (SOCKET) mAcceptorFd, 0);
            TcpIocpSessionState::create(*this)->accept();
        }
    }

} // namespace RCF
