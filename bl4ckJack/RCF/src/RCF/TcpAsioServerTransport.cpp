
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

#include <RCF/TcpAsioServerTransport.hpp>

#include <RCF/Asio.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/TcpClientTransport.hpp>
#include <RCF/TcpEndpoint.hpp>

namespace RCF {

    IpAddress boostToRcfIpAdress(const boost::asio::ip::tcp::endpoint & endpoint)
    {
        boost::asio::ip::address asioAddr = endpoint.address();

        IpAddress ipAddress;

        if (asioAddr.is_v4())
        {
            sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(endpoint.port());
            addr.sin_addr.s_addr = htonl(asioAddr.to_v4().to_ulong());
            ipAddress = IpAddress(addr);
        }
#ifdef RCF_USE_IPV6
        else if (asioAddr.is_v6())
        {
            RCF_ASSERT(asioAddr.is_v6());

            SockAddrIn6 addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin6_family = AF_INET6;
            addr.sin6_port = htons(endpoint.port());

            boost::asio::ip::address_v6 asioAddrV6 = asioAddr.to_v6();
            boost::asio::ip::address_v6::bytes_type bytes = asioAddrV6.to_bytes();
            memcpy(addr.sin6_addr.s6_addr, &bytes[0], bytes.size());

            ipAddress = IpAddress(addr);
        }
#endif

        return ipAddress;
    }

    class TcpAsioAcceptor : public AsioAcceptor
    {
    public:
        TcpAsioAcceptor(
            AsioIoService & ioService, 
            boost::asio::ip::tcp::acceptor::protocol_type protocolType, 
            int acceptorFd) : 
                mAcceptor(ioService, protocolType, acceptorFd)
        {}

        boost::asio::ip::tcp::acceptor mAcceptor;
    };

    typedef boost::asio::ip::tcp::socket        TcpAsioSocket;
    typedef boost::shared_ptr<TcpAsioSocket>    TcpAsioSocketPtr;

    // TcpAsioSessionState

    class TcpAsioSessionState : public AsioSessionState
    {
    public:

        TcpAsioSessionState(
            TcpAsioServerTransport &transport,
            AsioIoService & ioService) :
                AsioSessionState(transport, ioService),
                mSocketPtr(new TcpAsioSocket(ioService))
        {}

        const I_RemoteAddress & implGetRemoteAddress()
        {
            return mIpAddress;
        }

        void implRead(char * buffer, std::size_t bufferLen)
        {
            RCF_LOG_4()(bufferLen) 
                << "TcpAsioSessionState - calling async_read_some().";

            mThisPtr = sharedFromThis();

            mSocketPtr->async_read_some(
                boost::asio::buffer( buffer, bufferLen),
                ReadHandler(*this));
        }

        void implWrite(const std::vector<ByteBuffer> & buffers)
        {
            RCF_LOG_4()(RCF::lengthByteBuffers(buffers))
                << "TcpIocpSessionState - calling async_write_some().";

            mThisPtr = sharedFromThis();
            
            mBufs.mVecPtr->resize(0);
            for (std::size_t i=0; i<buffers.size(); ++i)
            {
                ByteBuffer buffer = buffers[i];

                mBufs.mVecPtr->push_back( 
                    boost::asio::buffer(buffer.getPtr(), buffer.getLength()) );
            }

            mSocketPtr->async_write_some(
                mBufs,
                WriteHandler(*this));
        }

        void implWrite(
            AsioSessionState &toBeNotified, 
            const char * buffer, 
            std::size_t bufferLen)
        {
            toBeNotified.mThisPtr = toBeNotified.sharedFromThis();

            boost::asio::async_write(
                *mSocketPtr,
                boost::asio::buffer(buffer, bufferLen),
                WriteHandler(toBeNotified));
        }

        void implAccept()
        {
            RCF_LOG_4()
                << "TcpAsioSessionState - calling async_accept().";

            TcpAsioAcceptor & tcpAsioAcceptor = 
                static_cast<TcpAsioAcceptor &>(*mTransport.getAcceptorPtr());

            tcpAsioAcceptor.mAcceptor.async_accept(
                *mSocketPtr,
                boost::bind(
                    &AsioSessionState::onAccept,
                    sharedFromThis(),
                    boost::asio::placeholders::error));
        }

        bool implOnAccept()
        {
            boost::asio::ip::tcp::endpoint endpoint = 
                mSocketPtr->remote_endpoint();

            mIpAddress = boostToRcfIpAdress(endpoint);

            TcpAsioServerTransport & transport = 
                static_cast<TcpAsioServerTransport &>(mTransport);

            bool ipAllowed = transport.isIpAllowed(mIpAddress);
            if (!ipAllowed)
            {
                RCF_LOG_2()(mIpAddress.getIp()) 
                    << "Client IP does not match server's IP access rules. Closing connection.";
            }

            return ipAllowed;
        }

        int implGetNative() const
        {
            return mSocketPtr->native();
        }

        boost::function0<void> implGetCloseFunctor()
        {
            return boost::bind(
                &TcpAsioSessionState::closeSocket,
                mSocketPtr);
        }

        void implClose()
        {
            mSocketPtr->close();
        }

        ClientTransportAutoPtr implCreateClientTransport()
        {
            int fd = implGetNative();

            std::auto_ptr<TcpClientTransport> tcpClientTransport(
                new TcpClientTransport(fd));

            boost::asio::ip::tcp::endpoint endpoint = 
                mSocketPtr->remote_endpoint();

            IpAddress ipAddress = boostToRcfIpAdress(endpoint);
            tcpClientTransport->setRemoteAddr(ipAddress);

            return ClientTransportAutoPtr(tcpClientTransport.release());
        }

        void implTransferNativeFrom(I_ClientTransport & clientTransport)
        {
            TcpClientTransport *pTcpClientTransport =
                dynamic_cast<TcpClientTransport *>(&clientTransport);

            if (pTcpClientTransport == NULL)
            {
                Exception e("Incompatible client transport.");
                RCF_THROW(e)(typeid(clientTransport));
            }

            TcpClientTransport & tcpClientTransport = *pTcpClientTransport;

            // TODO: exception safety
            mSocketPtr->assign(
                boost::asio::ip::tcp::v4(),
                tcpClientTransport.releaseFd());
        }

        static void closeSocket(TcpAsioSocketPtr socketPtr)
        {
            socketPtr->close();
        }

    private:

        TcpAsioSocketPtr            mSocketPtr;
        IpAddress                   mIpAddress;
    };

    // TcpAsioServerTransport

    TcpAsioServerTransport::TcpAsioServerTransport(
        const IpAddress & ipAddress) :
            mIpAddress(ipAddress),
            mAcceptorFd(-1)
    {
    }

    TcpAsioServerTransport::TcpAsioServerTransport(
        const std::string & ip, 
        int port) :
            mIpAddress(ip, port),
            mAcceptorFd(-1)
    {
    }

    ServerTransportPtr TcpAsioServerTransport::clone()
    {
        return ServerTransportPtr(new TcpAsioServerTransport(mIpAddress));
    }

    AsioSessionStatePtr TcpAsioServerTransport::implCreateSessionState()
    {
        return AsioSessionStatePtr( new TcpAsioSessionState(*this, getIoService()) );
    }

    int TcpAsioServerTransport::getPort() const
    {
        return mIpAddress.getPort();
    }

    void TcpAsioServerTransport::implOpen()
    {
        // We open the port manually, without asio. Then later, when we know
        // which io_service to use, we attach the socket to a regular tcp::acceptor.

        if (mAcceptorFd != -1)
        {
            // Listening socket has already been opened.
            return;
        }

        RCF_ASSERT_EQ(mAcceptorFd , -1);

        if (mIpAddress.getPort() != -1)
        {
            mIpAddress.resolve();
            mAcceptorFd = mIpAddress.createSocket(SOCK_STREAM, IPPROTO_TCP);

            sockaddr * pSockAddr = NULL;
            Platform::OS::BsdSockets::socklen_t sockAddrSize = 0;
            mIpAddress.getSockAddr(pSockAddr, sockAddrSize);

            // Set SO_REUSEADDR socket option.
            int enable = 1;
            int ret = setsockopt(mAcceptorFd, SOL_SOCKET, SO_REUSEADDR, (char *) &enable, sizeof(enable));
            int err = Platform::OS::BsdSockets::GetLastError();
            
            RCF_VERIFY(
                ret ==  0,
                Exception(
                    _RcfError_Socket(),
                    err,
                    RcfSubsystem_Os,
                        "setsockopt() with SO_REUSEADDR failed"));

            ret = ::bind(
                mAcceptorFd, 
                pSockAddr, 
                sockAddrSize);

            if (ret < 0)
            {
                err = Platform::OS::BsdSockets::GetLastError();
                if (err == Platform::OS::BsdSockets::ERR_EADDRINUSE)
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
            ret = listen(mAcceptorFd, 200);

            if (ret < 0)
            {
                int err = Platform::OS::BsdSockets::GetLastError();
                Exception e(_RcfError_Socket(), err, RcfSubsystem_Os, "listen() failed");
                RCF_THROW(e);
            }

            // retrieve the port number, if it's generated by the system
            if (mIpAddress.getPort() == 0)
            {
                IpAddress ip(mAcceptorFd, mIpAddress.getType());
                mIpAddress.setPort(ip.getPort());
            }

            RCF_LOG_2() << "TcpAsioServerTransport - listening on port " << mIpAddress.getPort() << ".";
        }
    }

    void TcpAsioServerTransport::onServerStart(RcfServer & server)
    {
        AsioServerTransport::onServerStart(server);

        mpIoService = mTaskEntries[0].getThreadPool().getIoService();

        if (mAcceptorFd != -1)
        {
            boost::asio::ip::tcp::acceptor::protocol_type protocolType = 
                boost::asio::ip::tcp::v4();

            if (mIpAddress.getType() == IpAddress::V6)
            {
                protocolType = boost::asio::ip::tcp::v6();
            }

            mAcceptorPtr.reset(
                new TcpAsioAcceptor(*mpIoService, protocolType, mAcceptorFd));

            mAcceptorFd = -1;

            startAccepting();
        }
    }

    ClientTransportAutoPtr TcpAsioServerTransport::implCreateClientTransport(
        const I_Endpoint &endpoint)
    {
        const TcpEndpoint &tcpEndpoint = 
            dynamic_cast<const TcpEndpoint &>(endpoint);

        ClientTransportAutoPtr clientTransportAutoPtr(
            new TcpClientTransport(tcpEndpoint.getIpAddress()));

        return clientTransportAutoPtr;
    }

} // namespace RCF
