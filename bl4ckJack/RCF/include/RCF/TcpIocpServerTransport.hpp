
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

#ifndef INCLUDE_RCF_TCPIOCPSERVERTRANSPORT_HPP
#define INCLUDE_RCF_TCPIOCPSERVERTRANSPORT_HPP

#include <boost/array.hpp>

#include <RCF/AsyncFilter.hpp>
#include <RCF/Iocp.hpp>
#include <RCF/Export.hpp>
#include <RCF/IocpServerTransport.hpp>
#include <RCF/IpServerTransport.hpp>

namespace RCF {

    class TcpIocpSessionState;

    class TcpIocpServerTransport;
    typedef boost::shared_ptr<TcpIocpServerTransport> TcpIocpServerTransportPtr;
    typedef boost::shared_ptr<TcpIocpSessionState> TcpIocpSessionStatePtr;

    // TcpIocpServerTransport

    class RCF_EXPORT TcpIocpServerTransport : 
        public IocpServerTransport, 
        public I_IpServerTransport
    {
    public:
        TcpIocpServerTransport(const IpAddress & ipAddress);
        TcpIocpServerTransport(const std::string & ip, int port);

        ServerTransportPtr      clone();

        int                     getPort() const;

        void                    setMaxPendingConnectionCount(
                                    std::size_t maxPendingConnectionCount);

        std::size_t             getMaxPendingConnectionCount() const;

    private:

        void                    onServiceAdded(RcfServer &server);
        void                    onServerStart(RcfServer &);

        IocpSessionStatePtr     implCreateServerSession(
                                    I_ClientTransport & clientTransport);

        ClientTransportAutoPtr  implCreateClientTransport(const I_Endpoint &endpoint);
        void                    implOpen();
        void                    implClose();

        IpAddress                       mIpAddress;
        Fd                              mAcceptorFd;
        Mutex                           mQueuedAcceptsMutex;
        Condition                       mQueuedAcceptsCondition;
        unsigned int                    mQueuedAccepts;
        const unsigned int              mQueuedAcceptsThreshold;
        const unsigned int              mQueuedAcceptsAugment;
        LPFN_ACCEPTEX                   mlpfnAcceptEx;
        LPFN_GETACCEPTEXSOCKADDRS       mlpfnGetAcceptExSockAddrs;
        std::size_t                     mMaxPendingConnectionCount;

        friend class TcpIocpSessionState;
    };

    // TcpIocpSessionState

    class RCF_EXPORT TcpIocpSessionState : public IocpSessionState
    {
    private:
        TcpIocpSessionState(TcpIocpServerTransport &tcpIocpServerTransport);
    public:
        typedef boost::shared_ptr<TcpIocpSessionState> TcpIocpSessionStatePtr;
        static TcpIocpSessionStatePtr create(TcpIocpServerTransport &tcpIocpServerTransport);
        ~TcpIocpSessionState();

        const I_RemoteAddress & getRemoteAddress();
        Fd                      getNativeHandle();

        void                    assignFd(Fd fd);
        void                    assignRemoteAddress(const IpAddress & ipAddress);

        void                    accept();

    private:

        void                    implOnAccept();
        
        void                    implRead(
                                    const ByteBuffer &byteBuffer, 
                                    std::size_t bufferLen);

        void                    implWrite(
                                    const std::vector<ByteBuffer> &byteBuffers, 
                                    IocpSessionState * pReflectee = NULL);

        ClientTransportAutoPtr  implCreateClientTransport();

        void                    implDelayCloseAfterSend();
        void                    implClose();
        bool                    implIsConnected();

        TcpIocpServerTransport &        mTransport;
        Fd                              mFd;
        IpAddress                       mLocalAddress;
        IpAddress                       mRemoteAddress;

        enum { AcceptExBufferSize = 2*(sizeof(SockAddrStorage) + 16) };
        boost::array<char, AcceptExBufferSize> mAcceptExBuffer;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_TCPIOCPSERVERTRANSPORT_HPP
