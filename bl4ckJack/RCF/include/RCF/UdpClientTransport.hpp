
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

#ifndef INCLUDE_RCF_UDPCLIENTTRANSPORT_HPP
#define INCLUDE_RCF_UDPCLIENTTRANSPORT_HPP

#include <RCF/ClientTransport.hpp>
#include <RCF/Export.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/IpClientTransport.hpp>

namespace RCF {

    class UdpClientTransport;

    typedef boost::shared_ptr<UdpClientTransport> UdpClientTransportPtr;
   
    class RCF_EXPORT UdpClientTransport : 
        public I_ClientTransport, 
        public I_IpClientTransport
    {
    private:
        IpAddress                                   mSrcIp;
        IpAddress                                   mDestIp;
        ReallocBufferPtr                            mReadVecPtr;
        ReallocBufferPtr                            mWriteVecPtr;
        bool                                        mAsync;
        int                                         mSock;

    public:
        UdpClientTransport(const IpAddress & ipAddress);
        UdpClientTransport(const UdpClientTransport &rhs);
         ~UdpClientTransport();

        ClientTransportAutoPtr 
                        clone() const;

        EndpointPtr     getEndpointPtr() const;

        void            connect(
                            I_ClientTransportCallback &clientStub, 
                            unsigned int timeoutMs);

        void            disconnect(unsigned int timeoutMs);

        int             send(
                            I_ClientTransportCallback &clientStub, 
                            const std::vector<ByteBuffer> &data, 
                            unsigned int timeoutMs);

        int             receive(
                            I_ClientTransportCallback &clientStub, 
                            ByteBuffer &byteBuffer, 
                            unsigned int timeoutMs);

        void            close();
        bool            isConnected();

        void            setTransportFilters(
                            const std::vector<FilterPtr> &filters);

        void            getTransportFilters(
                            std::vector<FilterPtr> &filters);

        int             getNativeHandle() const;

        void            setAsync(bool async);
        TimerEntry      setTimer(boost::uint32_t timeoutMs, I_ClientTransportCallback *pClientStub);
        void            killTimer(const TimerEntry & timerEntry);
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_UDPCLIENTTRANSPORT_HPP
