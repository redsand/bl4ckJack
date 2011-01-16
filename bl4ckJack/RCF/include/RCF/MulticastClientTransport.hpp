
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

#ifndef INCLUDE_RCF_MULTICASTCLIENTTRANSPORT_HPP
#define INCLUDE_RCF_MULTICASTCLIENTTRANSPORT_HPP

#include <list>
#include <memory>
#include <string>

#include <boost/shared_ptr.hpp>

#include <RCF/ClientTransport.hpp>
#include <RCF/Export.hpp>
#include <RCF/ThreadLibrary.hpp>

namespace RCF {

    typedef boost::shared_ptr< ClientTransportAutoPtr > ClientTransportAutoPtrPtr;

    /// Special purpose client transport for sending messages in parallel on multiple sub-transports.
    class RCF_EXPORT MulticastClientTransport : public I_ClientTransport
    {
    public:

        std::auto_ptr<I_ClientTransport> clone() const;

        EndpointPtr getEndpointPtr() const;

        int         send(
                        I_ClientTransportCallback &     clientStub, 
                        const std::vector<ByteBuffer> & data, 
                        unsigned int                    timeoutMs);

        int         receive(
                        I_ClientTransportCallback &     clientStub, 
                        ByteBuffer &                    byteBuffer, 
                        unsigned int                    timeoutMs);

        bool        isConnected();

        void        connect(
                        I_ClientTransportCallback &     clientStub, 
                        unsigned int                    timeoutMs);

        void        disconnect(
                        unsigned int                    timeoutMs);

        void        addTransport(
                        ClientTransportAutoPtr          clientTransportAutoPtr);

        void        setTransportFilters(
                        const std::vector<FilterPtr> &  filters);

        void        getTransportFilters(
                        std::vector<FilterPtr> &        filters);

        void        setAsync(bool async);
        TimerEntry  setTimer(boost::uint32_t timeoutMs, I_ClientTransportCallback *pClientStub);
        void        killTimer(const TimerEntry & timerEntry);

        void        dropIdleTransports();
        void        pingAllTransports();

    private:

        void        bringInNewTransports();

        typedef std::list< ClientTransportAutoPtrPtr >     ClientTransportList;

        Mutex                                           mClientTransportsMutex;
        ClientTransportList                             mClientTransports;

        Mutex                                           mAddedClientTransportsMutex;
        ClientTransportList                             mAddedClientTransports;

        ClientTransportAutoPtr                          mMulticastTemp;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_MULTICASTCLIENTTRANSPORT_HPP
