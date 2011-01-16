
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

#include <RCF/MulticastClientTransport.hpp>

#include <RCF/ClientStub.hpp>
#include <RCF/Exception.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    ClientTransportAutoPtr MulticastClientTransport::clone() const
    {
        RCF_ASSERT(0);
        return ClientTransportAutoPtr();
    }

    EndpointPtr MulticastClientTransport::getEndpointPtr() const
    {
        RCF_ASSERT(0);
        return EndpointPtr();
    }

    class DummyCallback : public I_ClientTransportCallback
    {
    public:
        void onConnectCompleted(bool alreadyConnected = false)
        {
            RCF_UNUSED_VARIABLE(alreadyConnected);
        }

        void onSendCompleted()
        {}
        
        void onReceiveCompleted()
        {}

        void onTimerExpired()
        {}
        
        void onError(const std::exception &e)
        {
            RCF_UNUSED_VARIABLE(e);
        }
    };

    int MulticastClientTransport::send(
        I_ClientTransportCallback &clientStub,
        const std::vector<ByteBuffer> &data,
        unsigned int timeoutMs)
    {
        // TODO: in some cases, may need to make a full copy of data for 
        // each individual sub-transport, as they might transform the data.

        bringInNewTransports();

        Lock lock(mClientTransportsMutex);

        // TODO: hardcoded
        timeoutMs = 1000;
        bool needToRemove = false;

        ClientTransportList::iterator iter;
        for (
            iter = mClientTransports.begin();
            iter != mClientTransports.end();
            ++iter)
        {
            try
            {
                if ((**iter)->isConnected())
                {
                    // Sending synchronously, so no use for the callback
                    DummyCallback dummyCallback;
                    (**iter)->send(dummyCallback, data, timeoutMs);
                }
                else
                {
                    needToRemove = true;
                    iter->reset();
                }
            }
            catch(const Exception &e)
            {
                RCF_LOG_1()(e) << "Error publishing to subscriber.";
                needToRemove = true;
                iter->reset();
            }
        }

        if (needToRemove)
        {
            mClientTransports.remove( ClientTransportAutoPtrPtr() );
        }       

        clientStub.onSendCompleted();

        return 1;
    }

    int MulticastClientTransport::receive(
        I_ClientTransportCallback &clientStub,
        ByteBuffer &byteBuffer,
        unsigned int timeoutMs)
    {
        RCF_UNUSED_VARIABLE(clientStub);
        RCF_UNUSED_VARIABLE(byteBuffer);
        RCF_UNUSED_VARIABLE(timeoutMs);
        RCF_ASSERT(0);
        return 1;
    }

    bool MulticastClientTransport::isConnected()
    {
        return true;
    }

    void MulticastClientTransport::connect(I_ClientTransportCallback &clientStub, unsigned int timeoutMs)
    {
        RCF_UNUSED_VARIABLE(clientStub);
        RCF_UNUSED_VARIABLE(timeoutMs);
        clientStub.onConnectCompleted(true);
    }

    void MulticastClientTransport::disconnect(unsigned int timeoutMs)
    {
        RCF_UNUSED_VARIABLE(timeoutMs);
    }

    void MulticastClientTransport::addTransport(
        ClientTransportAutoPtr clientTransportAutoPtr)
    {
        Lock lock(mAddedClientTransportsMutex);

        mAddedClientTransports.push_back( ClientTransportAutoPtrPtr( 
            new ClientTransportAutoPtr(clientTransportAutoPtr) ) );
    }

    void MulticastClientTransport::bringInNewTransports()
    {
        Lock lock1(mClientTransportsMutex);

        Lock lock2(mAddedClientTransportsMutex);

        std::copy(
            mAddedClientTransports.begin(),
            mAddedClientTransports.end(),
            std::back_inserter(mClientTransports));

        mAddedClientTransports.resize(0);
    }

    void MulticastClientTransport::setTransportFilters(
        const std::vector<FilterPtr> &)
    {
        // not supported
    }

    void MulticastClientTransport::getTransportFilters(
        std::vector<FilterPtr> &)
    {
        // not supported
    }

    void MulticastClientTransport::setAsync(bool async)
    {
        RCF_ASSERT(!async);
    }

    MulticastClientTransport::TimerEntry MulticastClientTransport::setTimer(
        boost::uint32_t timeoutMs,
        I_ClientTransportCallback *pClientStub)
    {
        RCF_UNUSED_VARIABLE(timeoutMs);
        RCF_UNUSED_VARIABLE(pClientStub);
        return TimerEntry();
    }

    void MulticastClientTransport::killTimer(const TimerEntry & timerEntry)
    {
        RCF_UNUSED_VARIABLE(timerEntry);
    }

    void MulticastClientTransport::dropIdleTransports()
    {
        bringInNewTransports();

        Lock lock(mClientTransportsMutex);

        bool needToRemove = false;

        ClientTransportList::iterator iter;
        for (iter = mClientTransports.begin(); iter != mClientTransports.end(); ++iter)
        {
            RCF::I_ClientTransport & transport = ***iter;

            RcfSessionWeakPtr rcfSessionWeakPtr = transport.getRcfSession();
            RcfSessionPtr rcfSessionPtr = rcfSessionWeakPtr.lock();
            if (!rcfSessionPtr)
            {
                RCF_LOG_2() << "Dropping subscription. Subscriber has closed connection.";
                iter->reset();
                needToRemove = true;
            }
            else
            {
                {
                    boost::uint32_t pingIntervalMs = rcfSessionPtr->getPingIntervalMs();
                    if (pingIntervalMs)
                    {
                        RCF::Timer pingTimer( rcfSessionPtr->getPingTimestamp() );
                        if (pingTimer.elapsed(5000 + 2*pingIntervalMs))
                        {
                            std::string subscriberUrl = rcfSessionPtr->getRemoteAddress().string();
                            RCF_LOG_2()(subscriberUrl)(pingIntervalMs) << "Dropping subscription. Subscriber has not sent pings.";
                            iter->reset();
                            needToRemove = true;
                        }
                    }
                }
            }
        }

        if (needToRemove)
        {
            mClientTransports.remove( ClientTransportAutoPtrPtr() );
        }
    }

    void MulticastClientTransport::pingAllTransports()
    {
        bringInNewTransports();

        Lock lock(mClientTransportsMutex);

        if (!mMulticastTemp.get())
        {
            mMulticastTemp.reset( new MulticastClientTransport() );
        }

        MulticastClientTransport & multicastTemp = 
            static_cast<MulticastClientTransport &>(*mMulticastTemp);

        multicastTemp.mClientTransports.resize(0);

        ClientTransportList::iterator iter;
        for (iter = mClientTransports.begin(); iter != mClientTransports.end(); ++iter)
        {
            I_ClientTransport & transport = ***iter;
            RcfSessionPtr rcfSessionPtr = transport.getRcfSession().lock();
            if (rcfSessionPtr)
            {
                boost::uint32_t pingIntervalMs = rcfSessionPtr->getPingIntervalMs();
                if (pingIntervalMs)
                {
                    multicastTemp.mClientTransports.push_back(*iter);
                }
            }
        }

        RcfClient<I_Null> nullClient( mMulticastTemp );
        nullClient.getClientStub().ping(RCF::Oneway);
        mMulticastTemp.reset( nullClient.getClientStub().releaseTransport().release() );
        multicastTemp.mClientTransports.resize(0);
    }

} // namespace RCF
