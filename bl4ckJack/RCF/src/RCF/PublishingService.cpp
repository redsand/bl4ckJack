
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

#include <RCF/PublishingService.hpp>

#include <RCF/CurrentSession.hpp>
#include <RCF/MulticastClientTransport.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/ServerTransport.hpp>

#ifdef RCF_USE_PROTOBUF
#include <RCF/protobuf/RcfMessages.pb.h>
#endif

#include <RCF/util/Platform/OS/Sleep.hpp>

namespace RCF {

    PublishingService::PublishingService(boost::uint32_t pingIntervalMs) :
        mPublishersMutex(WriterPriority),
        mPingIntervalMs(pingIntervalMs)
    {}

    bool PublishingService::beginPublishNamed(
        const std::string &publisherName,
        RcfClientPtr rcfClientPtr)
    {
        rcfClientPtr->getClientStub().setTransport(ClientTransportAutoPtr(new MulticastClientTransport));
        rcfClientPtr->getClientStub().setDefaultCallingSemantics(Oneway);
        rcfClientPtr->getClientStub().setTargetName("");
        rcfClientPtr->getClientStub().setTargetToken( Token());
        PublisherPtr publisherPtr( new Publisher(publisherName, rcfClientPtr));
        WriteLock lock(mPublishersMutex);
        mPublishers[publisherName] = publisherPtr;
        return true;
    }

    I_RcfClient &PublishingService::publishNamed(const std::string &publisherName)
    {
        ReadLock lock(mPublishersMutex);
        if (mPublishers.find(publisherName) != mPublishers.end())
        {
            return *mPublishers[ publisherName ]->mMulticastRcfClientPtr;
        }
        Exception e(_RcfError_UnknownPublisher(publisherName));
        RCF_THROW(e);
    }

    bool PublishingService::endPublishNamed(const std::string &publisherName)
    {
        WriteLock lock(mPublishersMutex);
        Publishers::iterator iter = mPublishers.find(publisherName);
        if (iter != mPublishers.end())
        {
            mPublishers.erase(iter);
        }
        return true;
    }

    void PublishingService::setOnConnectCallback(
        OnConnectCallback onConnectCallback)
    {
        WriteLock lock(mPublishersMutex);
        mOnConnectCallback = onConnectCallback;
    }

    void PublishingService::setOnDisconnectCallback(
        OnDisconnectCallback onDisconnectCallback)
    {
        WriteLock lock(mPublishersMutex);
        mOnDisconnectCallback = onDisconnectCallback;
    }

    void vc6_boost_bind_helper_2(
        PublishingService::OnDisconnectCallback onDisconnect, 
        RcfSession & rcfSession,
        const std::string & subscriptionName )
    {
        onDisconnect(rcfSession, subscriptionName);
    }

    // remotely accessible

    boost::int32_t PublishingService::RequestSubscription(
        const std::string &subscriptionName)
    {
        boost::uint32_t subToPubPingIntervalMs = 0;
        boost::uint32_t pubToSubPingIntervalMs = 0;

        return RequestSubscription(
            subscriptionName, 
            subToPubPingIntervalMs, 
            pubToSubPingIntervalMs);
    }

    boost::int32_t PublishingService::RequestSubscription(
        const std::string &subscriptionName,
        boost::uint32_t subToPubPingIntervalMs,
        boost::uint32_t & pubToSubPingIntervalMs)
    {
        std::string publisherName = subscriptionName;
        bool found = false;
        ReadLock lock(mPublishersMutex);
        if (mPublishers.find(publisherName) != mPublishers.end())
        {
            found = true;
        }
        lock.unlock();
        if (found)
        {
            RcfSession & rcfSession = getCurrentRcfSession();

            if (mOnConnectCallback)
            {
                mOnConnectCallback(rcfSession, subscriptionName);
            }            

            I_ServerTransportEx &serverTransport =
                dynamic_cast<I_ServerTransportEx &>(
                    rcfSession.getSessionState().getServerTransport());

            ClientTransportAutoPtrPtr clientTransportAutoPtrPtr(
                new ClientTransportAutoPtr(
                    serverTransport.createClientTransport(
                        rcfSession.shared_from_this())));

            (*clientTransportAutoPtrPtr)->setRcfSession(
                rcfSession.shared_from_this());

            rcfSession.setPingIntervalMs(subToPubPingIntervalMs);

            rcfSession.addOnWriteCompletedCallback(
                boost::bind(
                    &PublishingService::addSubscriberTransport,
                    this,
                    _1,
                    publisherName,
                    clientTransportAutoPtrPtr) );

            if (mOnDisconnectCallback)
            {

#if defined(_MSC_VER) && _MSC_VER < 1310

                rcfSession.setOnDestroyCallback(
                    boost::bind(vc6_boost_bind_helper_2, mOnDisconnectCallback, _1, subscriptionName));

#else

                rcfSession.setOnDestroyCallback(
                    boost::bind(mOnDisconnectCallback, _1, subscriptionName));

#endif

            }            
        }  
        pubToSubPingIntervalMs = mPingIntervalMs;
        return found ? RcfError_Ok : RcfError_Unspecified;
    }

#ifdef RCF_USE_PROTOBUF

    class PublishingServicePb
    {
    public:

        PublishingServicePb(PublishingService & ps) : mPs(ps)
        {
        }

        void RequestSubscription(const PbRequestSubscription & request)
        {
            int error = mPs.RequestSubscription(request.subscriptionname());

            if (error != RCF::RcfError_Ok)
            {
                RemoteException e(( Error(error) ));
                RCF_THROW(e);
            }
        }

    private:
        PublishingService & mPs;
    };

    void onServiceAddedProto(PublishingService & ps, RcfServer & server)
    {
        boost::shared_ptr<PublishingServicePb> psPbPtr(
            new PublishingServicePb(ps));

        server.bind((I_RequestSubscriptionPb *) NULL, psPbPtr);
    }

    void onServiceRemovedProto(PublishingService & ps, RcfServer & server)
    {
        server.unbind( (I_RequestSubscriptionPb *) NULL);
    }

#else

    void onServiceAddedProto(PublishingService &, RcfServer &)
    {
    }

    void onServiceRemovedProto(PublishingService &, RcfServer &)
    {
    }

#endif // RCF_USE_PROTOBUF

    void PublishingService::onServiceAdded(RcfServer & server)
    {
        server.bind( (I_RequestSubscription *) NULL, *this);

        onServiceAddedProto(*this, server);

        mStopFlag = false;
        mLastRunTimer.restart(Platform::OS::getCurrentTimeMs() - 2*mPingIntervalMs);

        if (mPingIntervalMs)
        {
            mTaskEntries.clear();

            mTaskEntries.push_back( TaskEntry(
                boost::bind(&PublishingService::cycle, this, _1, _2),
                boost::bind(&PublishingService::stop, this),
                "RCF Publishing Timeout"));
        }
    }

    void PublishingService::onServiceRemoved(RcfServer &server)
    {
        server.unbind( (I_RequestSubscription *) NULL);

        onServiceRemovedProto(*this, server);
    }

    void PublishingService::onServerStop(RcfServer &server)
    {
        // need to do this now, rather than implicitly, when RcfServer is destroyed, because
        // the client transport objects have links to the server transport (close functor)
        RCF_UNUSED_VARIABLE(server);
        WriteLock writeLock(mPublishersMutex);
        this->mPublishers.clear();
    }

    void PublishingService::addSubscriberTransport(
        RcfSession &session,
        const std::string &publisherName,
        ClientTransportAutoPtrPtr clientTransportAutoPtrPtr)
    {
        session.setPingTimestamp();

        WriteLock lock(mPublishersMutex);
        if (mPublishers.find(publisherName) != mPublishers.end())
        {
            I_ClientTransport &clientTransport =
                mPublishers[ publisherName ]->mMulticastRcfClientPtr->
                    getClientStub().getTransport();

            MulticastClientTransport &multiCastClientTransport =
                dynamic_cast<MulticastClientTransport &>(clientTransport);

            multiCastClientTransport.addTransport(*clientTransportAutoPtrPtr);
        }
    }

    bool PublishingService::cycle(
        int timeoutMs,
        const volatile bool &stopFlag)
    {
        RCF_UNUSED_VARIABLE(timeoutMs);

        if (    !mLastRunTimer.elapsed(mPingIntervalMs)
            &&  !stopFlag 
            &&  !mStopFlag)
        {
            Platform::OS::Sleep(1);
            return false;
        }

        mLastRunTimer.restart();

        WriteLock lock(mPublishersMutex);

        // Publish a oneway ping.
        Publishers::iterator iter;
        for (iter = mPublishers.begin(); iter != mPublishers.end(); ++iter)
        {
            Publisher & pub = * iter->second;

            I_ClientTransport & transport = 
                pub.mMulticastRcfClientPtr->getClientStub().getTransport();

            MulticastClientTransport & multiTransport = 
                static_cast<MulticastClientTransport &>(transport);

            multiTransport.pingAllTransports();
        }

        // Check ping timestamps on all subscribers, and kill off any stragglers.
        for (iter = mPublishers.begin(); iter != mPublishers.end(); ++iter)
        {
            Publisher & pub = * iter->second;

            I_ClientTransport & transport = 
                pub.mMulticastRcfClientPtr->getClientStub().getTransport();

            MulticastClientTransport & multiTransport = 
                static_cast<MulticastClientTransport &>(transport);

            multiTransport.dropIdleTransports();
        }

        return stopFlag || mStopFlag;
    }

    void PublishingService::stop()
    {
        mStopFlag = true;
    }
   
} // namespace RCF
