
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

#include <RCF/SubscriptionService.hpp>

#include <boost/bind.hpp>

#include <typeinfo>

#include <RCF/ClientTransport.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>

#include <RCF/util/Platform/OS/Sleep.hpp>

namespace RCF {

    bool Subscription::isConnected()
    {
        Lock lock(mMutex);
        return
            mClientTransportAutoPtr.get() &&
            mClientTransportAutoPtr->isConnected();
    }

    unsigned int Subscription::getPingTimestamp()
    {
        RcfSessionPtr rcfSessionPtr;
        {
            Lock lock(mMutex);
            rcfSessionPtr = mRcfSessionWeakPtr.lock();
        }
        if (rcfSessionPtr)
        {
            return rcfSessionPtr->getPingTimestamp();
        }
        return 0;
    }

    void Subscription::close()
    {
        Lock lock(mMutex);

        {
            RcfSessionPtr rcfSessionPtr(mRcfSessionWeakPtr.lock());
            if (rcfSessionPtr)
            {
                rcfSessionPtr->setOnDestroyCallback(
                    RcfSession::OnDestroyCallback());
            }
        }
        mRcfSessionWeakPtr.reset();
        mClientTransportAutoPtr.reset();
    }

    RcfSessionPtr Subscription::getRcfSessionPtr()
    {
        Lock lock(mMutex);
        return mRcfSessionWeakPtr.lock();
    }

    SubscriptionService::SubscriptionService(boost::uint32_t pingIntervalMs) :
        mpServer(RCF_DEFAULT_INIT),
        mSubscriptionsMutex(WriterPriority),
        mPingIntervalMs(pingIntervalMs)
    {}

    void vc6_boost_bind_helper_1(
        SubscriptionService::OnDisconnect onDisconnect, 
        RcfSession & rcfSession)
    {
        onDisconnect(rcfSession);
    }

    SubscriptionPtr SubscriptionService::onRequestSubscriptionCompleted(
        boost::int32_t                      ret,
        SubscriptionId                      whichSubscription,
        RcfClient<I_RequestSubscription> &  client,
        RcfClientPtr                        rcfClientPtr,
        OnDisconnect                        onDisconnect,
        boost::uint32_t                     pubToSubPingIntervalMs,
        bool                                pingsEnabled)
    {
        bool ok = (ret == RcfError_Ok);
        if (ok)
        {
            I_ServerTransportEx &serverTransportEx =
                dynamic_cast<I_ServerTransportEx &>(*mServerTransportPtr);

            ClientTransportAutoPtr clientTransportAutoPtr(
                client.getClientStub().releaseTransport());

            SessionPtr sessionPtr = serverTransportEx.createServerSession(
                clientTransportAutoPtr,
                StubEntryPtr(new StubEntry(rcfClientPtr)));

            RCF_ASSERT( sessionPtr );

            RcfSessionPtr rcfSessionPtr = sessionPtr;

            rcfSessionPtr->setUserData(client.getClientStub().getUserData());
            rcfSessionPtr->setPingTimestamp();

            if (onDisconnect)
            {

#if defined(_MSC_VER) && _MSC_VER < 1310

                rcfSessionPtr->setOnDestroyCallback(
                    boost::bind(vc6_boost_bind_helper_1, onDisconnect, _1));

#else

                rcfSessionPtr->setOnDestroyCallback(
                    onDisconnect);

#endif

            }

            std::string publisherUrl;
            EndpointPtr epPtr = client.getClientStub().getEndpoint();
            if (epPtr)
            {
                publisherUrl = epPtr->asString();
            }

            SubscriptionPtr subscriptionPtr( new Subscription(
                clientTransportAutoPtr, 
                rcfSessionPtr, 
                pubToSubPingIntervalMs, 
                publisherUrl,
                whichSubscription.first));

            subscriptionPtr->mPingsEnabled = pingsEnabled;

            return subscriptionPtr;                
        }
        return SubscriptionPtr();
    }

    SubscriptionPtr SubscriptionService::beginSubscribeNamed(
        SubscriptionId      whichSubscription,
        RcfClientPtr        rcfClientPtr,
        ClientStub &        clientStub,
        OnDisconnect        onDisconnect,
        const std::string & publisherName)
    {
        RcfClient<I_RequestSubscription> client(clientStub);
        client.getClientStub().setTransport(clientStub.releaseTransport());
        boost::uint32_t subToPubPingIntervalMs = mPingIntervalMs;
        boost::uint32_t pubToSubPingIntervalMs = 0;

        bool pingsEnabled = true;

        boost::int32_t ret = 0;
        if (clientStub.getRuntimeVersion() < 8)
        {
            pingsEnabled = false;

            ret = client.RequestSubscription(
                Twoway, 
                publisherName);
        }
        else
        {
            ret = client.RequestSubscription(
                Twoway, 
                publisherName, 
                subToPubPingIntervalMs, 
                pubToSubPingIntervalMs);
        }

        SubscriptionPtr subscriptionPtr = onRequestSubscriptionCompleted(
            ret,
            whichSubscription,
            client,
            rcfClientPtr,
            onDisconnect,
            pubToSubPingIntervalMs,
            pingsEnabled);

        if (subscriptionPtr)
        {
            WriteLock lock(mSubscriptionsMutex);
            mSubscriptions[ whichSubscription ] = subscriptionPtr;                
        }
        return subscriptionPtr;
    }

    void SubscriptionService::beginSubscribeNamedCb(
        Subscription::AsyncClientPtr    clientPtr,
        Future<boost::int32_t>          fRet,
        SubscriptionId                  whichSubscription,
        RcfClientPtr                    rcfClientPtr,
        OnDisconnect                    onDisconnect,
        boost::function2<void, SubscriptionPtr, ExceptionPtr> onCompletion,
        Future<boost::uint32_t>         incomingPingIntervalMs,
        bool                            pingsEnabled)
    {
        SubscriptionPtr subscriptionPtr;

        ExceptionPtr exceptionPtr(
            clientPtr->getClientStub().getAsyncException().release());

        if (!exceptionPtr)
        {
            boost::int32_t ret = fRet;

            subscriptionPtr = onRequestSubscriptionCompleted(
                ret,
                whichSubscription,
                *clientPtr,
                rcfClientPtr,
                onDisconnect,
                incomingPingIntervalMs,
                pingsEnabled);

            if (subscriptionPtr)
            {
                WriteLock lock(mSubscriptionsMutex);
                mSubscriptions[ whichSubscription ] = subscriptionPtr;
            }
        }

        onCompletion(subscriptionPtr, exceptionPtr);
    }

    void SubscriptionService::beginSubscribeNamed(
        SubscriptionId                  whichSubscription,
        RcfClientPtr                    rcfClientPtr,
        ClientStub &                    clientStub,
        OnDisconnect                    onDisconnect,
        const std::string &             publisherName,
        boost::function2<void, SubscriptionPtr, ExceptionPtr> onCompletion)
    {

        Subscription::AsyncClientPtr asyncClientPtr( 
            new Subscription::AsyncClient(clientStub));

        asyncClientPtr->getClientStub().setTransport(
            clientStub.releaseTransport());

        asyncClientPtr->getClientStub().setAsyncDispatcher(*mpServer);
      
        Future<boost::int32_t>      ret;
        boost::uint32_t             outgoingPingIntervalMs = mPingIntervalMs;
        Future<boost::uint32_t>     incomingPingIntervalMs;

        bool pingsEnabled = true;

        if (clientStub.getRuntimeVersion() < 8)
        {
            pingsEnabled = false;

            ret = asyncClientPtr->RequestSubscription(

                AsyncTwoway( boost::bind( 
                    &SubscriptionService::beginSubscribeNamedCb, 
                    this,
                    asyncClientPtr,
                    ret,
                    whichSubscription, 
                    rcfClientPtr,
                    onDisconnect,
                    onCompletion,
                    incomingPingIntervalMs,
                    pingsEnabled)),

                publisherName);
        }
        else
        {
            ret = asyncClientPtr->RequestSubscription(

                AsyncTwoway( boost::bind( 
                    &SubscriptionService::beginSubscribeNamedCb, 
                    this,
                    asyncClientPtr,
                    ret,
                    whichSubscription, 
                    rcfClientPtr,
                    onDisconnect,
                    onCompletion,
                    incomingPingIntervalMs,
                    pingsEnabled)),

                publisherName,
                outgoingPingIntervalMs,
                incomingPingIntervalMs);

        }
    }

    SubscriptionPtr SubscriptionService::beginSubscribeNamed(
        SubscriptionId          whichSubscription,
        RcfClientPtr            rcfClientPtr,
        const I_Endpoint &      publisherEndpoint,
        OnDisconnect            onDisconnect,
        const std::string &     publisherName)
    {
        ClientStub clientStub("");
        clientStub.setEndpoint(publisherEndpoint);
        clientStub.instantiateTransport();

        return
            beginSubscribeNamed(
                whichSubscription,
                rcfClientPtr,
                clientStub,
                onDisconnect,
                publisherName);
    }

    SubscriptionPtr SubscriptionService::beginSubscribeNamed(
        SubscriptionId          whichSubscription,
        RcfClientPtr            rcfClientPtr,
        ClientTransportAutoPtr  clientTransportAutoPtr,
        OnDisconnect            onDisconnect,
        const std::string &     publisherName)
    {
        ClientStub clientStub("");
        clientStub.setTransport(clientTransportAutoPtr);

        return
            beginSubscribeNamed(
                whichSubscription,
                rcfClientPtr,
                clientStub,
                onDisconnect,
                publisherName);

    }

    bool SubscriptionService::endSubscribeNamed(
        SubscriptionId whichSubscription)
    {
        SubscriptionPtr subscriptionPtr;
        {
            WriteLock lock(mSubscriptionsMutex);
            subscriptionPtr = mSubscriptions[ whichSubscription ];
            mSubscriptions.erase(whichSubscription);
        }
        if (subscriptionPtr)
        {
            RcfSessionPtr rcfSessionPtr =
                subscriptionPtr->mRcfSessionWeakPtr.lock();

            if (rcfSessionPtr)
            {
                // When this function returns, the caller is allowed to delete
                // the object that this subscription refers to. Hence, at this
                // point, we have to block any current published call that may 
                // be in execution, or else wait for it to complete.

                Lock lock(rcfSessionPtr->mStopCallInProgressMutex);
                rcfSessionPtr->mStopCallInProgress = true;

                // Remove subscription binding.
                rcfSessionPtr->setDefaultStubEntryPtr(StubEntryPtr());

                // Clear the destroy callback.
                // TODO: how do we know that we're not clearing someone else's callback?
                rcfSessionPtr->setOnDestroyCallback(
                    RcfSession::OnDestroyCallback());
            }
        }
        return true;
    }

    SubscriptionPtr SubscriptionService::getSubscriptionPtr(
        SubscriptionId whichSubscription)
    {
        ReadLock lock(mSubscriptionsMutex);
        Subscriptions::iterator iter = mSubscriptions.find(whichSubscription);
        return iter != mSubscriptions.end() ? 
            iter->second : 
            SubscriptionPtr();
    }

    void SubscriptionService::onServiceAdded(RcfServer &server)
    {
        mpServer = &server;
        mServerTransportPtr = server.getServerTransportPtr();

        mStopFlag = false;
        mLastRunTimer.restart(Platform::OS::getCurrentTimeMs() - 2*mPingIntervalMs);

        if (mPingIntervalMs)
        {
            mTaskEntries.clear();

            mTaskEntries.push_back( TaskEntry(
                boost::bind(&SubscriptionService::cycle, this, _1, _2),
                boost::bind(&SubscriptionService::stop, this),
                "RCF Subscription Timeout"));
        }
            
    }

    void SubscriptionService::onServiceRemoved(RcfServer &)
    {}

    void SubscriptionService::onServerStop(RcfServer &server)
    {
        RCF_UNUSED_VARIABLE(server);
        WriteLock writeLock(mSubscriptionsMutex);

        for (Subscriptions::iterator iter = mSubscriptions.begin();
            iter != mSubscriptions.end();
            ++iter)
        {
            SubscriptionPtr subscriptionPtr = iter->second;
            subscriptionPtr->close();
        }

        mSubscriptions.clear();
    }

    bool SubscriptionService::cycle(
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

        Subscriptions subsToDrop;

        {
            WriteLock lock(mSubscriptionsMutex);

            ClientTransportAutoPtr dummy;
            RcfClient<I_Null> nullClient(dummy);

            // Send pings on all our subscriptions.
            Subscriptions::iterator iter;
            for (iter = mSubscriptions.begin(); iter != mSubscriptions.end(); ++iter)
            {
                Subscription & sub = * iter->second;
                if (sub.mPingsEnabled)
                {
                    Lock lock(sub.mMutex);

                    nullClient.getClientStub().setTransport(sub.mClientTransportAutoPtr);
                    nullClient.getClientStub().setConnected(true);
                    nullClient.getClientStub().setAutoReconnect(false);

                    nullClient.getClientStub().ping(RCF::Oneway);
                    sub.mClientTransportAutoPtr.reset( nullClient.getClientStub().releaseTransport().release() );
                }
            }

            // Kill off any subscriptions that haven't received any recent pings.
            for (iter = mSubscriptions.begin(); iter != mSubscriptions.end(); ++iter)
            {
                Subscription & sub = * iter->second;

                Lock lock(sub.mMutex);
                RcfSessionPtr sessionPtr = sub.getRcfSessionPtr();
                
                if (!sessionPtr)
                {
                    RCF_LOG_2()(sub.mPublisherUrl)(sub.mTopic) << "Dropping subscription. Publisher has closed connection.";
                    subsToDrop[iter->first] = iter->second;
                }
                else if (sub.mPingsEnabled)
                {
                    boost::uint32_t pingIntervalMs = sub.mPingIntervalMs;
                    if (pingIntervalMs)
                    {
                        RCF::Timer pingTimer(sessionPtr->getPingTimestamp());
                        if (pingTimer.elapsed(5000 + 2*pingIntervalMs))
                        {
                            RCF_LOG_2()(sub.mPublisherUrl)(sub.mTopic)(sub.mPingIntervalMs) << "Dropping subscription. Publisher has not sent pings.";
                            subsToDrop[iter->first] = iter->second;
                        }
                    }
                }
            }

            for (iter = subsToDrop.begin(); iter != subsToDrop.end(); ++iter)
            {
                mSubscriptions.erase(iter->first);
            }
        }

        subsToDrop.clear();
        
        return stopFlag || mStopFlag;
    }

    void SubscriptionService::stop()
    {
        mStopFlag = true;
    }

    Subscription::Subscription(
        ClientTransportAutoPtr clientTransportAutoPtr,
        RcfSessionWeakPtr rcfSessionWeakPtr,
        boost::uint32_t incomingPingIntervalMs,
        const std::string & publisherUrl,
        const std::string & topic) :
            mClientTransportAutoPtr(clientTransportAutoPtr),
            mRcfSessionWeakPtr(rcfSessionWeakPtr),
            mPingIntervalMs(incomingPingIntervalMs),
            mPublisherUrl(publisherUrl),
            mTopic(topic)
    {}
   
} // namespace RCF
