
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

#ifndef INCLUDE_RCF_PUBLISHINGSERVICE_HPP
#define INCLUDE_RCF_PUBLISHINGSERVICE_HPP

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <RCF/ClientStub.hpp>
#include <RCF/Export.hpp>
#include <RCF/GetInterfaceName.hpp>
#include <RCF/Service.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/Timer.hpp>

namespace RCF {

    class                                           RcfServer;
    class                                           I_RcfClient;
    class                                           RcfSession;
    class                                           I_ClientTransport;
    typedef boost::shared_ptr<I_RcfClient>          RcfClientPtr;
    typedef boost::shared_ptr<I_ClientTransport>    ClientTransportPtr;

    struct Publisher
    {
        Publisher(const std::string &name, RcfClientPtr rcfClientPtr) :
            mName(name),
            mMulticastRcfClientPtr(rcfClientPtr)
        {}
        std::string     mName;
        RcfClientPtr    mMulticastRcfClientPtr;
    };

    typedef boost::shared_ptr<Publisher> PublisherPtr;

    /// Service for implementing the publish part of publish/subscribe functionality.
    class RCF_EXPORT PublishingService :
        public I_Service,
        boost::noncopyable
    {
    public:
        PublishingService(boost::uint32_t pingIntervalMs = 0);

        /// Obtains a reference to the requested publisher object.
        /// \param Interface RCF interface of the publisher object.
        /// \param publisherName Name of the publisher object.
        /// \return Reference to the publisher object.

#if !defined(_MSC_VER) || _MSC_VER > 1200

        template<typename Interface>
        bool beginPublish(const std::string &publisherName = "")
        {
            return beginPublish( (Interface *) 0, publisherName);
        }

        template<typename Interface>
        typename Interface::RcfClientT &publish(
            const std::string &publisherName = "")
        {
            return publish( (Interface *) NULL, publisherName);
        }

        template<typename Interface>
        bool endPublish(const std::string &publisherName = "")
        {
            return endPublish( (Interface *) 0, publisherName);
        }

#endif

        /// Creates and stores a publisher object, i.e. a RcfClient<Interface> object with an underlying multicasting transport.
        /// \param Interface RCF interface to publish.
        /// \param publisherName Name through which this publisher will be accessed.
        /// \return True if ok, false otherwise.
        template<typename Interface>
        bool beginPublish(Interface *, const std::string &publisherName_ = "")
        {
            const std::string &publisherName = (publisherName_ == "") ?
                getInterfaceName((Interface *) NULL) :
                publisherName_;

            typedef typename Interface::RcfClientT RcfClientT;
            RcfClientPtr rcfClientPtr( new RcfClientT( ClientStub(publisherName)));
            return beginPublishNamed(publisherName, rcfClientPtr);
        }

        // inline definition here for reasons of portability (vc6)
        template<typename Interface>
        typename Interface::RcfClientT &publish(
            Interface *,
            const std::string &publisherName_ = "")
        {
            const std::string &publisherName = (publisherName_ == "") ?
                getInterfaceName( (Interface *) NULL) :
                publisherName_;

            return dynamic_cast<typename Interface::RcfClientT &>(
                publishNamed(publisherName) );
        }

        /// Shuts down the given publisher object.
        /// \param Interface RCF interface of the publisher object.
        /// \param publisherName Name of the publisher object.
        /// \return True if ok, false otherwise.
        template<typename Interface>
        bool endPublish(Interface *, const std::string &publisherName_ = "")
        {
            const std::string &publisherName = (publisherName_ == "") ?
                getInterfaceName((Interface *) NULL) :
                publisherName_;

            return endPublishNamed(publisherName);
        }

        typedef boost::function2<void, RcfSession &, const std::string &> OnConnectCallback;
        typedef boost::function2<void, RcfSession &, const std::string &> OnDisconnectCallback;

        void            setOnConnectCallback(
                            OnConnectCallback onConnectCallback);

        void            setOnDisconnectCallback(
                            OnDisconnectCallback onDisconnectCallback);

        // remotely accessible
        boost::int32_t  RequestSubscription(
                            const std::string &subscriptionName);

        boost::int32_t  RequestSubscription(
                            const std::string &subscriptionName,
                            boost::uint32_t subToPubPingIntervalMs,
                            boost::uint32_t & pubToSubPingIntervalMs);

    private:

        void            onServiceAdded(RcfServer &server);
        void            onServiceRemoved(RcfServer &server);
        void            onServerStop(RcfServer &server);

        bool            beginPublishNamed(
                            const std::string &publisherName,
                            RcfClientPtr rcfClientPtr);

        I_RcfClient &   publishNamed(
                            const std::string &publisherName);

        bool            endPublishNamed(
                            const std::string &publisherName);

        void            addSubscriberTransport(
                            RcfSession &session,
                            const std::string &publisherName,
                            ClientTransportAutoPtrPtr clientTransportAutoPtrPtr);

        typedef std::map<std::string, PublisherPtr>     Publishers;

        ReadWriteMutex                                  mPublishersMutex;
        Publishers                                      mPublishers;
        OnConnectCallback                               mOnConnectCallback;
        OnDisconnectCallback                            mOnDisconnectCallback;



        bool cycle(
            int                         timeoutMs,
            const volatile bool &       stopFlag);

        void stop();

        volatile bool                   mStopFlag;
        Timer                           mLastRunTimer;
        boost::uint32_t                 mPingIntervalMs;
    };

    typedef boost::shared_ptr<PublishingService> PublishingServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_PUBLISHINGSERVICE_HPP
