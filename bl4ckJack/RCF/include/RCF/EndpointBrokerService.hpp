
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

#ifndef INCLUDE_RCF_ENDPOINTBROKERSERVICE_HPP
#define INCLUDE_RCF_ENDPOINTBROKERSERVICE_HPP

#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <RCF/ClientTransport.hpp>
#include <RCF/Export.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/ServerTransport.hpp>
#include <RCF/Service.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    class RCF_EXPORT EndpointBroker
    {
    public:

        EndpointBroker(
            ServerTransportPtr serverTransportPtr,
            const std::string &endpointName,
            const std::string &endpointClientPassword,
            const std::string &endpointServerPassword);

        boost::int32_t connectToEndpoint();

    private:
        friend class EndpointBrokerService;
        typedef RcfClient<I_EndpointServer> Client;
        typedef boost::shared_ptr<Client> ClientPtr;
        std::string mEndpointName;
        std::string mEndpointServerPassword;
        std::string mEndpointClientPassword;
        std::vector<RcfSessionPtr> mConnections;
        ClientPtr mMasterConnection;
        ServerTransportPtr mServerTransportPtr;
    };

    typedef boost::shared_ptr<EndpointBroker> EndpointBrokerPtr;

    class RCF_EXPORT EndpointBrokerService :
        public I_Service,
        boost::noncopyable
    {
    public:

        EndpointBrokerService();

        void onServiceAdded(RcfServer &server);
        void onServiceRemoved(RcfServer &server);
        void onServerStop(RcfServer &server);

        // remotely invoked
        boost::int32_t OpenEndpoint(
            const std::string &endpointName,
            const std::string &endpointClientPassword,
            std::string &endpointServerPassword);

        boost::int32_t CloseEndpoint(
            const std::string &endpointName,
            const std::string &endpointServerPassword);

        boost::int32_t EstablishEndpointConnection(
            const std::string &endpointName,
            const std::string &endpointServerPassword);

        boost::int32_t ConnectToEndpoint(
            const std::string &endpointName,
            const std::string &endpointClientPassword);

    private:
        ServerTransportPtr mServerTransportPtr;
        ReadWriteMutex mEndpointBrokersMutex;
        std::map<std::string, EndpointBrokerPtr> mEndpointBrokers;
    };

    typedef boost::shared_ptr<EndpointBrokerService> EndpointBrokerServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_ENDPOINTBROKERSERVICE_HPP
