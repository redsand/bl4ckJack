
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

#ifndef INCLUDE_RCF_ENDPOINTSERVERSERVICE_HPP
#define INCLUDE_RCF_ENDPOINTSERVERSERVICE_HPP

#include <list>
#include <map>
#include <memory>
#include <string>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <RCF/Export.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/Service.hpp>

namespace RCF {

    class RcfServer;
    class I_Endpoint;
    class I_ServerTransport;
    class I_ClientTransport;

    typedef std::auto_ptr<I_ClientTransport> ClientTransportAutoPtr;

    class RCF_EXPORT EndpointServer
    {
    public:

        typedef int EndpointId;

        EndpointServer();

        EndpointId getEndpointId();

        // remotely invoked (on the master connection)
        bool SpawnConnections(unsigned int requestedConnections);

    private:

        friend class EndpointServerService;

        std::string                             mEndpointName;
        std::string                             mEndpointClientPassword;
        std::string                             mEndpointServerPassword;
        EndpointId                              mEndpointId;

        typedef RcfClient<I_EndpointBroker>                 EndpointBrokerClient;
        typedef boost::shared_ptr< EndpointBrokerClient >   EndpointBrokerClientPtr;

        // TODO: eliminate need for this mutex.
        Mutex                                   mClientsMutex;
        std::list<EndpointBrokerClientPtr>      mClients;
    };

    typedef boost::shared_ptr<EndpointServer>   EndpointServerPtr;
    typedef boost::weak_ptr<EndpointServer>     EndpointServerWeakPtr;

    class RCF_EXPORT EndpointServerService :
        public I_Service,
        boost::noncopyable
    {
    public:
        typedef EndpointServer::EndpointId EndpointId;
        EndpointServerService();

        void onServiceAdded(RcfServer &server);
        void onServiceRemoved(RcfServer &server);
        void onServerStop(RcfServer &server);
       
        EndpointId openEndpoint(
            const I_Endpoint &brokerEndpoint,
            const std::string &endpointName);
       
        EndpointId openEndpoint(
            ClientTransportAutoPtr clientTransportAutoPtr,
            const std::string &endpointName);
       
        void closeEndpoint(EndpointId endpointId);

    private:
        EndpointId getNextEndpointId();

        ServerTransportPtr                          mServerTransportPtr;
        ReadWriteMutex                              mEndpointServersMutex;
        std::map<EndpointId, EndpointServerPtr>     mEndpointServers;
    };

    typedef boost::shared_ptr<EndpointServerService> EndpointServerServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_ENDPOINTSERVERSERVICE_HPP
