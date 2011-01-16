
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

#include <RCF/EndpointServerService.hpp>

#include <RCF/CurrentSession.hpp>
#include <RCF/Exception.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>
#include <RCF/StubFactory.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    EndpointServer::EndpointServer() :
        mEndpointId(RCF_DEFAULT_INIT)
    {}

    EndpointServer::EndpointId EndpointServer::getEndpointId()
    {
        return mEndpointId;
    }

    // remotely invoked (on the master connection)
    bool EndpointServer::SpawnConnections(unsigned int requestedConnections)
    {
        if (requestedConnections > 10)
        {
            requestedConnections = 10;
        }

        I_ServerTransport &serverTransport =
            getCurrentRcfSession().getSessionState().getServerTransport();

        I_ServerTransportEx *pServerTransportEx =
            dynamic_cast<I_ServerTransportEx *>(&serverTransport);

        if (NULL == pServerTransportEx)
        {
            return false;
        }
        I_ServerTransportEx &serverTransportEx = *pServerTransportEx;
        for (unsigned int i=0; i<requestedConnections && mClients.size() < 50; ++i)
        {

            ClientTransportAutoPtr clientTransportAutoPtr(
                mClients.front()->getClientStub().getTransport().clone());

            EndpointBrokerClient client(clientTransportAutoPtr);

            // TODO: handle exceptions
            client.EstablishEndpointConnection(
                Oneway,
                mEndpointName,
                mEndpointServerPassword);

            ClientTransportAutoPtr apClientTransport(
                client.getClientStub().releaseTransport());

            SessionPtr sessionPtr = serverTransportEx.createServerSession(
                    apClientTransport,
                    StubEntryPtr());

            EndpointBrokerClientPtr clientPtr( 
                new EndpointBrokerClient(apClientTransport) );

            Lock lock(mClientsMutex);
            mClients.push_back(clientPtr);
        }
        return true;
    }

    EndpointServerService::EndpointServerService() :
        mEndpointServersMutex(WriterPriority)
    {}

    void EndpointServerService::onServiceAdded(RcfServer &server)
    {
        mServerTransportPtr = server.getServerTransportPtr();
    }

    void EndpointServerService::onServiceRemoved(RcfServer &)
    {
        WriteLock writeLock(mEndpointServersMutex);
        mEndpointServers.clear();
    }

    void EndpointServerService::onServerStop(RcfServer &)
    {
        WriteLock writeLock(mEndpointServersMutex);
        mEndpointServers.clear();
    }

    EndpointServerService::EndpointId EndpointServerService::openEndpoint(
        const I_Endpoint &brokerEndpoint,
        const std::string &endpointName)
    {
        I_ServerTransportEx &serverTransportEx =
            dynamic_cast<I_ServerTransportEx &>(*mServerTransportPtr);

        ClientTransportAutoPtr clientTransportAutoPtr(
            serverTransportEx.createClientTransport(brokerEndpoint));

        return openEndpoint(clientTransportAutoPtr, endpointName);
    }

    EndpointServerService::EndpointId EndpointServerService::openEndpoint(
        ClientTransportAutoPtr clientTransportAutoPtr,
        const std::string &endpointName)
    {
        EndpointId endpointId = 0;
        std::string endpointClientPassword = "";
        std::string endpointServerPassword = "";
        I_ServerTransportEx &serverTransportEx =
            dynamic_cast<I_ServerTransportEx &>(*mServerTransportPtr);

        EndpointServerPtr endpointServerPtr;
        {
            WriteLock lock(mEndpointServersMutex);

            // TODO: user configuration of the limit
            while (
                ++endpointId < 10 &&
                mEndpointServers.find(endpointId) != mEndpointServers.end());

            if (endpointId < 10)
            {
                endpointServerPtr.reset( new EndpointServer() );
                endpointServerPtr->mEndpointName = endpointName;
                endpointServerPtr->mEndpointId = endpointId;
                endpointServerPtr->mEndpointClientPassword = endpointClientPassword;
                endpointServerPtr->mEndpointServerPassword = endpointServerPassword;
                mEndpointServers[endpointServerPtr->getEndpointId()] = endpointServerPtr;
            }
        }
        if (endpointServerPtr)
        {
            RcfClient<I_EndpointBroker> client(clientTransportAutoPtr);

            boost::int32_t ret = client.OpenEndpoint(
                Twoway,
                endpointName,
                endpointClientPassword,
                endpointServerPassword);

            bool ok = (ret == RcfError_Ok);

            if (ok)
            {
                
                // break the cycle session->endpointserverstub->clientConnection->session
                EndpointServerWeakPtr endpointServerWeakPtr(endpointServerPtr);
                boost::shared_ptr< I_Deref<EndpointServer> > derefPtr(
                    new DerefWeakPtr<EndpointServer>(endpointServerWeakPtr) );

                RcfClientPtr rcfClientPtr(
                    createServerStub(
                        (I_EndpointServer *) 0,
                        (EndpointServer *) 0,
                        derefPtr) );
                
                StubEntryPtr stubEntryPtr( new StubEntry(rcfClientPtr));

                clientTransportAutoPtr.reset( 
                    client.getClientStub().releaseTransport().release());

                SessionPtr sessionPtr = serverTransportEx.createServerSession(
                        clientTransportAutoPtr,
                        stubEntryPtr);
           
                {
                    Lock lock(endpointServerPtr->mClientsMutex);
                    endpointServerPtr->mClients.push_back(
                        EndpointServer::EndpointBrokerClientPtr(
                            new EndpointServer::EndpointBrokerClient(clientTransportAutoPtr) ) );
                }

                return endpointServerPtr->getEndpointId();
            }
            else
            {
                WriteLock writeLock(mEndpointServersMutex);
                mEndpointServers.erase( mEndpointServers.find(
                    endpointServerPtr->getEndpointId()) );
            }
        }
        return EndpointId();
    }

    void EndpointServerService::closeEndpoint(EndpointId endpointId)
    {
        WriteLock writeLock(mEndpointServersMutex);
        mEndpointServers.erase( mEndpointServers.find(endpointId) );
    }

} // namespace RCF
