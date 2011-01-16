
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

#include <RCF/SessionObjectFactoryService.hpp>

#include <RCF/ServerInterfaces.hpp>

#ifdef RCF_USE_PROTOBUF
#include <RCF/protobuf/RcfMessages.pb.h>
#endif

namespace RCF {

#ifdef RCF_USE_PROTOBUF

    class SessionObjectFactoryServicePb
    {
    public:

        SessionObjectFactoryServicePb(SessionObjectFactoryService & sofs) : mSofs(sofs)
        {

        }

        void CreateSessionObject(const PbCreateSessionObject & request)
        {
            int error = mSofs.CreateSessionObject(request.objectname());

            if (error != RCF::RcfError_Ok)
            {
                RemoteException e(( Error(error) ));
                RCF_THROW(e);
            }
        }

        void DeleteSessionObject(const PbDeleteSessionObject & request)
        {
            RCF_UNUSED_VARIABLE(request);

            int error = mSofs.DeleteSessionObject();

            if (error != RCF::RcfError_Ok)
            {
                RemoteException e(( Error(error) ));
                RCF_THROW(e);
            }
        }


    private:
        SessionObjectFactoryService & mSofs;
    };

    void onServiceAddedProto(SessionObjectFactoryService & sofs, RcfServer & server)
    {
        boost::shared_ptr<SessionObjectFactoryServicePb> sofsPbPtr(
            new SessionObjectFactoryServicePb(sofs));

        server.bind((I_SessionObjectFactoryPb *) NULL, sofsPbPtr);
    }

    void onServiceRemovedProto(SessionObjectFactoryService & sofs, RcfServer & server)
    {
        server.unbind( (I_SessionObjectFactoryPb *) NULL);
    }

#else

    void onServiceAddedProto(SessionObjectFactoryService &, RcfServer &)
    {
    }

    void onServiceRemovedProto(SessionObjectFactoryService &, RcfServer &)
    {
    }

#endif // RCF_USE_PROTOBUF

    void SessionObjectFactoryService::onServiceAdded(RcfServer &server)
    {
        server.bind((I_SessionObjectFactory *) NULL, *this);

        onServiceAddedProto(*this, server);
    }

    void SessionObjectFactoryService::onServiceRemoved(RcfServer &server)
    {
        server.unbind( (I_SessionObjectFactory *) NULL);

        onServiceRemovedProto(*this, server);
    }

} // namespace RCF
