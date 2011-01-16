
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

#ifndef INCLUDE_RCF_SESSIONOBJECTFACTORYSERVICE_HPP
#define INCLUDE_RCF_SESSIONOBJECTFACTORYSERVICE_HPP

#include <RCF/ObjectFactoryService.hpp>

namespace RCF {

    class RCF_EXPORT SessionObjectFactoryService :
        public I_Service,
        public StubFactoryRegistry,
        boost::noncopyable
    {
    public:
        boost::int32_t CreateSessionObject(const std::string &objectName);
        boost::int32_t DeleteSessionObject();

    private:
        void onServiceAdded(RcfServer &server);
        void onServiceRemoved(RcfServer &server);
    };

    typedef boost::shared_ptr<SessionObjectFactoryService> 
        SessionObjectFactoryServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_SESSIONOBJECTFACTORYSERVICE_HPP
