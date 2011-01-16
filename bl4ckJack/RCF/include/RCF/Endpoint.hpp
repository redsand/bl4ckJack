
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

#ifndef INCLUDE_RCF_ENDPOINT_HPP
#define INCLUDE_RCF_ENDPOINT_HPP

#include <memory>
#include <string>

#include <boost/shared_ptr.hpp>

#include <RCF/Exception.hpp>
#include <RCF/SerializationProtocol.hpp>

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/SfNew.hpp>
#endif

namespace RCF {

    class I_ServerTransport;
    class I_ClientTransport;

    class I_Endpoint;
    typedef boost::shared_ptr<I_Endpoint> EndpointPtr;

    class I_Endpoint
    {
    public:
        virtual ~I_Endpoint() {}
        virtual std::auto_ptr<I_ServerTransport>    createServerTransport() const = 0;
        virtual std::auto_ptr<I_ClientTransport>    createClientTransport() const = 0;
        virtual EndpointPtr                         clone() const = 0;
        virtual std::string                         asString() const = 0;
        void                                        serialize(SF::Archive &) {}
    };

} // namespace RCF

RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(RCF::I_Endpoint)

#ifdef RCF_USE_SF_SERIALIZATION
namespace SF {
    SF_NO_CTOR(RCF::I_Endpoint)
}
#endif

#include <boost/version.hpp>

#if defined(RCF_USE_BOOST_SERIALIZATION) && BOOST_VERSION < 103600
#include <boost/serialization/is_abstract.hpp>
#include <boost/serialization/shared_ptr.hpp>
BOOST_IS_ABSTRACT(RCF::I_Endpoint)
BOOST_SERIALIZATION_SHARED_PTR(RCF::I_Endpoint)
#endif

#if defined(RCF_USE_BOOST_SERIALIZATION) && BOOST_VERSION >= 103600
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/shared_ptr.hpp>
BOOST_SERIALIZATION_ASSUME_ABSTRACT(RCF::I_Endpoint)
BOOST_SERIALIZATION_SHARED_PTR(RCF::I_Endpoint)
#endif

// The following is a starting point for registering polymorphic serialization 
// for I_Endpoint-derived classes, with Boost.Serialization ...

/*
#ifdef RCF_USE_BOOST_SERIALIZATION
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
BOOST_CLASS_EXPORT_GUID(RCF::TcpEndpoint, "RCF::TcpEndpoint")
BOOST_SERIALIZATION_SHARED_PTR(RCF::TcpEndpoint)
#endif
*/

#endif // ! INCLUDE_RCF_ENDPOINT_HPP
