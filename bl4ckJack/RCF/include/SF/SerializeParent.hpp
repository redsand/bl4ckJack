
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

#ifndef INCLUDE_SF_SERIALIZEPARENT_HPP
#define INCLUDE_SF_SERIALIZEPARENT_HPP

#include <SF/Archive.hpp>
#include <SF/SerializePolymorphic.hpp>

namespace SF {

    template<typename Base, typename Derived>
    void serializeParent(Base *, Archive &ar, Derived &derived)
    {
        SF::SerializerPolymorphic<Base,Derived>::instantiate();
        ar & SF::Archive::PARENT & static_cast<Base &>(derived);
    }

#if !defined(_MSC_VER) || _MSC_VER >= 1310

    template<typename Base, typename Derived>
    void serializeParent(Archive &ar, Derived &derived)
    {
        serializeParent( (Base *) 0, ar, derived);
    }

#endif

} // namespace SF

// vc6 apparently doesn't do ADL, so we add a forwarding function in the global 
// namespace.

#if defined(_MSC_VER) && _MSC_VER == 1200

template<typename Base, typename Derived>
void serializeParent(Base *, SF::Archive &ar, Derived &derived)
{
    SF::serializeParent( (Base *) 0, ar, derived);
}

#endif

#endif // ! INCLUDE_SF_SERIALIZEPARENT_HPP
