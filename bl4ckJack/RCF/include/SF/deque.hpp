
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

#ifndef INCLUDE_SF_DEQUE_HPP
#define INCLUDE_SF_DEQUE_HPP

#include <deque>

#include <SF/SerializeStl.hpp>

namespace SF {

    // std::deque is not guaranteed to use contiguous storage, so even if T 
    // is a fundamental type, we can't do fast memcpy-style serialization, as
    // we do for std::vector.

    // std::deque
    template<typename T, typename A>
    inline void serialize_vc6(SF::Archive &ar, std::deque<T,A> &t, const unsigned int)
    {
        serializeStlContainer<PushBackSemantics, NoReserveSemantics>(ar, t);
    }

} // namespace SF

#endif // ! INCLUDE_SF_DEQUE_HPP
