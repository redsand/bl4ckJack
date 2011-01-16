
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

#ifndef INCLUDE_SF_UNORDERED_SET_HPP
#define INCLUDE_SF_UNORDERED_SET_HPP

#include <RCF/Config.hpp>
#ifdef RCF_USE_TR1

#include RCF_TR1_HEADER(unordered_set)

#include <SF/SerializeStl.hpp>

namespace SF {

    // std::tr1::unordered_set
    template<typename Key, typename Hash, typename Pred, typename Alloc>
    inline void serialize_vc6(Archive &ar, std::tr1::unordered_set<Key,Hash,Pred,Alloc> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

    // std::tr1::unordered_multiset
    template<typename Key, typename Hash, typename Pred, typename Alloc>
    inline void serialize_vc6(Archive &ar, std::tr1::unordered_multiset<Key,Hash,Pred,Alloc> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

}

#endif // RCF_USE_TR1

#endif // ! INCLUDE_SF_UNORDERED_SET_HPP
