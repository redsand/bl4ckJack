
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

#ifndef INCLUDE_SF_HASH_MAP_HPP
#define INCLUDE_SF_HASH_MAP_HPP

#include <RCF/Config.hpp>
#ifdef RCF_USE_HASH_MAP

#include RCF_HASH_MAP_HEADER(hash_map)

#include <SF/SerializeStl.hpp>

namespace SF {

    // hash_map
    template<typename Key, typename Value, typename Hash, typename Alloc>
    inline void serialize_vc6(Archive &ar, RCF_HASH_MAP_NS::hash_map<Key, Value, Hash, Alloc> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

    // hash_multimap
    template<typename Key, typename Value, typename Hash, typename Alloc>
    inline void serialize_vc6(Archive &ar, RCF_HASH_MAP_NS::hash_multimap<Key, Value, Hash, Alloc> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

}

#endif // RCF_USE_HASH_MAP

#endif // ! INCLUDE_SF_HASH_MAP_HPP
