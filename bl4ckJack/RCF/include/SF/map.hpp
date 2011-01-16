
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

#ifndef INCLUDE_SF_MAP_HPP
#define INCLUDE_SF_MAP_HPP

#include <map>

#include <SF/SerializeStl.hpp>
#include <SF/utility.hpp>

namespace SF {

    // std::map
    template<typename K, typename T1, typename T2, typename A>
    inline void serialize_vc6(Archive &ar, std::map<K,T1,T2,A> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

    // std::multimap
    template<typename K, typename T1, typename T2, typename A>
    inline void serialize_vc6(Archive &ar, std::multimap<K,T1,T2,A> &t, const unsigned int)
    {
        serializeStlContainer<InsertSemantics, NoReserveSemantics>(ar, t);
    }

}

#endif // ! INCLUDE_SF_MAP_HPP
