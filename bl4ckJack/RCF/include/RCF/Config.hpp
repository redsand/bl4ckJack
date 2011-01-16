
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

#ifndef INCLUDE_RCF_CONFIG_HPP
#define INCLUDE_RCF_CONFIG_HPP

#include <boost/config.hpp>

#ifndef RCF_MAX_METHOD_COUNT
#define RCF_MAX_METHOD_COUNT 100
#endif

// For borland, clamp the value to 35 as it complains about macro length otherwise.
#if defined(__BORLANDC__) && RCF_MAX_METHOD_COUNT > 35
#undef RCF_MAX_METHOD_COUNT
#define RCF_MAX_METHOD_COUNT 35
#endif

#if !defined(RCF_USE_SF_SERIALIZATION) && !defined(RCF_USE_BOOST_SERIALIZATION) && !defined(RCF_USE_BOOST_XML_SERIALIZATION)
#define RCF_USE_SF_SERIALIZATION
#endif

#if !defined(RCF_MULTI_THREADED) && !defined(RCF_SINGLE_THREADED)
#define RCF_MULTI_THREADED
#endif

#if defined(RCF_SINGLE_THREADED) && defined(RCF_USE_BOOST_THREADS)
#undef RCF_USE_BOOST_THREADS
#endif

// Detect TR1 availability.
#ifndef RCF_USE_TR1

    // MSVC
    #if defined(_MSC_VER) && _MSC_VER == 1500 && _MSC_FULL_VER >= 150030729
    #define RCF_USE_TR1
    #define RCF_TR1_HEADER(x) <x>
    #endif

    // GCC
    #if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
    #define RCF_USE_TR1
    #define RCF_TR1_HEADER(x) <tr1/x>
    #endif

#endif // RCF_USE_TR1

// Detect hash_map/hash_set availability.
#ifndef RCF_USE_HASH_MAP

    #if (defined(_MSC_VER) && _MSC_VER >= 1310) || (defined(__GNUC__) && __GNUC__ == 3)
        #define RCF_USE_HASH_MAP
        #if defined(_MSC_VER)
            #define RCF_HASH_MAP_HEADER(x) <x>
            #define RCF_HASH_MAP_NS stdext
        #elif defined(__GNUC__)
            #define RCF_HASH_MAP_HEADER(x) <ext/x>
            #define RCF_HASH_MAP_NS __gnu_cxx
        #endif
    #endif

#endif // RCF_USE_HASH_MAP

#endif // ! INCLUDE_RCF_CONFIG_HPP
