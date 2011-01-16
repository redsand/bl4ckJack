
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

#ifndef INCLUDE_SF_ADLWORKAROUND_HPP
#define INCLUDE_SF_ADLWORKAROUND_HPP

#include <SF/Archive.hpp>

// Argument dependent lookup doesn't work on vc6, so here's a workaround.

#if defined(_MSC_VER) && _MSC_VER < 1310

#define SF_ADL_WORKAROUND(Ns, T)                    \
namespace SF {                                      \
    inline void serialize(Archive &ar, Ns::T &t)    \
    {                                               \
        Ns::serialize(ar, t);                       \
    }                                               \
}

#else

#define SF_ADL_WORKAROUND(Ns, T)

#endif

#endif // ! INCLUDE_SF_ADLWORKAROUND_HPP
