
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

#ifndef INCLUDE_RCF_MINMAX_HPP
#define INCLUDE_RCF_MINMAX_HPP

/*
// Because of macros in Windows platform headers, it's rather difficult to use 
// std::min/max. So we define our own instead.

namespace RCF {

    template<typename T>
    T rcfMin(T t1, T t2)
    {
        return (t1 <= t2) ? t1 : t2;
    }
    template<typename T>
    T rcfMax(T t1, T t2)
    {
        return (t1 <= t2) ? t2 : t1;
    }

#define RCF_MIN RCF::rcfMin
#define RCF_MAX RCF::rcfMax

} // namespace RCF
*/

#include <algorithm>

#define RCF_MIN (std::min)
#define RCF_MAX (std::max)

#endif // ! INCLUDE_RCF_MINMAX_HPP
