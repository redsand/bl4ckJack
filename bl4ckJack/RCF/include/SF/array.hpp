
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

#ifndef INCLUDE_SF_ARRAY_HPP
#define INCLUDE_SF_ARRAY_HPP

#include <cstddef>

#include <RCF/Exception.hpp>
#include <SF/Archive.hpp>

namespace boost {

    template<class T, std::size_t N>
    class array;

}

namespace std { namespace tr1 {

    template<class T, std::size_t N>
    class array;

} }

namespace SF {

    class Archive;

    template<typename T, std::size_t N>
    void serialize_vc6(SF::Archive & ar, boost::array<T, N> & a, const unsigned int)
    {
        if (ar.isRead())
        {
            unsigned int count = 0;
            ar & count;

            RCF_VERIFY(
                count == a.size(), 
                RCF::Exception(RCF::_RcfError_RcfError_ArraySizeMismatch(a.size(), count)));

            for (std::size_t i=0; i<a.size(); ++i)
            {
                ar & a[i];
            }
        }
        else if (ar.isWrite())
        {
            unsigned int count = a.size();
            ar & count;

            for (std::size_t i=0; i<a.size(); ++i)
            {
                ar & a[i];
            }
        }
    }

    template<typename T, std::size_t N>
    void serialize_vc6(SF::Archive & ar, std::tr1::array<T, N> & a, const unsigned int)
    {
        if (ar.isRead())
        {
            unsigned int count = 0;
            ar & count;
            RCF_VERIFY(count == a.size(), RCF::Exception());

            for (std::size_t i=0; i<a.size(); ++i)
            {
                ar & a[i];
            }
        }
        else if (ar.isWrite())
        {
            unsigned int count = a.size();
            ar & count;

            for (std::size_t i=0; i<a.size(); ++i)
            {
                ar & a[i];
            }
        }
    }

} // namespace SF

#endif // ! INCLUDE_SF_ARRAY_HPP
