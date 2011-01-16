
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

#ifndef INCLUDE_SF_SCOPED_PTR_HPP
#define INCLUDE_SF_SCOPED_PTR_HPP

//#include <boost/scoped_ptr.hpp>
namespace boost {
    template<class T>
    class scoped_ptr;
}

#include <SF/SerializeSmartPtr.hpp>

namespace SF {

    // boost::scoped_ptr
    SF_SERIALIZE_SIMPLE_SMARTPTR( boost::scoped_ptr );

}

#endif // ! INCLUDE_SF_SCOPED_PTR_HPP
