
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

#ifndef INCLUDE_SF_INTRUSIVE_PTR_HPP
#define INCLUDE_SF_INTRUSIVE_PTR_HPP

//#include <boost/intrusive_ptr.hpp>
namespace boost {
    template<class T>
    class intrusive_ptr;
}

#include <SF/SerializeSmartPtr.hpp>

namespace SF {

    SF_SERIALIZE_REFCOUNTED_SMARTPTR( boost::intrusive_ptr );

} // namespace SF

#endif // ! INCLUDE_SF_INTRUSIVE_PTR_HPP
