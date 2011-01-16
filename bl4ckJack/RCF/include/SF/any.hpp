
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

#ifndef INCLUDE_SF_ANY_HPP
#define INCLUDE_SF_ANY_HPP

#include <boost/any.hpp>

#include <RCF/Export.hpp>

namespace SF {

    class Archive;

    RCF_EXPORT void serialize(SF::Archive &ar, boost::any &a);

} // namespace SF

#endif // ! INCLUDE_SF_ANY_HPP
