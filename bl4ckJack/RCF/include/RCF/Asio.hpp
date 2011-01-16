
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

#ifndef INCLUDE_RCF_ASIO_HPP
#define INCLUDE_RCF_ASIO_HPP

// Some issues with asio headers.
#if defined(__MACH__) && defined(__APPLE__)
#include <limits.h>
#ifndef IOV_MAX
#define IOV_MAX 1024
#endif
#endif

#include <boost/asio.hpp>

#endif // ! INCLUDE_RCF_ASIO_HPP
