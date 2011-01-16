
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

#ifndef INCLUDE_RCF_VERSION_HPP
#define INCLUDE_RCF_VERSION_HPP

#include <RCF/Export.hpp>

#include <boost/cstdint.hpp>

// RCF 0.9c - 903
// RCF 0.9d - 9040
// RCF 1.0 - 10000
// RCF 1.1 - 11000
// RCF 1.2 - 12000
// RCF 1.3 - 13000
#define RCF_VERSION 13000

namespace RCF {

    // Runtime versioning.

    // legacy       - version number 1

    // 2007-04-26   - version number 2
    // Released in 0.9c

    // 2008-03-29   - version number 3
    //      - Using I_SessionObjectFactory instead of I_ObjectFactoryService for session object creation and deletion.
    // Released in 0.9d

    // 2008-09-06   - version number 4
    //      - ByteBuffer compatible with std::vector etc.
    // Released in 1.0

    // 2008-12-06   - version number 5
    //      - Pingback field in MethodInvocationRequest
    // Released in 1.1

    // 2010-01-21   - version number 6
    //      - Archive version field in MethodInvocationRequest
    //      - Embedded version stamps in SF archives.
    //      - SF: Serialization of error arguments in RemoteException.
    // Released in 1.2

    // 2010-03-20   - version number 7
    //      - User data fields in request and response headers
    // Interim release (rev 1414).

    // 2010-03-30   - version number 8
    //      - Ping intervals between publishers and subscribers.
    //      - Byte reordering for fast vector serialization.
    //      - BSer: Serialization of error arguments in RemoteException.
    //      - Non-polymorphic marshaling of reference parameters
    //      - UTF-8 serialization of wstring (native as an option). Changes to request header.
    //      - BSer: remote exceptions serialized through raw pointer rather than auto_ptr.
    //      - Error response messages contain two custom args, rather than one.
    // Released in 1.3

    // Inherent runtime version - can't be changed.
    RCF_EXPORT boost::uint32_t  getInherentRuntimeVersion();

    // Default runtime version.
    RCF_EXPORT boost::uint32_t  getDefaultRuntimeVersion();
    RCF_EXPORT void             setDefaultRuntimeVersion(boost::uint32_t version);

    // Default archive version.
    RCF_EXPORT boost::uint32_t  getDefaultArchiveVersion();
    RCF_EXPORT void             setDefaultArchiveVersion(boost::uint32_t version);

} // namespace RCF

#endif // ! INCLUDE_RCF_VERSION_HPP
