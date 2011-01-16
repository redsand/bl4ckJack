
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

#ifndef INCLUDE_RCF_BYTEORDERING_HPP
#define INCLUDE_RCF_BYTEORDERING_HPP

#include <RCF/Export.hpp>

namespace RCF {

    RCF_EXPORT void machineToNetworkOrder(void *buffer, int width, int count);
    RCF_EXPORT void networkToMachineOrder(void *buffer, int width, int count);
    RCF_EXPORT bool machineOrderEqualsNetworkOrder();
    RCF_EXPORT bool isPlatformLittleEndian();

} // namespace RCF

#endif // ! INCLUDE_RCF_BYTEORDERING_HPP
