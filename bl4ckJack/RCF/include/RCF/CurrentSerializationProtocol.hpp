
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

#ifndef INCLUDE_RCF_CURRENTSERIALIZATIONPROTOCOL_HPP
#define INCLUDE_RCF_CURRENTSERIALIZATIONPROTOCOL_HPP

#include <cstddef>

#include <RCF/Export.hpp>

namespace RCF {

    class SerializationProtocolIn;
    class SerializationProtocolOut;

    RCF_EXPORT SerializationProtocolIn *getCurrentSerializationProtocolIn();
    RCF_EXPORT SerializationProtocolOut *getCurrentSerializationProtocolOut();

} // namespace RCF

#endif // ! INCLUDE_RCF_CURRENTSERIALIZATIONPROTOCOL_HPP
