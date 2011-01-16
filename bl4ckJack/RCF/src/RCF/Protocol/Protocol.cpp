
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

#include <RCF/Protocol/Protocol.hpp>
#include <RCF/Config.hpp>
#include <RCF/SerializationProtocol.hpp>

namespace RCF {

#ifdef RCF_USE_SF_SERIALIZATION
    const SerializationProtocol DefaultSerializationProtocol = Sp_SfBinary;
#else
    const SerializationProtocol DefaultSerializationProtocol = Sp_BsBinary;
#endif

#ifdef RCF_USE_PROTOBUF
    const MarshalingProtocol DefaultMarshalingProtocol = Mp_Rcf;
#else
    const MarshalingProtocol DefaultMarshalingProtocol = Mp_Rcf;
#endif

} // namespace RCF
