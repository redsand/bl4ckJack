
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

#ifndef INCLUDE_RCF_PROTOCOL_SF_HPP
#define INCLUDE_RCF_PROTOCOL_SF_HPP

#include <RCF/Protocol/Protocol.hpp>

#include <SF/IBinaryStream.hpp>
#include <SF/OBinaryStream.hpp>
#include <SF/string.hpp>

namespace RCF {

    template<>
    class Protocol< boost::mpl::int_<SfBinary> > : public ProtocolImpl1<SF::IBinaryStream, SF::OBinaryStream>
    {
    public:
        static std::string getName()
        {
            return "SF binary serialization protocol";
        }
    };

    inline void enableSfPointerTracking_1(SF::OBinaryStream &obinStream, bool enable)
    {
        enable ?
            obinStream.enableContext():
            obinStream.disableContext();
    }

} // namespace RCF

/*
#include <SF/ITextStream.hpp>
#include <SF/OTextStream.hpp>

namespace RCF {

    template<>
    class Protocol< boost::mpl::int_<SfText> > : public ProtocolImpl1<SF::ITextStream, SF::OTextStream>
    {
    public:
        static std::string getName()
        {
            return "SF text protocol";
        }
    };

    inline void enableSfPointerTracking_2(SF::OTextStream &otextStream, bool enable)
    {
        enable ?
            otextStream.enableContext():
            otextStream.disableContext();
    }

} // namespace RCF
*/

#endif //! INCLUDE_RCF_PROTOCOL_SF_HPP
