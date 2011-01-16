
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

#ifndef INCLUDE_SF_ONATIVEBINARYSTREAM_HPP
#define INCLUDE_SF_ONATIVEBINARYSTREAM_HPP

#include <SF/Stream.hpp>

namespace SF {

    class ONativeBinaryStream : public OStream
    {
    public:
        ONativeBinaryStream(std::ostream &os) : OStream(os)
        {}

        I_Encoding &getEncoding()
        {
            return mEncoding;
        }

    private:
        EncodingBinaryNative mEncoding;
    };


}

#endif // ! INCLUDE_SF_ONATIVEBINARYSTREAM_HPP
