
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

#ifndef INCLUDE_RCF_BSDCLIENTTRANSPORT_HPP
#define INCLUDE_RCF_BSDCLIENTTRANSPORT_HPP

#include <RCF/Export.hpp>
#include <RCF/ConnectionOrientedClientTransport.hpp>

namespace RCF {

    class RCF_EXPORT BsdClientTransport :
        public ConnectionOrientedClientTransport
    {
    public:
        BsdClientTransport();
        BsdClientTransport(const BsdClientTransport & rhs);
        BsdClientTransport(int fd);
        ~BsdClientTransport();

        
        int                     releaseFd();
        int                     getFd() const;

        int                        getNativeHandle() const;

    private:

        std::size_t             implRead(
                                    const ByteBuffer &byteBuffer,
                                    std::size_t bytesRequested);

        std::size_t             implReadAsync(
                                    const ByteBuffer &byteBuffer,
                                    std::size_t bytesRequested);

        std::size_t             implWrite(
                                    const std::vector<ByteBuffer> &byteBuffers);

        std::size_t             implWriteAsync(
                                    const std::vector<ByteBuffer> &byteBuffers);
        
        bool                    isConnected();

    protected:
        int                     mFd;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_TCPCLIENTTRANSPORT_HPP
