
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

#ifndef INCLUDE_RCF_ASIODEADLINETIMER_HPP
#define INCLUDE_RCF_ASIODEADLINETIMER_HPP

#include <RCF/Asio.hpp>

namespace RCF {

    class AsioDeadlineTimer
    {
    public:
        AsioDeadlineTimer(AsioIoService &ioService) :
            mImpl(ioService)
        {}

        boost::asio::deadline_timer mImpl;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_ASIODEADLINETIMER_HPP