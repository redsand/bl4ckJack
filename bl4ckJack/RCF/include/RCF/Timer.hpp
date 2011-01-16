
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

#ifndef INCLUDE_RCF_TIMER_HPP
#define INCLUDE_RCF_TIMER_HPP

#include <RCF/Export.hpp>

#include <RCF/util/Platform/OS/GetCurrentTime.hpp>

namespace RCF {

    class RCF_EXPORT Timer
    {
    public:

        Timer();
        Timer(boost::uint32_t startTimeMs);

        bool                elapsed(boost::uint32_t durationMs);
        void                restart();
        void                restart(boost::uint32_t startTimeMs);
        boost::uint32_t     getStartTimeMs();
        boost::uint32_t     getDurationMs();

    private:
        boost::uint32_t mStartTimeMs;
    };

}

#endif // ! INCLUDE_RCF_TIMER_HPP
