
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

#ifndef INCLUDE_RCF_CURRENTSESSION_HPP
#define INCLUDE_RCF_CURRENTSESSION_HPP

#include <boost/shared_ptr.hpp>

#include <RCF/RcfSession.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/ThreadLocalData.hpp>

namespace RCF {

    class CurrentRcfSessionSentry
    {
    public:
        CurrentRcfSessionSentry(RcfSession & session)
        {
            setCurrentRcfSessionPtr(& session);
        }

        ~CurrentRcfSessionSentry()
        {
            setCurrentRcfSessionPtr();
        }
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_CURRENTSESSION_HPP
