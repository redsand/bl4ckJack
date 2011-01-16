
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

#ifndef INCLUDE_RCF_SESSIONTIMEOUTSERVICE_HPP
#define INCLUDE_RCF_SESSIONTIMEOUTSERVICE_HPP

#include <set>

#include <RCF/Export.hpp>
#include <RCF/Service.hpp>
#include <RCF/Timer.hpp>

namespace RCF {

    class RcfSession;
    typedef boost::shared_ptr<RcfSession> RcfSessionPtr;
    typedef boost::weak_ptr<RcfSession> RcfSessionWeakPtr;

    class RCF_EXPORT SessionTimeoutService : public I_Service
    {
    public:
        SessionTimeoutService(
            boost::uint32_t sessionTimeoutMs,
            boost::uint32_t reapingIntervalMs = 30*1000);

    private:

        void onServiceAdded(RcfServer &server);
        void onServiceRemoved(RcfServer &server);

        void stop();

        bool cycle(
            int timeoutMs,
            const volatile bool &stopFlag);

    private:

        std::vector<RcfSessionWeakPtr>  mSessionsTemp;

        boost::uint32_t                 mSessionTimeoutMs;
        RCF::Timer                      mLastRunTimer;
        boost::uint32_t                 mReapingIntervalMs;

        bool                            mStopFlag;

        RcfServer *                     mpRcfServer;

    };

    typedef boost::shared_ptr<SessionTimeoutService> SessionTimeoutServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_SESSIONTIMEOUTSERVICE_HPP
