
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

#include <RCF/IpServerTransport.hpp>

namespace RCF {

    I_IpServerTransport::I_IpServerTransport() :
        mReadWriteMutex(WriterPriority)
    {}

    I_IpServerTransport::~I_IpServerTransport() 
    {}

    void I_IpServerTransport::setAllowIps(
        const std::vector<IpRule> &allowedIps)
    {
        WriteLock writeLock(mReadWriteMutex);
        mAllowedIps = allowedIps;

        // Resolve all.
        for (std::size_t i=0; i<mAllowedIps.size(); ++i)
        {
            mAllowedIps[i].first.resolve();
        }
    }

    void I_IpServerTransport::setDenyIps(
        const std::vector<IpRule> &disallowedIps)
    {
        WriteLock writeLock(mReadWriteMutex);
        mDisallowedIps = disallowedIps;

        // Resolve all.
        for (std::size_t i=0; i<mDisallowedIps.size(); ++i)
        {
            mDisallowedIps[i].first.resolve();
        }
    }

    std::vector<IpRule> I_IpServerTransport::getAllowIps() const
    {
        ReadLock readLock(mReadWriteMutex);
        return mAllowedIps;
    }

    std::vector<IpRule> I_IpServerTransport::getDenyIps() const
    {
        ReadLock readLock(mReadWriteMutex);
        return mDisallowedIps;
    }

    bool I_IpServerTransport::isIpAllowed(const IpAddress &ip) const
    {
        ReadLock readLock(mReadWriteMutex);

        if (!mAllowedIps.empty())
        {
            for (std::size_t i=0; i<mAllowedIps.size(); ++i)
            {
                if (ip.matches(mAllowedIps[i].first, mAllowedIps[i].second))
                {
                    return true;
                }
            }
            return false;
        }

        if (!mDisallowedIps.empty())
        {
            for (std::size_t i=0; i<mDisallowedIps.size(); ++i)
            {
                if (ip.matches(mDisallowedIps[i].first, mDisallowedIps[i].second))
                {
                    return false;
                }
            }
            return true;
        }

        return true;
    }    

} // namespace RCF
