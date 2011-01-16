
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

#include <RCF/PerformanceData.hpp>

#include <RCF/InitDeinit.hpp>
#include <RCF/ObjectPool.hpp>

#include <numeric>

namespace RCF {

    PerformanceData *gpPerformanceData = NULL;

    RCF_ON_INIT( gpPerformanceData = new PerformanceData() );

    RCF_ON_DEINIT( delete gpPerformanceData; gpPerformanceData = NULL; );

    PerformanceData & getPerformanceData()
    {
        return *gpPerformanceData;
    }

    void PerformanceData::collect()
    {
        getObjectPool().enumerateBuffers(mInBufferSizes);
        getObjectPool().enumerateOstrstreams(mOutBufferSizes);

        std::size_t inBufferSize = 
            std::accumulate(mInBufferSizes.begin(), mInBufferSizes.end(), std::size_t(0));

        std::size_t outBufferSize = 
            std::accumulate(mOutBufferSizes.begin(), mOutBufferSizes.end(), std::size_t(0));

        Lock lock(mMutex);
        
        mBufferCount = static_cast<boost::uint32_t>(
            mInBufferSizes.size() + mOutBufferSizes.size());

        mTotalBufferSize = static_cast<boost::uint32_t>(
            inBufferSize + outBufferSize);
    }

} // namespace RCF
