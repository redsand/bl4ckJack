
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

#include <RCF/ThreadLocalCache.hpp>

namespace RCF {

    ObjectCache::VectorByteBufferCache & ObjectCache::getCache(VectorByteBufferCache *)
    {
        return mVectorByteBufferCache;
    }

    ObjectCache::VectorIntCache & ObjectCache::getCache(VectorIntCache *)
    {
        return mVectorIntCache;
    }

    ObjectCache::VectorWsabufCache & ObjectCache::getCache(VectorWsabufCache *)
    {
        return mVectorWsabufCache;
    }

    ObjectCache::VectorFilterCache & ObjectCache::getCache(VectorFilterCache *)
    {
        return mVectorFilterCache;
    }

    ObjectCache::VectorRcfSessionCallbackCache & ObjectCache::getCache(VectorRcfSessionCallbackCache *)
    {
        return mVectorRcfSessionCallbackCache;
    }

    void ObjectCache::clear()
    {
        mVectorByteBufferCache.clear();
        mVectorIntCache.clear();
        mVectorWsabufCache.clear();
        mVectorFilterCache.clear();
        mVectorRcfSessionCallbackCache.clear();
    }

} // namespace RCF
