
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

#include <RCF/ObjectPool.hpp>

#include <RCF/ByteBuffer.hpp>
#include <RCF/Exception.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/Tools.hpp>

#include <boost/version.hpp>

namespace RCF {

    CbAllocatorBase::CbAllocatorBase(ObjectPool & objectPool) : 
        mObjectPool(objectPool)
    {}

    CbAllocatorBase::CbAllocatorBase(const CbAllocatorBase & rhs) : 
        mObjectPool(rhs.mObjectPool)
    {}

    void * CbAllocatorBase::allocate() 
    { 
        return mObjectPool.getPcb();
    }

    void CbAllocatorBase::deallocate(void * pcb) 
    { 
        mObjectPool.putPcb(pcb);
    }

    ObjectPool::ObjectPool() : 
        mObjPoolMutex(WriterPriority),
        mBufferCountLimit(10) 
    {
    }

#if defined(_MSC_VER) && _MSC_VER <= 1200
#define for if (0) {} else for
#endif

    ObjectPool::~ObjectPool()
    {
        for (std::size_t i=0; i<mVecPool.size(); ++i)
        {
            delete mVecPool[i];
        }

        for (std::size_t i=0; i<mOsPool.size(); ++i)
        {
            delete mOsPool[i];
        }

        for (std::size_t i=0; i<mCbPool.size(); ++i)
        {
            delete [] (char *) mCbPool[i];
        }

        ObjPool::iterator iter;
        for (iter = mObjPool.begin(); iter != mObjPool.end(); ++iter)
        {
            ObjList & objList = *(iter->second);
            for (std::size_t i=0; i<objList.mVec.size(); ++i)
            {
                objList.mOps->kill(objList.mVec[i]);
            }
        }
    }

#if defined(_MSC_VER) && _MSC_VER <= 1200
#undef for
#endif

    void ObjectPool::setBufferCountLimit(std::size_t bufferCountLimit)
    {
        mBufferCountLimit = bufferCountLimit;
    }

    std::size_t ObjectPool::getBufferCountLimit()
    {
        return mBufferCountLimit;
    }

    void * ObjectPool::getPcb()
    {
        void * pcb = NULL;

        Lock lock(mCbPoolMutex);
        if (mCbPool.empty())
        {
            pcb = new char[CbSize];
        }
        else
        {
            pcb = mCbPool.back();
            mCbPool.pop_back();
        }

        return pcb;
    }

    void ObjectPool::putPcb(void * pcb)
    {
        Lock lock(mCbPoolMutex);
        mCbPool.push_back(pcb);
    }

    void ObjectPool::get(VecPtr & vecPtr)
    {
        std::vector<char> * pt = NULL;
        getPtr(pt, vecPtr, mVecPool, mVecPoolMutex, &ObjectPool::putVec);
    }

    void ObjectPool::get(OstrStreamPtr & ostrStreamPtr)
    {
        std::ostrstream * pt = NULL;
        getPtr(pt, ostrStreamPtr, mOsPool, mOsPoolMutex, &ObjectPool::putOstrStream);
    }

    void ObjectPool::get(ReallocBufferPtr & bufferPtr)
    {
        ReallocBuffer * pt = NULL;
        getPtr(pt, bufferPtr, mRbPool, mRbPoolMutex, &ObjectPool::putReallocBuffer);
    }

    void ObjectPool::putVec(std::vector<char> * pVec)
    {
        std::auto_ptr<std::vector<char> > vecPtr(pVec);

        pVec->resize(0);
        Lock lock(mVecPoolMutex);
        if (mVecPool.size() < mBufferCountLimit)
        {
            mVecPool.push_back(vecPtr.release());
        }
    }

    void ObjectPool::putOstrStream(std::ostrstream * pOs)
    {
        std::auto_ptr<std::ostrstream> osPtr(pOs);

        pOs->clear(); // freezing may have set error state
        pOs->rdbuf()->freeze(false);
        pOs->rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);

        Lock lock(mOsPoolMutex);
        if (mOsPool.size() < mBufferCountLimit)
        {
            mOsPool.push_back(osPtr.release());
        }
    }

    void ObjectPool::putReallocBuffer(ReallocBuffer * pRb)
    {
        std::auto_ptr<ReallocBuffer> rbPtr(pRb);

        pRb->resize(0);
        Lock lock(mRbPoolMutex);
        if (mRbPool.size() < mBufferCountLimit)
        {
            mRbPool.push_back(rbPtr.release());
        }
    }

    typedef ObjectPool::VecPtr VecPtr;

    void ObjectPool::enumerateBuffers(std::vector<std::size_t> & bufferSizes)
    {
        bufferSizes.resize(0);

        Lock lock(mVecPoolMutex);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4267)
#endif

        for (std::size_t i=0; i<mVecPool.size(); ++i)
        {
            bufferSizes.push_back( mVecPool[i]->capacity() );
        }

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    }

    typedef ObjectPool::OstrStreamPtr OstrStreamPtr;
    
    std::size_t getOstrstreamSize(std::ostrstream * pOs)
    {
        // What's the current position.
        std::size_t currentPos = static_cast<std::size_t>(pOs->pcount());

        // What's the end position.
        pOs->rdbuf()->pubseekoff(0, std::ios::end, std::ios::out);
        std::size_t endPos = static_cast<std::size_t>(pOs->pcount());

        // Set it back to the current position.
        pOs->rdbuf()->pubseekoff(
            static_cast<std::ostrstream::off_type>(currentPos), 
            std::ios::beg, 
            std::ios::out);

        // Return the end position.
        return endPos;
    }

    void ObjectPool::enumerateOstrstreams(std::vector<std::size_t> & bufferSizes)
    {
        bufferSizes.resize(0);

        Lock lock(mOsPoolMutex);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4267)
#endif

        for (std::size_t i=0; i<mOsPool.size(); ++i)
        {
            bufferSizes.push_back( getOstrstreamSize(mOsPool[i]) );
        }

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    }

    void ObjectPool::enumerateReallocBuffers(std::vector<std::size_t> & bufferSizes)
    {
        bufferSizes.resize(0);

        Lock lock(mRbPoolMutex);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4267)
#endif

        for (std::size_t i=0; i<mRbPool.size(); ++i)
        {
            bufferSizes.push_back( mRbPool[i]->capacity() );
        }

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    }

    ObjectPool * gpObjectPool;

    ObjectPool & getObjectPool()
    {
        return *gpObjectPool;
    }

    RCF_ON_INIT_DEINIT( 
        gpObjectPool = new ObjectPool(); , 
        delete gpObjectPool; gpObjectPool = NULL; )

} // namespace RCF
