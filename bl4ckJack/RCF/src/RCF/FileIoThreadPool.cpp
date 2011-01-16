
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

#include <RCF/FileIoThreadPool.hpp>

#include <RCF/InitDeinit.hpp>
#include <RCF/ThreadLocalData.hpp>

namespace RCF {

    FileIoThreadPool::FileIoThreadPool() : 
        mSerializeFileIo(false),
        mThreadPool(1, 10, "RCF Async File IO", 30*1000, false),
        mStopFlag(false) 
        
    {
        mThreadPool.setTask( boost::bind(
            &FileIoThreadPool::ioTask,
            this));

        mThreadPool.setStopFunctor( boost::bind(
            &FileIoThreadPool::stopIoTask,
            this));
    }

    FileIoThreadPool::~FileIoThreadPool()
    {
        RCF_DTOR_BEGIN
            mThreadPool.stop();
        RCF_DTOR_END
    }

    void FileIoThreadPool::setSerializeFileIo(bool serializeFileIo)
    {
        mSerializeFileIo = serializeFileIo;
    }

    void FileIoThreadPool::stop()
    {
        mThreadPool.stop();
    }

    void FileIoThreadPool::registerOp(FileIoRequestPtr opPtr)
    {
        RCF::Lock lock(mOpsMutex);

        // Lazy start of the thread pool.
        if (!mThreadPool.isStarted())
        {
            mStopFlag = false;
            mThreadPool.start(mStopFlag);
        }

        if (    std::find(mOpsQueued.begin(), mOpsQueued.end(), opPtr) 
            !=  mOpsQueued.end())
        {
            RCF_ASSERT(0);
        }
        else if (       std::find(mOpsInProgress.begin(), mOpsInProgress.end(), opPtr) 
                    !=  mOpsInProgress.end())
        {
            RCF_ASSERT(0);
        }
        else
        {
            mOpsQueued.push_back(opPtr);
            mOpsCondition.notify_all();
        }
    }

    void FileIoThreadPool::unregisterOp(FileIoRequestPtr opPtr)
    {
        RCF::Lock lock(mOpsMutex);
        RCF::eraseRemove(mOpsQueued, opPtr);
        RCF::eraseRemove(mOpsInProgress, opPtr);
    }

    bool FileIoThreadPool::ioTask()
    {
        FileIoRequestPtr opPtr;

        {
            RCF::Lock lock(mOpsMutex);
            while (mOpsQueued.empty() && !mStopFlag)
            {
                mOpsCondition.timed_wait(lock, 1000);
            }
            if (mOpsQueued.empty() || mStopFlag)
            {
                return false;
            }
            RCF_ASSERT_GT(mOpsQueued.size() , 0);
            mOpsInProgress.push_back( mOpsQueued.front() );
            mOpsQueued.pop_front();
            opPtr = mOpsInProgress.back();
        }

        RCF::ThreadInfoPtr threadInfoPtr = RCF::getThreadInfoPtr();
        if (threadInfoPtr)
        {
            threadInfoPtr->notifyBusy();
        }

        // This is the part that blocks.
        opPtr->doTransfer();

        // Unregister op.
        unregisterOp(opPtr);

        // Notify completion.
        {
            RCF::Lock lock(mCompletionMutex);
            opPtr->mCompleted = true;
            mCompletionCondition.notify_all();
        }

        return false;
    }

    void FileIoThreadPool::stopIoTask()
    {
        RCF::Lock lock(mOpsMutex);
        mStopFlag = true;
        mOpsCondition.notify_all();
    }

    FileIoRequest::FileIoRequest() :
        mFts( getFileIoThreadPool() ),
        mBytesTransferred(0),
        mInitiated(false),
        mCompleted(true)
    {
        RCF_LOG_4() << "FileIoRequest::FileIoRequest";
    }

    FileIoRequest::~FileIoRequest()
    {
        RCF_LOG_4() << "FileIoRequest::~FileIoRequest";
    }

    bool FileIoRequest::initiated()
    {
        RCF_LOG_4() << "FileIoRequest::initiated";

        RCF::Lock lock(mFts.mCompletionMutex);
        return mInitiated;
    }

    bool FileIoRequest::completed()
    {
        RCF_LOG_4() << "FileIoRequest::completed";

        RCF::Lock lock(mFts.mCompletionMutex);
        return mCompleted;
    }

    void FileIoRequest::complete()
    {
        RCF_LOG_4() << "FileIoRequest::complete()";

        RCF::Lock lock(mFts.mCompletionMutex);
        while (!mCompleted)
        {
            mFts.mCompletionCondition.wait(lock);
        }
        mInitiated = false;
    }

    void FileIoRequest::read(boost::shared_ptr<std::ifstream> finPtr, RCF::ByteBuffer buffer)
    {
        RCF_LOG_4()(finPtr.get())((void*)buffer.getPtr())(buffer.getLength()) << "FileIoRequest::read()";

        mFinPtr = finPtr;
        mFoutPtr.reset();
        mBuffer = buffer;
        mBytesTransferred = 0;
        mInitiated = true;
        mCompleted = false;

        mFts.registerOp( shared_from_this() );

        // For debugging purposes, we can wait in this function until the file I/O is completed.
        if (mFts.mSerializeFileIo)
        {
            RCF::Lock lock(mFts.mCompletionMutex);
            while (!mCompleted)
            {
                mFts.mCompletionCondition.wait(lock);
            }
        }
    }

    void FileIoRequest::write(boost::shared_ptr<std::ofstream> foutPtr, RCF::ByteBuffer buffer)
    {
        RCF_LOG_4()(foutPtr.get())((void*)buffer.getPtr())(buffer.getLength()) << "FileIoRequest::write()";

        mFinPtr.reset();
        mFoutPtr = foutPtr;
        mBuffer = buffer;
        mBytesTransferred = 0;
        mInitiated = true;
        mCompleted = false;

        mFts.registerOp( shared_from_this() );

        // For debugging purposes, we can wait in this function until the file I/O is completed.
        if (mFts.mSerializeFileIo)
        {
            RCF::Lock lock(mFts.mCompletionMutex);
            while (!mCompleted)
            {
                mFts.mCompletionCondition.wait(lock);
            }
        }
    }

    void FileIoRequest::doTransfer()
    {
        if (mFinPtr)
        {
            RCF_LOG_4() << "FileIoRequest::doTransfer() - initiate read.";

            char * szBuffer = mBuffer.getPtr();
            std::size_t szBufferLen = mBuffer.getLength();
            mFinPtr->read(szBuffer, szBufferLen);
            mBytesTransferred = mFinPtr->gcount();
            mFinPtr.reset();

            RCF_LOG_4()(mBytesTransferred) << "FileIoRequest::doTransfer() - read complete.";
        }
        else if (mFoutPtr)
        {
            RCF_LOG_4() << "FileIoRequest::doTransfer() - initiate write.";

            char * szBuffer = mBuffer.getPtr();
            std::size_t szBufferLen = mBuffer.getLength();
            
            boost::uint64_t pos0 = mFoutPtr->tellp();
            mFoutPtr->write(szBuffer, szBufferLen);
            boost::uint64_t pos1 = mFoutPtr->tellp();

            RCF_ASSERT_GTEQ(pos1 , pos0);
            mBytesTransferred = pos1 - pos0;
            RCF_ASSERT_EQ(mBytesTransferred , szBufferLen);
            mFoutPtr.reset();

            RCF_LOG_4()(mBytesTransferred) << "FileIoRequest::doTransfer() - write complete.";
        }
        else
        {
            RCF_ASSERT(0);
            mBytesTransferred = 0;
        }
    }

    boost::uint64_t FileIoRequest::getBytesTransferred()
    {
        RCF_LOG_4()(mBytesTransferred) << "FileIoRequest::getBytesTransferred()";

        return mBytesTransferred;
    }

    static FileIoThreadPool * gpFileIoThreadPool = NULL;

    FileIoThreadPool & getFileIoThreadPool()
    {
        FileIoThreadPool * pFileIoThreadPool = gpFileIoThreadPool;
        return *pFileIoThreadPool;
    }

    RCF_ON_INIT_DEINIT_NAMED(
        gpFileIoThreadPool = new FileIoThreadPool(), 
        delete gpFileIoThreadPool; gpFileIoThreadPool = NULL, 
        FileIoThreadPoolInit)

} // namespace RCF
