
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

#include <RCF/FileTransferService.hpp>

#include <boost/function.hpp>

#include <cstdio>
#include <iomanip>
#include <boost/limits.hpp>

#include <sys/stat.h>

#include <RCF/Exception.hpp>
#include <RCF/ObjectFactoryService.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/util/Platform/OS/Sleep.hpp>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/operations.hpp>

namespace boost {
    namespace filesystem {

        void serialize(SF::Archive &ar, path &p)
        {
            if (ar.isWrite())
            {
                std::string s = p.string();
                ar & s;
            }
            else
            {
                std::string s;
                ar & s;
                p = path(s);
            }
        }

    }
}

namespace RCF {

    namespace fs = boost::filesystem;

    FileUploadInfo::~FileUploadInfo()
    {
        mFileStream->close();

        // Best effort only.
        try
        {
            if (!mCompleted && mUploadId == 0)
            {
                fs::remove_all(mUploadPath);
            }
        }
        catch(const std::exception & e)
        {
            std::string error = e.what();
        }
        catch(...)
        {

        }
    }

    FileDownloadInfo::~FileDownloadInfo()
    {
    }

    FileTransferService::FileTransferService(
        const fs::path & uploadDirectory) :
            mUploadDirectory(uploadDirectory),
            mTransferWindowS(5)
    {
    }

    namespace fs = boost::filesystem;

    fs::path makeTempDir(const fs::path & basePath, const std::string & prefix)
    {
        std::size_t tries = 0;
        while (tries++ < 10)
        {
            std::ostringstream os;
            os 
                << prefix
                << std::setw(10)
                << std::setfill('0')
                << rand();

            fs::path tempFolder = basePath / os.str();

            if (!fs::exists(tempFolder))
            {
                bool ok = fs::create_directories(tempFolder);
                if (ok)
                {
                    return tempFolder;
                }
            }
        }

        // TODO
        RCF_ASSERT(0);
        return fs::path();
    }

    void trimManifest(
        const FileManifest & manifest, 
        boost::uint64_t & bytesAlreadyTransferred,
        FileChunk & startPos);

    void FileTransferService::BeginUpload(
        const FileManifest & manifest,
        const std::vector<FileChunk> & chunks,
        FileChunk & startPos,
        boost::uint32_t & maxMessageLength,
        boost::uint32_t & uploadId,
        boost::uint32_t & bps,
        boost::uint32_t sessionLocalId)
    {

        RCF_LOG_3()(sessionLocalId) << "FileTransferService::BeginUpload() - entry.";

        namespace fs = boost::filesystem;

        I_SessionState & sessionState = getCurrentRcfSession().getSessionState();
        maxMessageLength = sessionState.getServerTransport().getMaxMessageLength();

        FileUploadInfoPtr uploadInfoPtr( new FileUploadInfo(mUploadThrottle) );
        uploadInfoPtr->mManifest = manifest;
        uploadInfoPtr->mTimeStampMs = Platform::OS::getCurrentTimeMs();
        uploadInfoPtr->mSessionLocalId = sessionLocalId;

        // Check the access callback (if any).
        if (    !mUploadCallback.empty()
            &&  !mUploadCallback(*uploadInfoPtr))
        {
            RCF_THROW( Exception(_RcfError_AccessDenied()) );
        }

        if (uploadId)
        {
            fs::path uploadPath = findUpload(uploadId);
            if (!uploadPath.empty())
            {
                uploadInfoPtr->mUploadPath = uploadPath;

                // Trim the manifest to account for already uploaded fragments.
                const_cast<FileManifest &>(manifest).mManifestBase = uploadPath;

                boost::uint64_t bytesAlreadyTransferred = 0;
                trimManifest(manifest, bytesAlreadyTransferred, startPos);
                uploadInfoPtr->mResume = true;
            }
        }

        if (uploadInfoPtr->mUploadPath.empty())
        {
            // Create a temp folder to upload to.
            uploadInfoPtr->mUploadPath = makeTempDir(
                fs::path(mUploadDirectory), 
                "RCF-Upload-");

            uploadId = addUpload(uploadInfoPtr->mUploadPath);
            startPos = FileChunk();
        }

        uploadInfoPtr->mUploadId = uploadId;
        uploadInfoPtr->mCurrentFile = startPos.mFileIndex;
        uploadInfoPtr->mCurrentPos = startPos.mOffset;

        {
            RcfSession& session = getCurrentRcfSession();
            Lock lock(session.mMutex);
            session.mUploadInfoPtr = uploadInfoPtr;
        }

        Throttle::Reservation & reservation = uploadInfoPtr->mReservation;
        reservation.allocate();
        bps = reservation.getBps();
        if (bps == boost::uint32_t(-1))
        {
            bps = 0;
        }

        RCF_LOG_3()(startPos.mFileIndex)(startPos.mOffset)(maxMessageLength)(uploadId)(bps) 
            << "FileTransferService::BeginUpload() - exit.";
    }

    void FileTransferService::UploadChunks(
        const std::vector<FileChunk> & chunks,
        boost::uint32_t & bps)
    {
        RCF_LOG_3()(chunks.size()) 
            << "FileTransferService::UploadChunks() - entry.";

        namespace fs = boost::filesystem;

        // Find the upload.
        FileUploadInfoPtr uploadInfoPtr = getCurrentRcfSession().mUploadInfoPtr;
        
        if (!uploadInfoPtr)
        {
            RCF_THROW( Exception(_RcfError_NoUpload()) );
        }

        FileUploadInfo & uploadInfo = * uploadInfoPtr;

        if (uploadInfo.mCompleted)
        {
            // TODO: better error message.
            RCF_THROW( Exception(_RcfError_NoUpload()) );
        }

        Throttle::Reservation & reservation = uploadInfo.mReservation;
        reservation.allocate();
        bps = reservation.getBps();
        if (bps == boost::uint32_t(-1))
        {
            bps = 0;
        }

        for (std::size_t i=0; i<chunks.size(); ++i)
        {
            const FileChunk & chunk = chunks[i];

            if (chunk.mFileIndex != uploadInfo.mCurrentFile)
            {
                // TODO: better error message.
                RCF_THROW( Exception(_RcfError_FileOffset()) );
            }

            FileInfo file = uploadInfo.mManifest.mFiles[uploadInfo.mCurrentFile];

            if (uploadInfo.mCurrentPos == 0 || uploadInfo.mResume)
            {
                RCF_ASSERT(uploadInfo.mWriteOp->completed());

                fs::path filePath =  file.mFilePath;
                filePath = uploadInfo.mUploadPath / filePath;

                if ( !fs::exists( filePath.branch_path() ) )
                {
                    fs::create_directories( filePath.branch_path() );
                }

                if (file.mRenameFile.size() > 0)
                {
                    filePath = filePath.branch_path() / file.mRenameFile;
                }

                RCF_LOG_3()(uploadInfo.mCurrentFile)(filePath) 
                    << "FileTransferService::UploadChunks() - opening file.";

                if (uploadInfo.mResume)
                {
                    uploadInfo.mFileStream->open( 
                        filePath.string().c_str(), 
                        std::ios::binary | std::ios::app );
                }
                else
                {
                    uploadInfo.mFileStream->open( 
                        filePath.string().c_str(), 
                        std::ios::binary | std::ios::trunc );
                }

                if (! uploadInfo.mFileStream->good())
                {
                    RCF_THROW( Exception(_RcfError_FileWrite(filePath.string(), 0)));
                }

                if (uploadInfo.mResume && uploadInfo.mCurrentPos > 0)
                {
                    RCF_LOG_3()(uploadInfo.mCurrentFile)(uploadInfo.mCurrentPos) 
                        << "FileTransferService::UploadChunks() - seeking in file.";

                    uploadInfo.mFileStream->seekp( 
                        static_cast<std::streamoff>(uploadInfo.mCurrentPos) );
                }

                uploadInfo.mResume = false;
            }

            OfstreamPtr fout = uploadInfoPtr->mFileStream;

            // Wait for previous write to complete.
            if (uploadInfo.mWriteOp->initiated())
            {
                uploadInfo.mWriteOp->complete();

                boost::uint64_t bytesWritten = uploadInfo.mWriteOp->getBytesTransferred();
                if (bytesWritten == 0)
                {
                    fout->close();
                    // TODO: args
                    RCF_THROW( Exception(_RcfError_FileWrite("", 0)) );
                }
            }

            // Initiate next write.

            // Check stream state.
            if (!fout->good())
            {
                // TODO: args
                RCF_THROW( Exception(_RcfError_FileWrite("", 0)) );
            }

            // Check the offset position.
            uploadInfo.mCurrentPos = fout->tellp();
            if (chunk.mOffset != uploadInfo.mCurrentPos)
            {
                RCF_THROW( Exception(_RcfError_FileOffset()) );
            }

            // Check the chunk size.
            boost::uint64_t fileSize = file.mFileSize;
            boost::uint64_t remainingFileSize = fileSize - uploadInfo.mCurrentPos;
            if (chunk.mData.getLength() > remainingFileSize)
            {
                RCF_THROW( Exception(_RcfError_UploadFileSize()) );
            }

            uploadInfo.mWriteOp->write(uploadInfo.mFileStream, chunk.mData);

            uploadInfoPtr->mTimeStampMs = Platform::OS::getCurrentTimeMs();

            // Check if last chunk.
            uploadInfo.mCurrentPos += chunk.mData.getLength();
            if (uploadInfo.mCurrentPos == fileSize)
            {
                RCF_LOG_3()(uploadInfo.mCurrentFile) 
                    << "FileTransferService::UploadChunks() - closing file.";

                uploadInfo.mWriteOp->complete();
                fout->close();
                ++uploadInfo.mCurrentFile;
                uploadInfo.mCurrentPos = 0;

                if (uploadInfo.mCurrentFile == uploadInfo.mManifest.mFiles.size())
                {
                    RCF_LOG_3()(uploadInfo.mCurrentFile) 
                        << "FileTransferService::UploadChunks() - upload completed.";

                    uploadInfo.mCompleted = true;

                    if (uploadInfo.mSessionLocalId)
                    {
                        RcfSession& session = getCurrentRcfSession();
                        Lock lock(session.mMutex);
                        session.mSessionUploads[uploadInfo.mSessionLocalId] = uploadInfoPtr;
                        session.mUploadInfoPtr.reset();
                    }

                    if (uploadInfo.mUploadId)
                    {
                        removeUpload(uploadInfo.mUploadId);
                    }
                }
            }
        }

        RCF_LOG_3() << "FileTransferService::UploadChunks() - exit.";
    }

    namespace fs = boost::filesystem;

    boost::uint32_t FileTransferService::addUpload(const fs::path & uploadPath)
    {
        Lock lock(mUploadsInProgressMutex);

        boost::uint32_t uploadId = 0;
        while (true)
        {
            uploadId = rand();
            if (mUploadsInProgress.find(uploadId) == mUploadsInProgress.end())
            {
                break;
            }
        }
        mUploadsInProgress[uploadId] = uploadPath; 
        return uploadId;
    }

    void FileTransferService::removeUpload(boost::uint32_t uploadId)
    {
        Lock lock(mUploadsInProgressMutex);

        std::map<boost::uint32_t, fs::path>::iterator iter = 
            mUploadsInProgress.find(uploadId);

        if (iter != mUploadsInProgress.end())
        {
            mUploadsInProgress.erase(iter);
        }
    }

    fs::path FileTransferService::findUpload(boost::uint32_t uploadId)
    {
        Lock lock(mUploadsInProgressMutex);

        std::map<boost::uint32_t, fs::path>::iterator iter = 
            mUploadsInProgress.find(uploadId);

        if (iter != mUploadsInProgress.end())
        {
            return iter->second;
        }

        return fs::path();
    }

    void FileTransferService::BeginDownload(
        FileManifest & manifest,
        const FileTransferRequest & request,
        std::vector<FileChunk> & chunks,
        boost::uint32_t & maxMessageLength,
        boost::uint32_t sessionLocalId)
    {
        RCF_LOG_3()(sessionLocalId) << "FileTransferService::BeginDownload() - entry.";

        FileDownloadInfoPtr downloadInfoPtr;

        if (sessionLocalId)
        {
            RcfSession& session = getCurrentRcfSession();
            Lock lock(session.mMutex);
            FileStream & fs = session.mSessionDownloads[sessionLocalId];
            downloadInfoPtr.reset( new FileDownloadInfo(mDownloadThrottle) );
            downloadInfoPtr->mManifest = fs.mImplPtr->mManifest;
            downloadInfoPtr->mDownloadPath = downloadInfoPtr->mManifest.mManifestBase;
            downloadInfoPtr->mSessionLocalId = sessionLocalId;
            session.mDownloadInfoPtr = downloadInfoPtr;
        }
        else
        {
            RCF_THROW( Exception(_RcfError_NoDownload()) );
        }
        
        FileDownloadInfo & di = * downloadInfoPtr;

        manifest = di.mManifest;
        di.mCurrentFile = 0;
        di.mCurrentPos = 0;
        if (!di.mManifest.mFiles.empty())
        {
            di.mCurrentPos = di.mManifest.mFiles[0].mFileStartPos;
        }

        // TODO: optional first chunks.
        RCF_UNUSED_VARIABLE(request);
        chunks.clear();

        I_SessionState & sessionState = getCurrentRcfSession().getSessionState();
        maxMessageLength = sessionState.getServerTransport().getMaxMessageLength();

        RCF_LOG_3()(manifest.mFiles.size())(maxMessageLength) 
            << "FileTransferService::BeginDownload() - exit.";
    }

    void FileTransferService::TrimDownload(
        const FileChunk & startPos)
    {
        RCF_LOG_3()(startPos.mFileIndex)(startPos.mOffset) 
            << "FileTransferService::TrimDownload() - entry.";

        FileDownloadInfoPtr downloadInfoPtr = getCurrentRcfSession().mDownloadInfoPtr;

        if (!downloadInfoPtr)
        {
            RCF_THROW( Exception(_RcfError_NoDownload()) );
        }

        FileDownloadInfo & di = * downloadInfoPtr;

        RCF_ASSERT_LTEQ(startPos.mFileIndex , di.mManifest.mFiles.size());
        if (startPos.mFileIndex < di.mManifest.mFiles.size())
        {
            RCF_ASSERT_LT(startPos.mOffset , di.mManifest.mFiles[startPos.mFileIndex].mFileSize)(startPos.mFileIndex);
        }
        else
        {
            RCF_ASSERT_EQ(startPos.mOffset , 0);
        }

        di.mCurrentFile = startPos.mFileIndex;
        di.mCurrentPos = startPos.mOffset;
        di.mResume = true;

        RCF_LOG_3() << "FileTransferService::TrimDownload() - exit.";
    }

    void FileTransferService::DownloadChunks(
        const FileTransferRequest & request,
        std::vector<FileChunk> & chunks,
        boost::uint32_t & adviseWaitMs)
    {
        RCF_LOG_3()(request.mFile)(request.mPos) 
            << "FileTransferService::DownloadChunks() - entry.";

        // Find the download.
        FileDownloadInfoPtr & diPtr = getCurrentRcfSession().mDownloadInfoPtr;

        if (!diPtr)
        {
            RCF_THROW( Exception(_RcfError_NoDownload()) );
        }

        FileDownloadInfo & di = *diPtr;

        if (di.mCancel)
        {
            // TODO: reset mDownloadInfoPtr?
            RCF_THROW( Exception(_RcfError_DownloadCancelled()) );
        }

        adviseWaitMs = 0;


        // Check offset.
        if (    request.mFile != di.mCurrentFile 
            ||  request.mPos != di.mCurrentPos)
        {
            RCF_THROW( Exception(_RcfError_FileOffset()) );
        }

        chunks.clear();

        boost::uint32_t chunkSize = request.mChunkSize;

        // Trim the chunk size, according to throttle settings.
        Throttle::Reservation & reservation = di.mReservation;
        reservation.allocate();
        boost::uint32_t bps = reservation.getBps();
        if (bps != boost::uint32_t(-1))
        {
            const boost::uint32_t mThrottleRateBytesPerS = bps;

            if (mThrottleRateBytesPerS)
            {
                if (di.mTransferWindowTimer.elapsed(mTransferWindowS*1000))
                {
                    RCF_ASSERT_GTEQ(di.mTransferWindowBytesTotal , di.mTransferWindowBytesSoFar);

                    boost::uint32_t carryOver = 
                        di.mTransferWindowBytesTotal - di.mTransferWindowBytesSoFar;

                    di.mTransferWindowTimer.restart();

                    di.mTransferWindowBytesTotal = mThrottleRateBytesPerS * mTransferWindowS;
                    di.mTransferWindowBytesTotal += carryOver;

                    di.mTransferWindowBytesSoFar = 0;

                    RCF_LOG_3()(mTransferWindowS)(di.mTransferWindowBytesTotal)(di.mTransferWindowBytesSoFar)(carryOver) 
                        << "FileTransferService::DownloadChunks() - new throttle transfer window.";
                }

                if (di.mTransferWindowBytesTotal == 0)
                {
                    di.mTransferWindowBytesTotal = mThrottleRateBytesPerS * mTransferWindowS;
                }

                boost::uint32_t bytesWindowRemaining = 
                    di.mTransferWindowBytesTotal - di.mTransferWindowBytesSoFar;

                if (bytesWindowRemaining < chunkSize)
                {
                    boost::uint32_t windowStartMs = di.mTransferWindowTimer.getStartTimeMs();
                    boost::uint32_t windowEndMs = windowStartMs + 1000*mTransferWindowS;
                    boost::uint32_t nowMs = getCurrentTimeMs();
                    if (nowMs < windowEndMs)
                    {
                        adviseWaitMs = windowEndMs - nowMs;

                        RCF_LOG_3()(adviseWaitMs) 
                            << "FileTransferService::DownloadChunks() - advising client wait.";
                    }
                }

                RCF_LOG_3()(chunkSize)(bytesWindowRemaining)(di.mTransferWindowBytesTotal) 
                    << "FileTransferService::DownloadChunks() - trimming chunk size to transfer window.";

                chunkSize = RCF_MIN(chunkSize, bytesWindowRemaining);
            }
        }

        boost::uint32_t totalBytesRead = 0;

        while (
                totalBytesRead < chunkSize 
            &&  di.mCurrentFile != di.mManifest.mFiles.size())
        {
            FileInfo & currentFileInfo = di.mManifest.mFiles[di.mCurrentFile];

            if (di.mCurrentPos == 0 || di.mResume)
            {
                di.mResume = false;

                fs::path manifestBase = di.mDownloadPath;
                FileInfo & currentFileInfo = di.mManifest.mFiles[di.mCurrentFile];
                fs::path filePath = currentFileInfo.mFilePath;
                fs::path totalPath = manifestBase / filePath;

                RCF_LOG_3()(di.mCurrentFile)(totalPath)
                    << "FileTransferService::DownloadChunks() - opening file.";

                di.mFileStream->clear();
                di.mFileStream->open(
                    totalPath.string().c_str(),
                    std::ios::in | std::ios::binary);

                // TODO: error handling.
                RCF_VERIFY(
                    di.mFileStream->good(), 
                    Exception(_RcfError_FileOpen(totalPath.file_string())));

                if (di.mCurrentPos != 0)
                {
                    RCF_LOG_3()(di.mCurrentFile)(di.mCurrentPos) 
                        << "FileTransferService::DownloadChunks() - seeking in file.";

                    di.mFileStream->seekg( static_cast<std::streamoff>(di.mCurrentPos) );

                    RCF_VERIFY(
                        di.mFileStream->good(), 
                        Exception(_RcfError_FileSeek(totalPath.file_string(), diPtr->mCurrentPos)));
                }
            }
            
            boost::uint64_t fileSize = currentFileInfo.mFileSize;
            boost::uint64_t bytesRemainingInFile =  fileSize - di.mCurrentPos;
            boost::uint64_t bytesRemainingInChunk = chunkSize - totalBytesRead;

            if (di.mReadOp->initiated())
            {
                RCF_LOG_3() 
                    << "FileTransferService::DownloadChunks() - completing read.";

                // Wait for async read to complete.
                di.mReadOp->complete();
                std::size_t bytesRead = static_cast<std::size_t>(
                    di.mReadOp->getBytesTransferred());

                if (bytesRead == 0)
                {
                    // TODO: args
                    RCF_THROW( Exception(_RcfError_FileRead("", 0)));
                }
                di.mSendBuffer.swap(di.mReadBuffer);
                di.mSendBufferRemaining = ByteBuffer(di.mSendBuffer, 0, bytesRead);

                RCF_LOG_3()(bytesRead) 
                    << "FileTransferService::DownloadChunks() - read completed.";
            }

            ByteBuffer byteBuffer;
            IfstreamPtr fin = di.mFileStream;

            if (di.mSendBufferRemaining)
            {
                std::size_t bytesToRead = RCF_MIN(
                    di.mSendBufferRemaining.getLength(), 
                    static_cast<std::size_t>(bytesRemainingInChunk));

                byteBuffer = ByteBuffer(di.mSendBufferRemaining, 0, bytesToRead);
                di.mSendBufferRemaining = ByteBuffer(di.mSendBufferRemaining, bytesToRead);
            }
            else
            {
                // No asynchronously data available. Do a synchronous read.
                byteBuffer = ByteBuffer( static_cast<std::size_t>( 
                    RCF_MIN(bytesRemainingInChunk, bytesRemainingInFile) ));

                RCF_LOG_3()(di.mCurrentFile)(byteBuffer.getLength()) 
                    << "FileTransferService::DownloadChunks() - reading from file.";

                boost::uint32_t bytesRead = static_cast<boost::uint32_t>(fin->read( 
                    byteBuffer.getPtr(), 
                    static_cast<std::size_t>(byteBuffer.getLength()) ).gcount());

                if (fin->fail() && !fin->eof())
                {
                    // TODO: error handling
                    // TODO: args
                    RCF_THROW( Exception(_RcfError_FileRead("", 0)) );
                }

                byteBuffer = ByteBuffer(byteBuffer, 0, bytesRead);
            }

            FileChunk fileChunk;
            fileChunk.mFileIndex = di.mCurrentFile;
            fileChunk.mOffset = di.mCurrentPos;
            fileChunk.mData = byteBuffer;
            chunks.push_back(fileChunk);

            totalBytesRead += byteBuffer.getLength();
            diPtr->mCurrentPos += byteBuffer.getLength();

            if (diPtr->mCurrentPos == currentFileInfo.mFileSize)
            {
                RCF_LOG_3()(diPtr->mCurrentFile) 
                    << "FileTransferService::DownloadChunks() - closing file.";

                fin->close();
                ++di.mCurrentFile;
                di.mCurrentPos = 0;

                if (di.mCurrentFile < di.mManifest.mFiles.size())
                {
                    FileInfo & nextFile = di.mManifest.mFiles[di.mCurrentFile];
                    di.mCurrentPos = nextFile.mFileStartPos;
                }
            }
        }

        di.mTransferWindowBytesSoFar += totalBytesRead;

        if (di.mCurrentFile == di.mManifest.mFiles.size())
        {
            RCF_LOG_3()(di.mCurrentFile) 
                << "FileTransferService::DownloadChunks() - download completed.";

            di.mReservation.release();

            // TODO: this is broken if there is more than one FileStream.
            if (diPtr->mSessionLocalId)
            {
                std::map<boost::uint32_t, FileDownload> & downloads = 
                    getCurrentRcfSession().mSessionDownloads;

                std::map<boost::uint32_t, FileDownload>::iterator iter = 
                    downloads.find(diPtr->mSessionLocalId);

                RCF_ASSERT(iter != downloads.end());

                downloads.erase(iter);
            }
            diPtr.reset();
        }

        // Initiate read for next chunk.
        if (    di.mSendBufferRemaining.isEmpty()
            &&  di.mCurrentFile < di.mManifest.mFiles.size()
            &&  0 < di.mCurrentPos
            &&  ! di.mReadOp->initiated())
        {
            boost::uint64_t fileSize = di.mManifest.mFiles[di.mCurrentFile].mFileSize;
            if (di.mCurrentPos < fileSize)
            {
                if (di.mReadBuffer.isEmpty())
                {
                    RCF_ASSERT(di.mSendBuffer.isEmpty());
                    RCF_ASSERT(di.mSendBufferRemaining.isEmpty());

                    di.mReadBuffer = ByteBuffer(request.mChunkSize);
                    di.mSendBuffer = ByteBuffer(request.mChunkSize);
                }

                std::size_t bytesToRead = static_cast<std::size_t>(
                    fileSize - di.mCurrentPos);

                bytesToRead = RCF_MIN(bytesToRead, di.mReadBuffer.getLength());

                RCF_LOG_3()(di.mCurrentFile)(di.mCurrentPos)(fileSize)(bytesToRead) 
                    << "FileTransferService::DownloadChunks() - initiate read for next chunk.";

                di.mReadOp->read(di.mFileStream, ByteBuffer(di.mReadBuffer, 0, bytesToRead));
            }
        }

        RCF_LOG_3()(chunks.size()) 
            << "FileTransferService::DownloadChunks() - exit.";
    }

    void FileTransferService::setDownloadThrottleSettings(const ThrottleSettings & settings)
    {
        mDownloadThrottle.set(settings);
    }

    void FileTransferService::getDownloadThrottleSettings(ThrottleSettings & settings)
    {
        mDownloadThrottle.get(settings);
    }

    void FileTransferService::setUploadThrottleSettings(const ThrottleSettings & settings)
    {
        mUploadThrottle.set(settings);
    }

    void FileTransferService::getUploadThrottleSettings(ThrottleSettings & settings)
    {
        mUploadThrottle.get(settings);
    }

    void FileTransferService::setUploadCallback(UploadAccessCallback uploadCallback)
    {
        mUploadCallback = uploadCallback;
    }

    void FileTransferService::onServiceAdded(RcfServer &server)
    {
        server.bind( (I_FileTransferService *) NULL, *this);
    }

    void FileTransferService::onServiceRemoved(RcfServer &server)
    {
        server.unbind( (I_FileTransferService *) NULL);
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void FileManifest::serialize(SF::Archive & ar) 
    {
        ar & mFiles;
    }

    void FileInfo::serialize(SF::Archive & ar) 
    {
        ar & mFilePath & mFileStartPos & mFileSize & mFileCrc & mRenameFile;
    }

    void FileChunk::serialize(SF::Archive & ar)
    {
        ar & mFileIndex & mOffset & mData;
    }

    void FileTransferRequest::serialize(SF::Archive & ar)
    {
        ar & mFile & mPos & mChunkSize;
    }

#endif

    FileManifest::FileManifest(boost::filesystem::path pathToFiles) 
    {
        if (!fs::exists(pathToFiles))
        {
            RCF::Exception e( _RcfError_FileOpen(pathToFiles.file_string()) );
            RCF_THROW(e)(pathToFiles.file_string());
        }

        if (fs::is_directory(pathToFiles))
        {
            for ( 
                fs::recursive_directory_iterator iter(pathToFiles); 
                iter != fs::recursive_directory_iterator(); 
                ++iter )
            { 

                fs::path fullPath = *iter;
                if ( !fs::is_directory(fullPath) )
                {
                    FileInfo fileInfo;
                    fileInfo.mFileSize = fs::file_size(fullPath);
                    fileInfo.mFileCrc = 0;

                    // TODO: this is a bit of a kludge.
                    std::string base = (pathToFiles / "..").normalize().string();
                    std::string full = fullPath.string();
                    RCF_ASSERT_EQ(full.substr(0, base.length()) , base);
                    std::string relative = full.substr(base.length()+1);
                    fs::path relativePath(relative);

                    fileInfo.mFilePath = relativePath;

                    mFiles.push_back(fileInfo);
                }
            }
        }
        else
        {
            FileInfo fileInfo;
            fileInfo.mFileSize = fs::file_size(pathToFiles);
            fileInfo.mFileCrc = 0;
            fileInfo.mFilePath = pathToFiles.leaf();
            mFiles.push_back(fileInfo);
        }

        if (fs::is_directory( fs::path(pathToFiles) ))
        {
            mManifestBase = (fs::path(pathToFiles) / "..").normalize();
        }
        else
        {
            mManifestBase = fs::path(pathToFiles).branch_path();
        }
    }

    boost::uint64_t FileManifest::getTotalByteSize() const
    {
        boost::uint64_t totalByteSize = 0;
        for (std::size_t i=0; i<mFiles.size(); ++i)
        {
            const FileInfo & fi = mFiles[i];
            boost::uint64_t fileSize = fi.mFileSize - fi.mFileStartPos;
            totalByteSize += fileSize;
        }
        return totalByteSize;
    }

    FileUploadInfo::FileUploadInfo(Throttle & uploadThrottle) : 
        mFileStream( new std::ofstream() ),
        mWriteOp( new FileIoRequest() ),
        mCompleted(false),
        mResume(false),
        mTimeStampMs(0),
        mCurrentFile(0),
        mCurrentPos(0),
        mSessionLocalId(0),
        mUploadId(0),
        mReservation(uploadThrottle)
    {}

    FileDownloadInfo::FileDownloadInfo(Throttle & downloadThrottle) :
        mFileStream( new std::ifstream() ),
        mReadOp( new FileIoRequest() ),
        mCurrentFile(0),
        mCurrentPos(0),
        mResume(false),
        mTransferWindowBytesSoFar(0),
        mTransferWindowBytesTotal(0),
        mReservation(downloadThrottle),
        mCancel(false),
        mSessionLocalId(0)
    {}

    Throttle::Reservation::Reservation(Throttle & throttle) : 
        mThrottle(throttle), 
        mBps(0),
        mReserved(false),
        mChangeCounter(0)
    {
    }

    Throttle::Reservation::~Reservation()
    {
        release();
    }

    void Throttle::Reservation::allocate()
    {
        // See if we can upgrade our existing reservation, or create one if necessary.
        if (mBps != boost::uint32_t(-1))
        {
            Lock lock(mThrottle.mMutex);
            if (mThrottle.mSettings.empty())
            {
                release(false);
                
                mBps = boost::uint32_t(-1);
                mReserved = false;
            }
            else
            {
                for (
                    Settings::reverse_iterator iter = mThrottle.mSettings.rbegin(); 
                    iter != mThrottle.mSettings.rend(); 
                    ++iter)
                {
                    boost::uint32_t bps = iter->first;
                    boost::uint32_t maxAllowed = iter->second.first;
                    boost::uint32_t & booked = iter->second.second;

                    if (bps <= mBps)
                    {
                        break;
                    }
                    
                    if (booked < maxAllowed)
                    {
                        release(false);

                        ++booked;
                        mBps = bps;
                        mReserved = true;
                        mChangeCounter = mThrottle.mChangeCounter;

                        break;
                    }
                }
            }
        }
    }

    void Throttle::Reservation::release(bool sync)
    {
        if (mReserved)
        {
            if (sync)
            {
                Lock lock(mThrottle.mMutex);
                releaseImpl();
            }
            else
            {
                releaseImpl();
            }
        }
    }

    void Throttle::Reservation::releaseImpl()
    {
        RCF_ASSERT(mReserved);
        if (mChangeCounter == mThrottle.mChangeCounter)
        {
            Settings::iterator iter = mThrottle.mSettings.find(mBps);
            RCF_ASSERT(iter != mThrottle.mSettings.end());
            boost::uint32_t & booked = iter->second.second;
            --booked;
            RCF_ASSERT_NEQ(booked , boost::uint32_t(-1));
            mBps = 0;
            mReserved = false;
        }
    }

    boost::uint32_t Throttle::Reservation::getBps()
    {
        return mBps;
    }

    Throttle::Throttle() : mChangeCounter(1)
    {
    }

    void Throttle::set(const ThrottleSettings & settings)
    {
        Lock lock(mMutex);
        ++mChangeCounter;
        mSettings.clear();
        for (std::size_t i=0; i<settings.size(); ++i)
        {
            boost::uint32_t users = settings[i].first;
            boost::uint32_t bps = settings[i].second;
            mSettings[bps].first = users;
            mSettings[bps].second = 0;
        }
    }

    void Throttle::get(ThrottleSettings& settings)
    {
        settings.clear();
        Lock lock(mMutex);
        for (Settings::iterator iter = mSettings.begin(); iter != mSettings.end(); ++iter)
        {
            settings.push_back( std::make_pair(iter->second.first, iter->first) );
        }
    }

    void FileTransferService::setTransferWindowS(boost::uint32_t transferWindowS)
    {
        mTransferWindowS = transferWindowS;
    }

    boost::uint32_t FileTransferService::getTransferWindowS()
    {
        return mTransferWindowS;
    }

} // namespace RCF
