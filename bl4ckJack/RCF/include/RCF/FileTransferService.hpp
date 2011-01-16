
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

#ifndef INCLUDE_RCF_FILETRANSFERSERVICE_HPP
#define INCLUDE_RCF_FILETRANSFERSERVICE_HPP

#include <fstream>
#include <map>

#include <RCF/FileIoThreadPool.hpp>
#include <RCF/FileStream.hpp>
#include <RCF/Idl.hpp>
#include <RCF/Service.hpp>
#include <RCF/StubEntry.hpp>
#include <RCF/Timer.hpp>
#include <RCF/Token.hpp>

#include <SF/vector.hpp>
#include <SF/map.hpp>

#include <boost/filesystem/path.hpp>

#ifdef RCF_USE_SF_SERIALIZATION

namespace boost { namespace filesystem {

    RCF_EXPORT void serialize(SF::Archive &ar, path &p);
} }

#endif

#ifdef RCF_USE_BOOST_SERIALIZATION

namespace boost { namespace filesystem {

    template<typename Archive>
    void serialize(Archive & ar, path & p, const unsigned int)
    {
        typedef typename Archive::is_saving IsSaving;
        const bool isSaving = IsSaving::value;

        if (isSaving)
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

} }

#endif


namespace RCF {

    //--------------------------------------------------------------------------
    // I_FileTransferService
    
    class FileChunk
    {
    public:

        FileChunk() : mFileIndex(0), mOffset(0)
        {}

        boost::uint32_t mFileIndex;
        boost::uint64_t mOffset;
        ByteBuffer mData;

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & mFileIndex & mOffset & mData;
        }
#endif

    };

    class FileTransferRequest
    {
    public:
        FileTransferRequest() : mFile(0), mPos(0), mChunkSize(0)
        {}

        boost::uint32_t mFile;
        boost::uint64_t mPos;
        boost::uint32_t mChunkSize;

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & mFile & mPos & mChunkSize;
        }
#endif

    };


    RCF_BEGIN(I_FileTransferService, "I_FileTransferService")

        RCF_METHOD_V7(
            void,
                BeginUpload,
                    const FileManifest &,           // upload manifest
                    const std::vector<FileChunk> &, // optional first chunks
                    FileChunk &,                    // where to start uploading
                    boost::uint32_t &,              // max message length
                    boost::uint32_t &,              // upload id
                    boost::uint32_t &,              // bps
                    boost::uint32_t)                // session local id
        
        RCF_METHOD_V2(
            void,
                UploadChunks, 
                    const std::vector<FileChunk> &, // file chunks to upload
                    boost::uint32_t &)              // bps

        RCF_METHOD_V5(
            void,
                BeginDownload,
                    FileManifest &,                 // download manifest
                    const FileTransferRequest &,    // transfer request
                    std::vector<FileChunk> &,       // optional first chunks
                    boost::uint32_t &,              // max message length
                    boost::uint32_t)                // session local id

        RCF_METHOD_V1(
            void,
                TrimDownload,
                    const FileChunk &)              // where to start downloading

        RCF_METHOD_V3(
            void,
                DownloadChunks,
                    const FileTransferRequest &,    // transfer request
                    std::vector<FileChunk> &,       // file chunks to download
                    boost::uint32_t &)              // advised wait for next call

    RCF_END(I_FileTransferService)

    typedef std::vector< std::pair<boost::uint32_t, boost::uint32_t> > ThrottleSettings;

    class Throttle
    {
    public:
        Throttle();

        void set(const ThrottleSettings & settings);
        void get(ThrottleSettings& settings);

        class Reservation
        {
        public:
            Reservation(Throttle & throttle);
            ~Reservation();

            void allocate();
            void release(bool sync = true);
            boost::uint32_t getBps();

        private:

            void releaseImpl();

            Throttle & mThrottle;
            unsigned int mChangeCounter;
            boost::uint32_t mBps;
            bool mReserved;
        };

    private:

        Mutex mMutex;

        // bps -> (max, booked)
        typedef 
        std::map<
            boost::uint32_t, 
            std::pair<boost::uint32_t, boost::uint32_t> > Settings;
        
        unsigned int mChangeCounter;
        Settings mSettings;
    };

    class FileUploadInfo;
    class FileDownloadInfo;

    typedef boost::shared_ptr<FileUploadInfo>                   FileUploadInfoPtr;
    typedef boost::shared_ptr<FileDownloadInfo>                 FileDownloadInfoPtr;

    typedef boost::function1<bool, const FileUploadInfo &>      UploadAccessCallback;
    typedef boost::function1<bool, const FileDownloadInfo &>    DownloadAccessCallback;
    
    class FileUploadInfo : 
        public TokenMapped, 
        boost::noncopyable
    {
    public:
        FileUploadInfo(Throttle & uploadThrottle);
        ~FileUploadInfo();

        FileManifest            mManifest;
        boost::filesystem::path mUploadPath;
        
        OfstreamPtr             mFileStream;
        FileIoRequestPtr        mWriteOp;

        bool                    mCompleted;
        bool                    mResume;
        boost::uint32_t         mTimeStampMs;
        boost::uint32_t         mCurrentFile;
        boost::uint64_t         mCurrentPos;
        boost::uint32_t         mSessionLocalId;
        boost::uint32_t         mUploadId;
        Throttle::Reservation   mReservation;
    };

    class FileDownloadInfo : 
        public TokenMapped, 
        boost::noncopyable
    {
    public:
        FileDownloadInfo(Throttle & downloadThrottle);
        ~FileDownloadInfo();
        
        boost::filesystem::path mDownloadPath;
        FileManifest            mManifest;

        IfstreamPtr             mFileStream;
        FileIoRequestPtr        mReadOp;
        ByteBuffer              mReadBuffer;
        ByteBuffer              mSendBuffer;
        ByteBuffer              mSendBufferRemaining;

        boost::uint32_t         mCurrentFile;
        boost::uint64_t         mCurrentPos;
        bool                    mResume;

        Timer                   mTransferWindowTimer;
        boost::uint32_t         mTransferWindowBytesSoFar;
        boost::uint32_t         mTransferWindowBytesTotal;

        Throttle::Reservation   mReservation;

        bool                    mCancel;

        boost::uint32_t         mSessionLocalId;
    };

    typedef boost::shared_ptr<FileUploadInfo>   FileUploadInfoPtr;
    typedef boost::shared_ptr<FileDownloadInfo> FileDownloadInfoPtr;

    class RCF_EXPORT FileTransferService : public I_Service
    {
    public:
        FileTransferService(const boost::filesystem::path & tempFileDirectory = "");

        void setDownloadThrottleSettings(const ThrottleSettings & settings);
        void getDownloadThrottleSettings(ThrottleSettings & settings);

        void setUploadThrottleSettings(const ThrottleSettings & settings);
        void getUploadThrottleSettings(ThrottleSettings & settings);

        void setUploadCallback(UploadAccessCallback uploadCallback);

        // For testing.
        void setTransferWindowS(boost::uint32_t transferWindowS);
        boost::uint32_t getTransferWindowS();

        //----------------------------------------------------------------------
        // Remotely accessible.

        void                BeginUpload(
                                const FileManifest & manifest,
                                const std::vector<FileChunk> & chunks,
                                FileChunk & startPos,
                                boost::uint32_t & maxMessageLength,
                                boost::uint32_t & uploadId,
                                boost::uint32_t & bps,
                                boost::uint32_t sessionLocalId);

        void                UploadChunks(
                                const std::vector<FileChunk> & chunks,
                                boost::uint32_t & bps);

        void                BeginDownload(
                                FileManifest & manifest,
                                const FileTransferRequest & request,
                                std::vector<FileChunk> & chunks,
                                boost::uint32_t & maxMessageLength,
                                boost::uint32_t sessionLocalId);

        void                TrimDownload(
                                const FileChunk & startPos);

        void                DownloadChunks(
                                const FileTransferRequest & request,
                                std::vector<FileChunk> & chunks,
                                boost::uint32_t & adviseWaitMs);

        //----------------------------------------------------------------------

    private:

        boost::uint32_t             addUpload(const boost::filesystem::path & uploadPath);
        void                        removeUpload(boost::uint32_t uploadId);
        boost::filesystem::path     findUpload(boost::uint32_t uploadId);

        typedef std::map<
            boost::uint32_t, 
            boost::filesystem::path> UploadsInProgress;

        Mutex                   mUploadsInProgressMutex;
        UploadsInProgress       mUploadsInProgress; 


        void                onServiceAdded(
                                RcfServer &         server);

        void                onServiceRemoved(
                                RcfServer &         server);

        boost::filesystem::path mUploadDirectory;

        UploadAccessCallback    mUploadCallback;

        Throttle                mDownloadThrottle;
        Throttle                mUploadThrottle;

        boost::uint32_t         mTransferWindowS;
    };

    typedef boost::shared_ptr<FileTransferService> FileTransferServicePtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_FILETRANSFERSERVICE_HPP
