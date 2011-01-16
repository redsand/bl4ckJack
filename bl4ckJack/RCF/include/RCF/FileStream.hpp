
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

#ifndef INCLUDE_RCF_FILESTREAM_HPP
#define INCLUDE_RCF_FILESTREAM_HPP

#ifndef RCF_USE_BOOST_FILESYSTEM
#error RCF_USE_BOOST_FILESYSTEM must be defined
#endif

#include <RCF/Config.hpp>
#include <RCF/Export.hpp>

#include <boost/filesystem/path.hpp>

#ifdef RCF_USE_BOOST_SERIALIZATION
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#endif

namespace SF {

    class Archive;

} // namespace SF

namespace RCF {

    class FileUploadInfo;
    typedef boost::shared_ptr<FileUploadInfo> FileUploadInfoPtr;

    class FileInfo;
    class FileManifest;
    class FileStreamImpl;

    typedef boost::shared_ptr<FileStreamImpl> FileStreamImplPtr;

    class ClientStub;

    class RCF_EXPORT FileInfo
    {
    public:
        FileInfo() : mFileStartPos(0), mFileSize(0), mFileCrc(0) 
        {}

        boost::filesystem::path mFilePath;
        boost::uint64_t mFileStartPos;
        boost::uint64_t mFileSize;
        boost::uint32_t mFileCrc;
        std::string mRenameFile;

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & mFilePath & mFileStartPos & mFileSize & mFileCrc & mRenameFile;
        }
#endif

    };

    class RCF_EXPORT FileManifest
    {
    public:
        typedef std::vector< FileInfo > Files;
        Files mFiles;

        boost::filesystem::path mManifestBase;

        FileManifest() {}

        FileManifest(boost::filesystem::path pathToFiles);

        boost::uint64_t getTotalByteSize() const;

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & mFiles;
        }
#endif

    };

    class RCF_EXPORT FileStreamImpl : public boost::enable_shared_from_this<FileStreamImpl>
    {
    public:
        
        enum Direction
        {
            Unspecified,
            Upload,
            Download
        };


        FileStreamImpl();
        FileStreamImpl(const std::string & filePath);
        FileStreamImpl(const FileManifest & manifest);

        ~FileStreamImpl();

        void serializeGeneric(
            bool isWriting,
            boost::function2<void, boost::uint32_t &, Direction &> serializeImpl);

#ifdef RCF_USE_SF_SERIALIZATION
        void serializeImplSf(SF::Archive & ar, boost::uint32_t & transferId, Direction & dir);
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION

        template<typename Archive>
        void serializeImplBser(Archive & ar, boost::uint32_t & transferId, Direction & dir)
        {
            ar & transferId & dir;
        }

        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            typedef typename Archive::is_saving IsSaving;
            const bool isSaving = IsSaving::value;

            serializeGeneric( 
                isSaving,
                boost::bind( 
                    &FileStreamImpl::serializeImplBser<Archive>, 
                    this, 
                    boost::ref(ar),
                    _1, 
                    _2) );
        }

#endif

        boost::uint32_t mUploadId;
        boost::filesystem::path mDownloadPath;
        FileManifest mManifest;
        boost::uint32_t mTransferRateBps;
        boost::uint32_t mSessionLocalId;

        Direction mDirection;
    };

    class RCF_EXPORT FileStream
    {
    protected:

        FileStream();
        FileStream(FileStreamImplPtr implPtr);
        FileStream(const std::string & filePath);
        FileStream(const FileManifest & manifest);
  
    public:

        // Made this inline as it was not being linked in, in DLL builds.
        FileStream & operator=(const FileStream & rhs)
        {
            *mImplPtr = *rhs.mImplPtr;
            return *this;
        }

        // FileStream recipient calls these.
        std::string getLocalPath() const;
        FileManifest & getManifest() const;

        // Client calls these.
        void setDownloadPath(const std::string & downloadPath);
        std::string getDownloadPath() const;

        void setTransferRateBps(boost::uint32_t transferRateBps);
        boost::uint32_t getTransferRateBps();
        
        void upload(RCF::ClientStub & clientStub);
        void download(RCF::ClientStub & clientStub);

        FileStreamImplPtr mImplPtr;

#ifdef RCF_USE_SF_SERIALIZATION
        void serialize(SF::Archive & ar);
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            ar & *mImplPtr;
        }
#endif

    };

    class RCF_EXPORT FileUpload : public FileStream
    {
    public:
        FileUpload();
        FileUpload(const std::string & filePath);
        FileUpload(const FileManifest & manifest);
        FileUpload(FileStreamImplPtr implPtr);
    };

    class RCF_EXPORT FileDownload : public FileStream
    {
    public:
        FileDownload();
        FileDownload(const std::string & filePath);
        FileDownload(const FileManifest & manifest);
        FileDownload(FileStreamImplPtr implPtr);
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_FILESTREAM_HPP
