
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

#include <RCF/Exception.hpp>

#include <RCF/ClientStub.hpp>
#include <RCF/Config.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ThreadLocalData.hpp>
#include <RCF/Tools.hpp>
#include <RCF/Version.hpp>

#include <RCF/util/Platform/OS/BsdSockets.hpp> // GetErrorString()

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/Archive.hpp>
#include <SF/string.hpp>
#include <SF/vector.hpp>
#endif

namespace RCF {

    std::string getErrorString(int rcfError)
    {
        Error error(rcfError);
        return error.getErrorString();
    }

    std::string Error::getErrorString() const
    {
        std::string s = getRawErrorString();

        // Replace placeholders with arguments.
        for (std::size_t i=0; i<mArgs.size(); ++i)
        {
            if (!mArgs[i].empty())
            {
                std::string placeHolder = "%";
                placeHolder += char('1' + i);

                std::size_t pos = s.find(placeHolder);
                if (pos != std::string::npos)
                {
                    s.replace(pos, placeHolder.length(), mArgs[i]);
                }
            }
        }

        return s;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void Error::serialize(SF::Archive & ar)
    {
        ar & mErrorId & mArgs;
    }

#endif

    std::string Error::getRawErrorString() const
    {
        int rcfError = mErrorId;

        if (rcfError >= RcfError_User)
        {
            return "non-RCF error";
        }

        switch (rcfError)
        {
        case RcfError_Ok                            :   return "No error.";
        case RcfError_Unspecified                   :   return "Unknown error.";
        case RcfError_ServerMessageLength           :   return "Server-side message length error.";
        case RcfError_ClientMessageLength           :   return "Client-side message length error. Incoming message length: %1. Max allowed message length: %2.";
        case RcfError_Serialization                 :   return "Data serialization error. Type name: %1. Exception type: %2. Exception message: %3.";
        case RcfError_Deserialization               :   return "Data deserialization error. Type name: %1. Exception type: %2. Exception message: %3.";
        case RcfError_UserModeException             :   return "Server-side user exception. Exception type: %1. Exception message: %2.";
        case RcfError_UnknownEndpoint               :   return "Unknown endpoint.";
        case RcfError_EndpointPassword              :   return "Incorrect endpoint password.";
        case RcfError_EndpointDown                  :   return "Endpoint unavailable.";
        case RcfError_EndpointRetry                 :   return "Endpoint temporarily unavailable (try again).";
        case RcfError_ClientConnectTimeout          :   return "Client connect operation to %2 timed out after %1 ms (server not started?).";
        case RcfError_PeerDisconnect                :   return "Unexpected peer disconnection.";
        case RcfError_ClientCancel                  :   return "Remote call cancelled by client.";
        case RcfError_PayloadFilterMismatch         :   return "Message filter mismatch.";
        case RcfError_OpenSslFilterInit             :   return "Failed to initialize OpenSSL filter. OpenSSL error: %1";
        case RcfError_OpenSslLoadCert               :   return "Failed to load OpenSSL certificate file. File: %1. OpenSSL error: %2";
        case RcfError_UnknownPublisher              :   return "Unknown publisher name: %1.";
        case RcfError_UnknownFilter                 :   return "Unknown filter type.";
        case RcfError_NoServerStub                  :   return "Server-side object not found. Service: %1. Interface: %2. Function id: %3.";
        case RcfError_Sspi                          :   return "SSPI filter error.";
        case RcfError_SspiInit                      :   return "Failed to initialize SSPI filter.";
        case RcfError_ClientReadTimeout             :   return "Client read operation timed out (no response from peer).";
        case RcfError_ClientReadFail                :   return "Client read operation failed.";
        case RcfError_ClientWriteTimeout            :   return "Client write operation timed out.";
        case RcfError_ClientWriteFail               :   return "Client write operation failed.";
        case RcfError_ClientConnectFail             :   return "Client connect operation failed.";
        case RcfError_Socket                        :   return "Socket error.";
        case RcfError_FnId                          :   return "Invalid function id. Function id: %1";
        case RcfError_UnknownInterface              :   return "Unknown object interface. Interface: %1.";
        case RcfError_NoEndpoint                    :   return "No endpoint.";
        case RcfError_TransportCreation             :   return "Failed to create transport.";
        case RcfError_FilterCount                   :   return "Invalid number of filters. Requested: %1. Max allowed: %2.";
        case RcfError_FilterMessage                 :   return "Failed to filter message.";
        case RcfError_UnfilterMessage               :   return "Failed to unfilter message.";
        case RcfError_SspiCredentials               :   return "SSPI credentials failure.";
        case RcfError_SspiEncrypt                   :   return "SSPI encryption failure.";
        case RcfError_SspiDecrypt                   :   return "SSPI decryption failure.";
        case RcfError_SspiImpersonation             :   return "SSPI impersonation failure.";
        case RcfError_SocketClose                   :   return "Failed to close socket.";
        case RcfError_ZlibDeflate                   :   return "Zlib compression error.";
        case RcfError_ZlibInflate                   :   return "Zlib decompression error.";
        case RcfError_Zlib                          :   return "Zlib error.";
        case RcfError_UnknownSerializationProtocol  :   return "Unknown serialization protocol. Protocol: %1.";
        case SfError_NoCtor                         :   return "Construction not supported for this type.";
        case SfError_RefMismatch                    :   return "Can't deserialize a reference into a non-reference object.";
        case SfError_DataFormat                     :   return "Input data format error.";
        case SfError_ReadFailure                    :   return "Failed to read data from underlying stream.";
        case SfError_WriteFailure                   :   return "Failed to write data to underlying stream.";
        case SfError_BaseDerivedRegistration        :   return "Base/derived pair not registered. Base: %1. Derived: %2.";
        case SfError_TypeRegistration               :   return "Type not registered. Type: %1.";
        case RcfError_NonStdException               :   return "Non std::exception-derived exception was thrown.";
        case RcfError_SocketBind                    :   return "Failed to bind socket to port (port already in use?). Network interface: %1. Port: %2.";
        case RcfError_Decoding                      :   return "Decoding error.";
        case RcfError_Encoding                      :   return "Encoding error.";
        case RcfError_TokenRequestFailed            :   return "No tokens available.";
        case RcfError_ObjectFactoryNotFound         :   return "Object factory not found.";
        case RcfError_PortInUse                     :   return "Port already in use. Network interface: %1. Port: %2.";
        case RcfError_DynamicObjectNotFound         :   return "Server-side object for given token not found. Token id: %1.";
        case RcfError_VersionMismatch               :   return "Version mismatch.";
        case RcfError_SslCertVerification           :   return "SSL certificate verification failure.";
        case RcfError_FiltersLocked                 :   return "Filters locked.";
        case RcfError_Pipe                          :   return "Pipe error.";
        case RcfError_AnySerializerNotFound         :   return "boost::any serializer not registered for the given type. Type: %1.";
        case RcfError_ConnectionLimitExceeded       :   return "The server has reached its incoming connection limit.";
        case RcfError_DeserializationNullPointer    :   return "Null pointer deserialization error.";
        case RcfError_PipeNameTooLong               :   return "Pipe name too long. Pipe name: %1. Max length: %2.";
        case RcfError_PingBack                      :   return "Received ping back message from peer.";
        case RcfError_NoPingBackService             :   return "A ping back service is not available on the server.";
        case RcfError_NoDownload                    :   return "The specified download does not exist.";
        case RcfError_FileOffset                    :   return "The specified file offset is invalid.";
        case RcfError_NoUpload                      :   return "The specified upload does not exist.";            
        case RcfError_FileOpen                      :   return "Failed to open file. Path: %1.";
        case RcfError_FileRead                      :   return "Failed to read from file. Path: %1. Offset: %2.";
        case RcfError_FileWrite                     :   return "Failed to write to file.";
        case RcfError_UploadFailed                  :   return "Upload failed to complete.";
        case RcfError_UploadInProgress              :   return "Upload still in progress.";
        case RcfError_ConcurrentUpload              :   return "Cannot upload on several connections simultaneously.";
        case RcfError_UploadFileSize                :   return "File upload exceeding size limit.";
        case RcfError_AccessDenied                  :   return "Access denied.";
        case RcfError_PingBackTimeout               :   return "Failed to receive pingbacks from server. Expected pingback interval (ms): %1.";
        case RcfError_AllThreadsBusy                :   return "All server threads are busy.";
        case RcfError_UnsupportedRuntimeVersion     :   return "Unsupported RCF runtime version. Requested version: %1. Max supported version: %2.";
        case RcfError_FdSetSize                     :   return "FD_SETSIZE limit exceeded. Maximum: %1.";
        case RcfError_DnsLookup                     :   return "DNS lookup of network address failed, for name \"%1\".";
        case RcfError_SspiHandshakeExtraData        :   return "SSPI handshake protocol error (extra data).";
        case RcfError_ProtobufWrite                 :   return "Failed to serialize Protocol Buffer object. Type: %1.";
        case RcfError_ProtobufRead                  :   return "Failed to deserialize Protocol Buffer object. Type: %1.";
        case RcfError_ExtractSlice                  :   return "Failed to read from marshaling buffer. Position: %1. Length: %2. Marshaling buffer size: %3.";
        case RcfError_ServerStubExpired             :   return "Server stub no longer available";
        case RcfError_VariantDeserialization        :   return "Failed to deserialize variant object. Variant index: %1. Variant size: %2.";
        case RcfError_SspiAuthFailServer            :   return "Server-side SSPI authorization failed.";
        case RcfError_SspiAuthFailClient            :   return "Client-side SSPI authorization failed.";
        case RcfError_Win32ApiError                 :   return "The Win32 function %1 failed. Error: %3. Error code: %2.";
        case RcfError_SspiLengthField               :   return "SSPI invalid format. Packet length exceeds total message length. Packet length: %1. Message length: %2.";
        case RcfError_DownloadFailed                :   return "Download failed to complete. %1";
        case RcfError_FileSeek                      :   return "Failed to seek in file. File: %1. Seek offset: %2.";
        case RcfError_DownloadCancelled             :   return "Download canceled by server.";
        case RcfError_ParseSockAddr                 :   return "Unable to parse socket address structure.";
        case RcfError_GetSockName                   :   return "Unable to retrieve local address information from socket.";
        case RcfError_ProtobufWriteSize             :   return "Unable to determine serialized size of Protocol Buffers object. ByteSize() returned 0. Type: %1.";
        case RcfError_ProtobufWriteInit             :   return "Failed to serialize Protocol Buffers object. Not all required fields were initialized. Type: %1.";
        case RcfError_ArraySizeMismatch             :   return "Array size mismatch while deserializing. Array size: %1. Array size from archive: %2.";
        case RcfError_WcharSizeMismatch             :   return "Unicode character size mismatch. Native wchar_t size: %1. wchar_t size from archive: %2.";
        

        // Errors that are no longer in use.
        case RcfError_StubAssignment                :   return "Incompatible stub assignment.";
        case RcfError_SspiAuthFail                  :   return "SSPI authorization failed.";
        case RcfError_UnknownSubscriber             :   return "Unknown subscriber.";
        case RcfError_Filter                        :   return "Filter error.";
        case RcfError_NotConnected                  :   return "Send operation attempted without connecting.";
        case RcfError_InvalidErrorMessage           :   return "Invalid error message from server.";

        default                                     :   return "No available error message.";
        }
    }

    bool shouldDisconnectOnRemoteError(const Error & err)
    {
        switch (err.getErrorId())
        {
        case RcfError_ConnectionLimitExceeded:
        case RcfError_ServerMessageLength:
        case RcfError_ZlibDeflate:
        case RcfError_ZlibInflate:
            return true;

        default:
            return false;
        }
    }

    std::string getOsErrorString(int osError)
    {
        return Platform::OS::GetErrorString(osError);
    }

    std::string getSubSystemName(int subSystem)
    {
        switch (subSystem)
        {
        case RcfSubsystem_None                      :   return "No sub system.";
        case RcfSubsystem_Os                        :   return "Operating system";
        case RcfSubsystem_Zlib                      :   return "Zlib";
        case RcfSubsystem_OpenSsl                   :   return "OpenSSL";
        case RcfSubsystem_Asio                      :   return "Boost.Asio";
        default                                     :   return "No available sub system name.";
        }
    }

    // Exception

    Exception::Exception() :
        std::runtime_error(""),
        mWhat(),
        mError(RcfError_Ok),
        mSubSystemError(RCF_DEFAULT_INIT),
        mSubSystem(RCF_DEFAULT_INIT)
    {}

    Exception::Exception(
        const std::string &     what, 
        const std::string &     context) :
            std::runtime_error(""),
            mWhat(what),
            mContext(context),
            mError(RcfError_Unspecified),
            mSubSystemError(RCF_DEFAULT_INIT),
            mSubSystem(RCF_DEFAULT_INIT)
    {}

    Exception::Exception(
        Error                   error,
        const std::string &     what,
        const std::string &     context) :
            std::runtime_error(""),
            mWhat(what),
            mContext(context),
            mError(error),
            mSubSystemError(RCF_DEFAULT_INIT),
            mSubSystem(RCF_DEFAULT_INIT)
    {}

    Exception::Exception(
        Error                   error,
        int                     subSystemError,
        int                     subSystem,
        const std::string &     what,
        const std::string &     context) :
            std::runtime_error(""),
            mWhat(what),
            mContext(context),
            mError(error),
            mSubSystemError(subSystemError),
            mSubSystem(subSystem)
    {}

    Exception::~Exception() throw()
    {}

    bool Exception::good() const
    {
        return mError.getErrorId() == RcfError_Ok;
    }

    bool Exception::bad() const
    {
        return !good();
    }

    const char *Exception::what() const throw()
    {
        mTranslatedWhat = translate();
        return mTranslatedWhat.c_str();
    }

    int Exception::getErrorId() const
    {
        return mError.getErrorId();
    }

    std::string Exception::getErrorString() const
    {
        if (mError.getErrorId() >= RcfError_User)
        {
            std::ostringstream os;
            os << "Non-RCF error (" << mError.getErrorId() << ")";
            std::string w = getWhat();
            if (!w.empty())
            {
                os << ": " << w;
            }
            else
            {
                os << ".";
            }
            return os.str();
        }
        else if (mSubSystem == RcfSubsystem_Os)
        {
            std::ostringstream os;
            os << mError.getErrorString();

            if (mSubSystem == RcfSubsystem_Os)
            {
                os << " OS: " << mSubSystemError << " - " << getOsErrorString(mSubSystemError);
            }
            return os.str();
        }
        else
        {
            return mError.getErrorString();
        }
    }

    const Error & Exception::getError() const
    {
        return mError;
    }

    int Exception::getSubSystemError() const
    {
        return mSubSystemError;
    }

    int Exception::getSubSystem() const
    {
        return mSubSystem;
    }

    std::string Exception::getSubSystemName() const
    {
        return RCF::getSubSystemName(mSubSystem);
    }

    void Exception::setContext(const std::string &context)
    {
        mContext = context;
    }

    std::string Exception::getContext() const
    {
        return mContext;
    }

    void Exception::setWhat(const std::string &what)
    {
        mWhat = what;
    }

    std::string Exception::getWhat() const
    {
        return mWhat;
    }

    std::string Exception::translate() const
    {
        std::string osErr;
        if (mSubSystem == RcfSubsystem_Os)
        {
            osErr = getOsErrorString(mSubSystemError);
        }

        int errorId = mError.getErrorId();

        std::ostringstream os;
        os
            << "[" << errorId << ": " << getErrorString() << "]"
            << "[" << mSubSystem << ": " << RCF::getSubSystemName(mSubSystem) << "]"
            << "[" << mSubSystemError << ": " << osErr << "]"
            << "[What: " << mWhat << "]"
            << "[Context: " << mContext << "]";
        return os.str();
    }

    void Exception::throwSelf() const
    {
        // If your code traps here, check that you have overridden throwSelf() in 
        // all classes derived from Exception.

        RCF_ASSERT(
            typeid(*this) == typeid(Exception))
            (typeid(*this));

        RCF_THROW((*this));
    }

    // RemoteException

    RemoteException::RemoteException()
    {}

    RemoteException::RemoteException(
        Error                   remoteError,
        const std::string &     remoteWhat,
        const std::string &     remoteContext,
        const std::string &     remoteExceptionType) :
            Exception(
                remoteError,
                remoteWhat,
                remoteContext),
                mRemoteExceptionType(remoteExceptionType)
    {}

    RemoteException::RemoteException(
        Error                   remoteError,
        int                     remoteSubSystemError,
        int                     remoteSubSystem,
        const std::string &     remoteWhat,
        const std::string &     remoteContext,
        const std::string &     remoteExceptionType) :
            Exception(
                remoteError,
                remoteSubSystemError,
                remoteSubSystem,
                remoteWhat,
                remoteContext),
                mRemoteExceptionType(remoteExceptionType)
    {}

    RemoteException::~RemoteException() throw()
    {}

    const char *RemoteException::what() const throw()
    {
        mTranslatedWhat = "[Remote]" + translate();
        return mTranslatedWhat.c_str();
    }

    std::string RemoteException::getRemoteExceptionType() const
    {
        return mRemoteExceptionType;
    }

    std::auto_ptr<Exception> RemoteException::clone() const
    {
        // If your code traps here, check that you have overridden clone() in 
        // all classes derived from RemoteException.

        RCF_ASSERT(
            typeid(*this) == typeid(RemoteException))
            (typeid(*this));

        return std::auto_ptr<Exception>(
            new RemoteException(*this));
    }

    void RemoteException::throwSelf() const
    {
        // If your code traps here, check that you have overridden throwSelf() in 
        // all classes derived from RemoteException.

        RCF_ASSERT(
            typeid(*this) == typeid(RemoteException))
            (typeid(*this));

        RCF_THROW((*this));
    }

    // VersioningException

    VersioningException::VersioningException(
        boost::uint32_t runtimeVersion, 
        boost::uint32_t archiveVersion) :
            RemoteException(_RcfError_VersionMismatch()),
            mRuntimeVersion(runtimeVersion),
            mArchiveVersion(archiveVersion)
    {}

    VersioningException::~VersioningException() throw()
    {}

    boost::uint32_t VersioningException::getRuntimeVersion() const
    {
        return mRuntimeVersion;
    }

    boost::uint32_t VersioningException::getArchiveVersion() const
    {
        return mArchiveVersion;
    }

#ifdef RCF_USE_SF_SERIALIZATION

    void RemoteException::serialize(SF::Archive &ar)
    {
        if (ar.getRuntimeVersion() <= 5)
        {
            int errorId = mError.getErrorId();

            ar
                & mWhat
                & mContext
                & errorId
                & mSubSystemError
                & mSubSystem
                & mRemoteExceptionType;

            if (ar.isRead())
            {
                mError.setErrorId(errorId);
            }
        }
        else
        {
            ar 
                & mWhat 
                & mContext 
                & mError 
                & mSubSystemError 
                & mSubSystem 
                & mRemoteExceptionType;
        }
    }

#endif

    int getRuntimeVersionOfThisRemoteCall()
    {
        int runtimeVersion = 0;
        RcfSession * pRcfSession = getCurrentRcfSessionPtr();
        ClientStub * pClientStub = getCurrentClientStubPtr();
        if (pRcfSession)
        {
            runtimeVersion = pRcfSession->getRuntimeVersion();
        }
        else if (pClientStub)
        {
            runtimeVersion = pClientStub->getRuntimeVersion();
        }
        else
        {
            // This function must be called from within the client-side
            // or server-side portion of a remote call.
            RCF_ASSERT(0);
            runtimeVersion = getDefaultRuntimeVersion();
        }

        return runtimeVersion;
    }

} // namespace RCF
