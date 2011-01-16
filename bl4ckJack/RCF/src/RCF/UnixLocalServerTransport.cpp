
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

#include <RCF/UnixLocalServerTransport.hpp>

#include <RCF/Asio.hpp>
#include <RCF/UnixLocalClientTransport.hpp>
#include <RCF/UnixLocalEndpoint.hpp>

#include <RCF/util/Platform/OS/BsdSockets.hpp>

// std::remove
#include <cstdio>

namespace RCF {

    using boost::asio::local::stream_protocol;

    class UnixLocalAcceptor : public AsioAcceptor
    {
    public:
        UnixLocalAcceptor(AsioIoService & ioService, const std::string & fileName) : 
            mFileName(fileName),
            mAcceptor(ioService, stream_protocol::endpoint(fileName))
            
        {}

        ~UnixLocalAcceptor()
        {
            mAcceptor.close();

            // Delete the underlying file as well.
            int ret = std::remove(mFileName.c_str());
            int err = Platform::OS::BsdSockets::GetLastError();

            if (ret != 0)
            {
                // Couldn't delete it, not a whole lot we can do about it.
                RCF_LOG_1()(mFileName)(err)(Platform::OS::GetErrorString(err)) 
                    << "Failed to delete underlying file of UNIX domain socket.";
            }
        }

        std::string mFileName;
        stream_protocol::acceptor mAcceptor;
    };

    typedef stream_protocol::socket                 UnixLocalSocket;
    typedef boost::shared_ptr<UnixLocalSocket>      UnixLocalSocketPtr;

    // UnixLocalSessionState

    class UnixLocalSessionState : public AsioSessionState
    {
    public:
        UnixLocalSessionState(
            UnixLocalServerTransport & transport,
            AsioIoService & ioService) :
                AsioSessionState(transport, ioService),
                mSocketPtr(new UnixLocalSocket(ioService))
        {}

        const I_RemoteAddress & implGetRemoteAddress()
        {
            return mRemoteAddress;
        }

        void implRead(char * buffer, std::size_t bufferLen)
        {
            RCF_LOG_4()(bufferLen) 
                << "UnixLocalSessionState - calling async_read_some().";

            mThisPtr = sharedFromThis();

            mSocketPtr->async_read_some(
                boost::asio::buffer( buffer, bufferLen),
                ReadHandler(*this));
        }

        void implWrite(const std::vector<ByteBuffer> & buffers)
        {
            RCF_LOG_4()(RCF::lengthByteBuffers(buffers))
                << "UnixLocalSessionState - calling async_write_some().";

            mThisPtr = sharedFromThis();

            mBufs.mVecPtr->resize(0);
            for (std::size_t i=0; i<buffers.size(); ++i)
            {
                ByteBuffer buffer = buffers[i];

                mBufs.mVecPtr->push_back( 
                    boost::asio::buffer(buffer.getPtr(), buffer.getLength()) );
            }

            mSocketPtr->async_write_some(
                mBufs,
                WriteHandler(*this));
        }

        void implWrite(AsioSessionState &toBeNotified, const char * buffer, std::size_t bufferLen)
        {
            boost::asio::async_write(
                *mSocketPtr,
                boost::asio::buffer(buffer, bufferLen),
                boost::bind(
                    &AsioSessionState::onWriteCompletion,
                    toBeNotified.sharedFromThis(),
                    boost::asio::placeholders::error,
                    boost::asio::placeholders::bytes_transferred));
        }

        void implAccept()
        {
            RCF_LOG_4()<< "UnixLocalSessionState - calling async_accept().";

            UnixLocalAcceptor & unixLocalAcceptor = 
                static_cast<UnixLocalAcceptor &>(*mTransport.getAcceptorPtr());

            unixLocalAcceptor.mAcceptor.async_accept(
                *mSocketPtr,
                boost::bind(
                    &AsioSessionState::onAccept,
                    sharedFromThis(),
                    boost::asio::placeholders::error));
        }

        bool implOnAccept()
        {
            return true;
        }

        int implGetNative() const
        {
            return mSocketPtr->native();
        }

        void implSetNative(int fd)
        {
            mSocketPtr->assign(
                stream_protocol(),
                fd);
        }

        boost::function0<void> implGetCloseFunctor()
        {
            return boost::bind(
                &UnixLocalSessionState::closeSocket,
                mSocketPtr);
        }

        void implClose()
        {
            mSocketPtr->close();
        }

        ClientTransportAutoPtr implCreateClientTransport()
        {
            int fd = implGetNative();

            std::auto_ptr<UnixLocalClientTransport> unixLocalClientTransport(
                new UnixLocalClientTransport(fd, mRemoteFileName));

            return ClientTransportAutoPtr(unixLocalClientTransport.release());
        }

        void implTransferNativeFrom(I_ClientTransport & clientTransport)
        {
            UnixLocalClientTransport *pUnixLocalClientTransport =
                dynamic_cast<UnixLocalClientTransport *>(&clientTransport);

            if (pUnixLocalClientTransport == NULL)
            {
                Exception e("incompatible client transport");
                RCF_THROW(e)(typeid(clientTransport));
            }

            UnixLocalClientTransport & unixLocalClientTransport = *pUnixLocalClientTransport;

            // TODO: exception safety
            mSocketPtr->assign(
                stream_protocol(),
                unixLocalClientTransport.releaseFd());

            mRemoteFileName = unixLocalClientTransport.getPipeName();
        }
        static void closeSocket(UnixLocalSocketPtr socketPtr)
        {
            socketPtr->close();
        }

    private:
        UnixLocalSocketPtr          mSocketPtr;
        std::string                 mRemoteFileName;
        NoRemoteAddress             mRemoteAddress;
    };

    // UnixLocalServerTransport

    std::string UnixLocalServerTransport::getPipeName() const
    {
        return mFileName;
    }

    UnixLocalServerTransport::UnixLocalServerTransport(
        const std::string &fileName) :
            mFileName(fileName)
    {}

    ServerTransportPtr UnixLocalServerTransport::clone()
    {
        return ServerTransportPtr(new UnixLocalServerTransport(mFileName));
    }

    AsioSessionStatePtr UnixLocalServerTransport::implCreateSessionState()
    {
        return AsioSessionStatePtr( new UnixLocalSessionState(*this, *mpIoService) );
    }

    void UnixLocalServerTransport::implOpen()
    {
    }

    ClientTransportAutoPtr UnixLocalServerTransport::implCreateClientTransport(
        const I_Endpoint &endpoint)
    {
        const UnixLocalEndpoint & unixLocalEndpoint = 
            dynamic_cast<const UnixLocalEndpoint &>(endpoint);

        ClientTransportAutoPtr clientTransportAutoPtr(
            new UnixLocalClientTransport(unixLocalEndpoint.getPipeName()));

        return clientTransportAutoPtr;
    }

    void UnixLocalServerTransport::onServerStart(RcfServer & server)
    {
        AsioServerTransport::onServerStart(server);

        mpIoService = mTaskEntries[0].getThreadPool().getIoService();

        RCF_ASSERT(mAcceptorPtr.get() == NULL);

        if ( !mFileName.empty() )
        {
            boost::shared_ptr<UnixLocalAcceptor> acceptorPtr(
                new UnixLocalAcceptor(*mpIoService, mFileName));

            mAcceptorPtr = acceptorPtr;

            startAccepting();
        }

        RCF_LOG_2()(mFileName) << "UnixLocalServerTransport - listening on local socket.";
    }

    void UnixLocalServerTransport::onServerStop(RcfServer & server)
    {
        AsioServerTransport::onServerStop(server);
    }

} // namespace RCF
