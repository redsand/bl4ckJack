
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

#ifndef INCLUDE_RCF_ASIOSERVERTRANSPORT_HPP
#define INCLUDE_RCF_ASIOSERVERTRANSPORT_HPP

#include <set>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#include <boost/asio/buffer.hpp>

#include <RCF/Export.hpp>
#include <RCF/IpAddress.hpp>
#include <RCF/IpServerTransport.hpp>
#include <RCF/ServerTransport.hpp>
#include <RCF/Service.hpp>
#include <RCF/ThreadLibrary.hpp>

namespace boost {
    namespace asio {
        class io_service;
    }
    namespace system {
        class error_code;
    }
}

namespace RCF {

    class RcfServer;
    class TcpClientTransport;
    class AsioSessionState;
    class AsioAcceptor;
    class AsioDeadlineTimer;
    class AsioServerTransport;

    typedef boost::asio::io_service    AsioIoService;

    typedef boost::shared_ptr<AsioIoService>            AsioIoServicePtr;
    typedef boost::shared_ptr<AsioAcceptor>             AsioAcceptorPtr;
    typedef boost::shared_ptr<AsioDeadlineTimer>        AsioDeadlineTimerPtr;
    typedef boost::shared_ptr<AsioSessionState>         AsioSessionStatePtr;
    typedef boost::weak_ptr<AsioSessionState>           AsioSessionStateWeakPtr;

    class AsioAcceptor
    {
    public:
        virtual ~AsioAcceptor()
        {}
    };

    class RCF_EXPORT AsioServerTransport :
        public I_ServerTransport,
        public I_ServerTransportEx,
        public I_Service
    {
    private:

        // Needs to call open().
        friend class TcpAsioTransportFactory;

        typedef boost::weak_ptr<I_Session>              SessionWeakPtr;

        AsioSessionStatePtr createSessionState();
        
        void                notifyClose(
                                AsioSessionStateWeakPtr sessionStateWeakPtr);

        void                closeSessionState(
                                AsioSessionStateWeakPtr sessionStateWeakPtr);

        

        // I_ServerTransportEx implementation
        ClientTransportAutoPtr  
                            createClientTransport(
                                const I_Endpoint &endpoint);

        SessionPtr          createServerSession(
                                ClientTransportAutoPtr & clientTransportAutoPtr, 
                                StubEntryPtr stubEntryPtr,
                                bool keepClientConnection);

        ClientTransportAutoPtr  
                            createClientTransport(
                                SessionPtr sessionPtr);

        bool                reflect(
                                const SessionPtr &sessionPtr1, 
                                const SessionPtr &sessionPtr2);

        bool                isConnected(const SessionPtr &sessionPtr);

        // I_Service implementation
        void                open();
        void                close();
        void                stop();

        void                onServiceAdded(     RcfServer & server);
        void                onServiceRemoved(   RcfServer & server);

    protected:

        void                onServerStart(      RcfServer & server);
        void                onServerStop(       RcfServer & server);
        void                setServer(          RcfServer & server);

        void                startAccepting();

    private:

        RcfServer &         getServer();
        RcfServer &         getSessionManager();

    private:

        void                registerSession(AsioSessionStateWeakPtr session);
        void                unregisterSession(AsioSessionStateWeakPtr session);
        void                cancelOutstandingIo();

        friend class AsioSessionState;
        friend class FilterAdapter;

    protected:

        AsioServerTransport();
        ~AsioServerTransport();
        
        AsioIoService *                 mpIoService;
        AsioAcceptorPtr                 mAcceptorPtr;

    private:
        
        bool                            mOpen;
        bool                            mInterrupt;
        volatile bool                   mStopFlag;
        RcfServer *                     mpServer;

        Mutex                               mSessionsMutex;
        std::set<AsioSessionStateWeakPtr>   mSessions;

    private:

        virtual AsioSessionStatePtr implCreateSessionState() = 0;
        virtual void implOpen() = 0;

        virtual ClientTransportAutoPtr implCreateClientTransport(
            const I_Endpoint &endpoint) = 0;

    public:

        AsioAcceptorPtr getAcceptorPtr();

        AsioIoService & getIoService();
    };

    class ReadHandler
    {
    public:
        ReadHandler(const ReadHandler & rhs);
        ReadHandler(AsioSessionState & sessionState);
        void operator()(boost::system::error_code err, std::size_t bytes);
        void * allocate(std::size_t size);
        AsioSessionState & mSessionState;
    };

    class WriteHandler
    {
    public:
        WriteHandler(const WriteHandler & rhs);
        WriteHandler(AsioSessionState & sessionState);
        void operator()(boost::system::error_code err, std::size_t bytes);
        void * allocate(std::size_t size);
        AsioSessionState & mSessionState;
    };

    void *  asio_handler_allocate(std::size_t size, ReadHandler * pHandler);
    void    asio_handler_deallocate(void * pointer, std::size_t size, ReadHandler * pHandler);
    void *  asio_handler_allocate(std::size_t size, WriteHandler * pHandler);
    void    asio_handler_deallocate(void * pointer, std::size_t size, WriteHandler * pHandler);

    // This adapter around a std::vector prevents asio from making a deep copy
    // of the buffer list, when sending multiple buffers. The deep copy would
    // involve making memory allocations.
    class AsioBuffers
    {
    public:

        typedef std::vector<boost::asio::const_buffer>  BufferVec;
        typedef boost::shared_ptr<BufferVec>            BufferVecPtr;

        typedef boost::asio::const_buffer               value_type;
        typedef BufferVec::const_iterator               const_iterator;

        AsioBuffers()
        {
            mVecPtr.reset( new std::vector<boost::asio::const_buffer>() );
        }

        const_iterator begin() const
        {
            return mVecPtr->begin();
        }

        const_iterator end() const
        {
            return mVecPtr->end();
        }

        BufferVecPtr mVecPtr;
    };


    class AsioSessionState :
        public I_SessionState,
        boost::noncopyable
    {
    public:

        friend class ReadHandler;
        friend class WriteHandler;


        typedef boost::weak_ptr<AsioSessionState>       AsioSessionStateWeakPtr;
        typedef boost::shared_ptr<AsioSessionState>     AsioSessionStatePtr;

        AsioSessionState(
            AsioServerTransport &transport,
            AsioIoService & ioService);

        virtual ~AsioSessionState();

        AsioSessionStatePtr sharedFromThis();

        void            setSessionPtr(SessionPtr sessionPtr);
        SessionPtr      getSessionPtr();

        void            close();
        void            invokeAsyncAccept();

        int             getNativeHandle() const;

    protected:

        void            onReadCompletion(
                            boost::system::error_code error, 
                            size_t bytesTransferred);

        void            onWriteCompletion(
                            boost::system::error_code error, 
                            size_t bytesTransferred);

        std::vector<char>       mReadHandlerBuffer;
        std::vector<char>       mWriteHandlerBuffer;
        AsioSessionStatePtr     mThisPtr;

    private:

        void            read(
                            const ByteBuffer &byteBuffer, 
                            std::size_t bytesRequested);

        void            write(
                            const std::vector<ByteBuffer> &byteBuffers);

        

        void            setTransportFilters(
                            const std::vector<FilterPtr> &filters);

        void            getTransportFilters(
                            std::vector<FilterPtr> &filters);

        void            invokeAsyncRead();
        void            invokeAsyncWrite();
        void            onAccept(const boost::system::error_code& error);

        void            onReadWrite(
                            size_t bytesTransferred);

        void            sendServerError(int error);

        void            onReflectedReadWrite(
                            const boost::system::error_code& error, 
                            size_t bytesTransferred);

        ReallocBuffer &             getReadBuffer();
        ReallocBuffer &             getUniqueReadBuffer();
        ByteBuffer                  getReadByteBuffer() const;

        ReallocBuffer &             getReadBufferSecondary();
        ReallocBuffer &             getUniqueReadBufferSecondary();
        ByteBuffer                  getReadByteBufferSecondary() const;

        // TODO: too many friends
        friend class    AsioServerTransport;
        friend class    TcpAsioSessionState;
        friend class    UnixLocalSessionState;
        friend class    FilterAdapter;

        enum State
        {
            Ready,
            Accepting,
            ReadingDataCount,
            ReadingData,
            WritingData
        };

        State                       mState;
        bool                        mIssueZeroByteRead;
        std::size_t                 mReadBufferRemaining;
        std::size_t                 mWriteBufferRemaining;
        SessionPtr                  mSessionPtr;
        std::vector<FilterPtr>      mTransportFilters;
        AsioIoService &             mIoService;

        AsioServerTransport &       mTransport;

        std::vector<ByteBuffer>     mWriteByteBuffers;
        std::vector<ByteBuffer>     mSlicedWriteByteBuffers;

        ReallocBufferPtr            mReadBufferPtr;
        ReallocBufferPtr            mReadBufferSecondaryPtr;

        ByteBuffer                  mReadByteBuffer;
        ByteBuffer                  mTempByteBuffer;

        FilterPtr                   mFilterAdapterPtr;

        Mutex                       mMutex;
        bool                        mHasBeenClosed;
        bool                        mCloseAfterWrite;
        AsioSessionStateWeakPtr     mReflecteeWeakPtr;
        AsioSessionStatePtr         mReflecteePtr;
        bool                        mReflecting;

        AsioSessionStateWeakPtr     mWeakThisPtr;

        AsioBuffers                 mBufs;

        // I_SessionState

    private:
        
        void                    postRead();
        ByteBuffer              getReadByteBuffer();
        void                    postWrite(std::vector<ByteBuffer> &byteBuffers);
        void                    postClose();
        I_ServerTransport &     getServerTransport();
        const I_RemoteAddress & getRemoteAddress();

    private:

        virtual const I_RemoteAddress & implGetRemoteAddress() = 0;
        virtual void implRead(char * buffer, std::size_t bufferLen) = 0;
        virtual void implWrite(const std::vector<ByteBuffer> & buffers) = 0;
        virtual void implWrite(AsioSessionState &toBeNotified, const char * buffer, std::size_t bufferLen) = 0;
        virtual void implAccept() = 0;
        virtual bool implOnAccept() = 0;
        virtual int implGetNative() const = 0;
        virtual boost::function0<void> implGetCloseFunctor() = 0;
        virtual void implClose() = 0;
        virtual void implTransferNativeFrom(I_ClientTransport & clientTransport) = 0;
        virtual ClientTransportAutoPtr implCreateClientTransport() = 0;
    };

} // namespace RCF


#endif // ! INCLUDE_RCF_ASIOSERVERTRANSPORT_HPP
