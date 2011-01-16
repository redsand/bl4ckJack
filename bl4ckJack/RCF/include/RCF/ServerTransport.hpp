
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

#ifndef INCLUDE_RCF_SERVERTRANSPORT_HPP
#define INCLUDE_RCF_SERVERTRANSPORT_HPP

#include <memory>
#include <string>
#include <vector>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <RCF/ByteBuffer.hpp>
#include <RCF/Export.hpp>

namespace RCF {

    class Filter;
    class I_Endpoint;
    class I_ClientTransport;
    class I_SessionState;
    class StubEntry;

    typedef boost::shared_ptr<Filter>               FilterPtr;
    typedef std::auto_ptr<I_ClientTransport>        ClientTransportAutoPtr;
    typedef boost::shared_ptr<StubEntry>            StubEntryPtr;

    class I_ServerTransport;
    typedef boost::shared_ptr<I_ServerTransport>    ServerTransportPtr;
    typedef std::auto_ptr<I_ServerTransport>        ServerTransportAutoPtr;

    class I_RemoteAddress
    {
    public:
        virtual ~I_RemoteAddress()
        {}

        virtual std::string string() const
        {
            return "";
        }
    };

    class NoRemoteAddress : public I_RemoteAddress
    {};
   
    class I_SessionState : public boost::enable_shared_from_this<I_SessionState>
    {
    public:

        I_SessionState();
        virtual ~I_SessionState() {}

        virtual void        postRead() = 0;
        virtual ByteBuffer  getReadByteBuffer() = 0;
        virtual void        postWrite(std::vector<ByteBuffer> &byteBuffers) = 0;
        virtual void        postClose() = 0;
       
        virtual I_ServerTransport & 
                            getServerTransport() = 0;

        virtual const I_RemoteAddress & 
                            getRemoteAddress() = 0;

        virtual void        setTransportFilters(const std::vector<FilterPtr> &filters) = 0;
        virtual void        getTransportFilters(std::vector<FilterPtr> &filters) = 0;
        void                setEnableReconnect(bool enableReconnect);
        bool                getEnableReconnect();

    protected:
        bool mEnableReconnect;
    };

    typedef boost::shared_ptr<I_SessionState> SessionStatePtr;

    class RcfSession;
    typedef boost::shared_ptr<RcfSession> RcfSessionPtr;
    
    typedef RcfSession I_Session;
    typedef RcfSessionPtr SessionPtr;
    class ThreadPool;
    typedef boost::shared_ptr<ThreadPool> ThreadPoolPtr;

    /// Base class of all server transport services.
    class RCF_EXPORT I_ServerTransport
    {
    public:
        I_ServerTransport();

        virtual ~I_ServerTransport() {}

        virtual ServerTransportPtr 
                        clone() = 0;

        I_ServerTransport & setMaxMessageLength(std::size_t maxMessageLength);
        std::size_t         getMaxMessageLength() const;

        std::size_t         getConnectionLimit() const;
        I_ServerTransport & setConnectionLimit(std::size_t connectionLimit);

        I_ServerTransport & setThreadPool(ThreadPoolPtr threadPoolPtr);
        
    private:

        mutable ReadWriteMutex      mReadWriteMutex;
        std::size_t                 mMaxMessageLength;
        std::size_t                 mConnectionLimit;       
    };

    class I_ServerTransportEx
    {
    public:

        virtual ~I_ServerTransportEx() {}

        virtual ClientTransportAutoPtr 
            createClientTransport(
                const I_Endpoint &endpoint) = 0;
       
        virtual SessionPtr 
            createServerSession(
                ClientTransportAutoPtr & clientTransportAutoPtr,
                StubEntryPtr stubEntryPtr,
                bool keepClientConnection = true) = 0;

        virtual ClientTransportAutoPtr 
            createClientTransport(
                SessionPtr sessionPtr) = 0;
       
        virtual bool 
            reflect(
                const SessionPtr &sessionPtr1,
                const SessionPtr &sessionPtr2) = 0;
       
        virtual bool 
            isConnected(
                const SessionPtr &sessionPtr) = 0;
    };   

    RCF_EXPORT std::size_t  getDefaultMaxMessageLength();

    RCF_EXPORT void         setDefaultMaxMessageLength(
                                std::size_t maxMessageLength);

} // namespace RCF

#endif // ! INCLUDE_RCF_SERVERTRANSPORT_HPP
