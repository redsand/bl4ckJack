
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

#ifndef INCLUDE_RCF_IDL_HPP
#define INCLUDE_RCF_IDL_HPP

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/int.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

#include <RCF/ClientStub.hpp>
#include <RCF/ClientTransport.hpp>
#include <RCF/Config.hpp>
#include <RCF/Endpoint.hpp>
#include <RCF/Exception.hpp>
#include <RCF/GetInterfaceName.hpp>
#include <RCF/Marshal.hpp>
#include <RCF/RcfClient.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerStub.hpp>
#include <RCF/ThreadLocalData.hpp>
    
#define RCF_BEGIN(InterfaceT, Name) RCF_BEGIN_I0(InterfaceT, Name)

#define RCF_BEGIN_I0(InterfaceT, Name)                                                                  \
    RCF_BEGIN_IMPL_PRELUDE(InterfaceT, Name)                                                            \
    RCF_BEGIN_IMPL_INHERITED_0(InterfaceT, Name)                                                        \
    RCF_BEGIN_IMPL_POSTLUDE(InterfaceT, Name)

#define RCF_BEGIN_I1(InterfaceT, Name, InheritT1)                                                       \
    RCF_BEGIN_IMPL_PRELUDE(InterfaceT, Name)                                                            \
    RCF_BEGIN_IMPL_INHERITED_1(InterfaceT, Name, InheritT1)                                             \
    RCF_BEGIN_IMPL_POSTLUDE(InterfaceT, Name)

#define RCF_BEGIN_I2(InterfaceT, Name, InheritT1, InheritT2)                                            \
    RCF_BEGIN_IMPL_PRELUDE(InterfaceT, Name)                                                            \
    RCF_BEGIN_IMPL_INHERITED_2(InterfaceT, Name, InheritT1, InheritT2)                                  \
    RCF_BEGIN_IMPL_POSTLUDE(InterfaceT, Name)

#define RCF_BEGIN_IMPL_PRELUDE(InterfaceT, Name)                                                        \
                                                                                                        \
    template<typename T>                                                                                \
    class RcfClient;                                                                                    \
                                                                                                        \
    class InterfaceT                                                                                    \
    {                                                                                                   \
    public:                                                                                             \
        typedef RcfClient<InterfaceT> RcfClientT;                                                       \
        static std::string getInterfaceName()                                                           \
        {                                                                                               \
            return std::string(Name) == "" ? #InterfaceT : Name;                                        \
        }                                                                                               \
    };

#define RCF_BEGIN_IMPL_INHERITED_0(InterfaceT, Name)                                                    \
    template<>                                                                                          \
    class RcfClient< InterfaceT > :                                                                     \
        public virtual ::RCF::I_RcfClient                                                               \
    {                                                                                                   \
    private:                                                                                            \
        template<typename DerefPtrT>                                                                    \
        void registerInvokeFunctors(::RCF::InvokeFunctorMap &invokeFunctorMap, DerefPtrT derefPtr)      \
        {                                                                                               \
            ::RCF::registerInvokeFunctors(*this, invokeFunctorMap, derefPtr);                           \
        }                                                                                               \
        void setClientStubPtr(::RCF::ClientStubPtr clientStubPtr)                                       \
        {                                                                                               \
            mClientStubPtr = clientStubPtr;                                                             \
        }

#define RCF_BEGIN_IMPL_INHERITED_1(InterfaceT, Name, InheritT1)                                         \
    template<>                                                                                          \
    class RcfClient< InterfaceT > :                                                                     \
        public virtual ::RCF::I_RcfClient,                                                              \
        public virtual ::RCF::GetInterface<InheritT1>::type                                             \
    {                                                                                                   \
    private:                                                                                            \
        template<typename DerefPtrT>                                                                    \
        void registerInvokeFunctors(::RCF::InvokeFunctorMap &invokeFunctorMap, DerefPtrT derefPtr)      \
        {                                                                                               \
            ::RCF::registerInvokeFunctors(*this, invokeFunctorMap, derefPtr);                           \
            ::RCF::StubAccess().registerParentInvokeFunctors(                                           \
                (InheritT1 *) NULL,                                                                     \
                *this,                                                                                  \
                invokeFunctorMap,                                                                       \
                derefPtr);                                                                              \
        }                                                                                               \
        void setClientStubPtr(::RCF::ClientStubPtr clientStubPtr)                                       \
        {                                                                                               \
            mClientStubPtr = clientStubPtr;                                                             \
            ::RCF::StubAccess().setClientStubPtr( (InheritT1*) 0, *this);                               \
        }

#define RCF_BEGIN_IMPL_INHERITED_2(InterfaceT, Name, InheritT1, InheritT2)                              \
    template<>                                                                                          \
    class RcfClient< InterfaceT > :                                                                     \
        public virtual ::RCF::I_RcfClient,                                                              \
        public virtual ::RCF::GetInterface<InheritT1>::type,                                            \
        public virtual ::RCF::GetInterface<InheritT2>::type                                             \
    {                                                                                                   \
    private:                                                                                            \
        template<typename DerefPtrT>                                                                    \
        void registerInvokeFunctors(::RCF::InvokeFunctorMap &invokeFunctorMap, DerefPtrT derefPtr)      \
        {                                                                                               \
            ::RCF::registerInvokeFunctors(*this, invokeFunctorMap, derefPtr);                           \
                                                                                                        \
            ::RCF::StubAccess().registerParentInvokeFunctors(                                           \
                (InheritT1 *) NULL,                                                                     \
                *this,                                                                                  \
                invokeFunctorMap,                                                                       \
                derefPtr);                                                                              \
                                                                                                        \
            ::RCF::StubAccess().registerParentInvokeFunctors(                                           \
                (InheritT2 *) NULL,                                                                     \
                *this,                                                                                  \
                invokeFunctorMap,                                                                       \
                derefPtr);                                                                              \
        }                                                                                               \
        void setClientStubPtr(::RCF::ClientStubPtr clientStubPtr)                                       \
        {                                                                                               \
            mClientStubPtr = clientStubPtr;                                                             \
            ::RCF::StubAccess().setClientStubPtr( (InheritT1*) 0, *this);                               \
            ::RCF::StubAccess().setClientStubPtr( (InheritT2*) 0, *this);                               \
        }

#define RCF_BEGIN_IMPL_POSTLUDE(InterfaceT, Name)                                                       \
    public:                                                                                             \
                                                                                                        \
        RcfClient()                                                                                     \
        {}                                                                                              \
                                                                                                        \
        template<typename DerefPtrT>                                                                    \
        RcfClient(                                                                                      \
            ::RCF::ServerStubPtr serverStubPtr,                                                         \
            DerefPtrT derefPtr,                                                                         \
            boost::mpl::true_ *)                                                                        \
        {                                                                                               \
            serverStubPtr->registerInvokeFunctors(*this, derefPtr);                                     \
            mServerStubPtr = serverStubPtr;                                                             \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            const ::RCF::I_Endpoint &endpoint)                                                          \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            const std::string &targetName = interfaceName;                                              \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(interfaceName, targetName) );     \
            clientStubPtr->setEndpoint(endpoint);                                                       \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            const ::RCF::I_Endpoint &endpoint,                                                          \
            const std::string &targetName)                                                              \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(interfaceName, targetName) );     \
            clientStubPtr->setEndpoint(endpoint);                                                       \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            ::RCF::ClientTransportAutoPtr clientTransportAutoPtr)                                       \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            const std::string &targetName = interfaceName;                                              \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(interfaceName, targetName) );     \
            clientStubPtr->setTransport(clientTransportAutoPtr);                                        \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            ::RCF::ClientTransportAutoPtr clientTransportAutoPtr,                                       \
            const std::string &targetName)                                                              \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(interfaceName, targetName) );     \
            clientStubPtr->setTransport(clientTransportAutoPtr);                                        \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            const ::RCF::ClientStub &clientStub)                                                        \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            const std::string &targetName = interfaceName;                                              \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(clientStub) );                    \
            clientStubPtr->setInterfaceName(interfaceName);                                             \
            clientStubPtr->setTargetName(targetName);                                                   \
            clientStubPtr->setTargetToken(::RCF::Token());                                              \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            const ::RCF::ClientStub &clientStub,                                                        \
            const std::string &targetName)                                                              \
        {                                                                                               \
            const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);           \
            ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(clientStub) );                    \
            clientStubPtr->setInterfaceName(interfaceName);                                             \
            clientStubPtr->setTargetName(targetName);                                                   \
            clientStubPtr->setTargetToken(::RCF::Token());                                              \
            setClientStubPtr(clientStubPtr);                                                            \
        }                                                                                               \
                                                                                                        \
        RcfClient(                                                                                      \
            const ::RCF::I_RcfClient & rhs)                                                             \
        {                                                                                               \
            if (rhs.getClientStubPtr())                                                                 \
            {                                                                                           \
                const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);       \
                const std::string &targetName = interfaceName;                                          \
                ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(rhs.getClientStub()));        \
                clientStubPtr->setInterfaceName(interfaceName);                                         \
                clientStubPtr->setTargetName(targetName);                                               \
                clientStubPtr->setTargetToken(::RCF::Token());                                          \
                setClientStubPtr(clientStubPtr);                                                        \
            }                                                                                           \
        }                                                                                               \
                                                                                                        \
        ~RcfClient()                                                                                    \
        {                                                                                               \
        }                                                                                               \
                                                                                                        \
        RcfClient &operator=(const RcfClient &rhs)                                                      \
        {                                                                                               \
            if (&rhs != this)                                                                           \
            {                                                                                           \
                if (rhs.mClientStubPtr)                                                                 \
                {                                                                                       \
                    const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);   \
                    const std::string &targetName = interfaceName;                                      \
                    ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(rhs.getClientStub()));    \
                    clientStubPtr->setInterfaceName(interfaceName);                                     \
                    clientStubPtr->setTargetName(targetName);                                           \
                    clientStubPtr->setTargetToken(::RCF::Token());                                      \
                    setClientStubPtr(clientStubPtr);                                                    \
                }                                                                                       \
                else                                                                                    \
                {                                                                                       \
                    RCF_ASSERT(!rhs.mServerStubPtr);                                                    \
                    mClientStubPtr = rhs.mClientStubPtr;                                                \
                }                                                                                       \
            }                                                                                           \
            return *this;                                                                               \
        }                                                                                               \
                                                                                                        \
        RcfClient &operator=(const ::RCF::I_RcfClient &rhs)                                             \
        {                                                                                               \
            if (rhs.getClientStubPtr())                                                                 \
            {                                                                                           \
                const std::string &interfaceName = ::RCF::getInterfaceName( (InterfaceT *) NULL);       \
                const std::string &targetName = interfaceName;                                          \
                ::RCF::ClientStubPtr clientStubPtr( new ::RCF::ClientStub(rhs.getClientStub()));        \
                clientStubPtr->setInterfaceName(interfaceName);                                         \
                clientStubPtr->setTargetName(targetName);                                               \
                clientStubPtr->setTargetToken(::RCF::Token());                                          \
                setClientStubPtr(clientStubPtr);                                                        \
            }                                                                                           \
            else                                                                                        \
            {                                                                                           \
                RCF_ASSERT(!rhs.getServerStubPtr());                                                    \
                mClientStubPtr.reset();                                                                 \
            }                                                                                           \
            return *this;                                                                               \
        }                                                                                               \
                                                                                                        \
        void swap(RcfClient & rhs)                                                                      \
        {                                                                                               \
            ::RCF::ClientStubPtr clientStubPtr = rhs.mClientStubPtr;                                    \
            ::RCF::ServerStubPtr serverStubPtr = rhs.mServerStubPtr;                                    \
                                                                                                        \
            rhs.mClientStubPtr = mClientStubPtr;                                                        \
            rhs.mServerStubPtr = mServerStubPtr;                                                        \
                                                                                                        \
            mClientStubPtr = clientStubPtr;                                                             \
            mServerStubPtr = serverStubPtr;                                                             \
        }                                                                                               \
                                                                                                        \
    public:                                                                                             \
        ::RCF::ClientStub &getClientStub()                                                              \
        {                                                                                               \
            return *mClientStubPtr;                                                                     \
        }                                                                                               \
                                                                                                        \
        const ::RCF::ClientStub &getClientStub() const                                                  \
        {                                                                                               \
            return *mClientStubPtr;                                                                     \
        }                                                                                               \
                                                                                                        \
        ::RCF::ClientStubPtr getClientStubPtr() const                                                   \
        {                                                                                               \
            return mClientStubPtr;                                                                      \
        }                                                                                               \
                                                                                                        \
        ::RCF::ServerStubPtr getServerStubPtr() const                                                   \
        {                                                                                               \
            return mServerStubPtr;                                                                      \
        }                                                                                               \
                                                                                                        \
    private:                                                                                            \
        ::RCF::ServerStub &getServerStub()                                                              \
        {                                                                                               \
            return *mServerStubPtr;                                                                     \
        }                                                                                               \
                                                                                                        \
    public:                                                                                             \
        template<typename Archive>                                                                      \
        void serialize(Archive &ar)                                                                     \
        {                                                                                               \
            ::RCF::StubAccess().serialize(ar, *this);                                                   \
        }                                                                                               \
                                                                                                        \
        template<typename Archive>                                                                      \
        void serialize(Archive &ar, const unsigned int)                                                 \
        {                                                                                               \
            ::RCF::StubAccess().serialize(ar, *this, 0);                                                \
        }                                                                                               \
                                                                                                        \
    private:                                                                                            \
                                                                                                        \
        template<typename N, typename T>                                                                \
        void invoke(                                                                                    \
            const N &,                                                                                  \
            ::RCF::RcfSession &,                                                                        \
            const T &)                                                                                  \
        {                                                                                               \
            ::RCF::Exception e(RCF::_RcfError_FnId(N::value));                                          \
            RCF_THROW(e);                                                                               \
        }                                                                                               \
                                                                                                        \
        ::RCF::ClientStubPtr            mClientStubPtr;                                                 \
        ::RCF::ServerStubPtr            mServerStubPtr;                                                 \
                                                                                                        \
        typedef ::RCF::Void             V;                                                              \
        typedef RcfClient< InterfaceT > ThisT;                                                          \
        typedef ::RCF::Dummy<ThisT>     DummyThisT;                                                     \
                                                                                                        \
        friend class ::RCF::StubAccess;                                                                 \
        friend ::RCF::default_ RCF_make_next_dispatch_id_func(DummyThisT *, ThisT *,...);               \
    public:                                                                                             \
        typedef InterfaceT              Interface;
        


#define RCF_END( InterfaceT )                                                                           \
    };

#define RCF_METHOD_PLACEHOLDER()                                                                        \
    RCF_METHOD_PLACEHOLDER_(RCF_MAKE_UNIQUE_ID(PlaceHolder, V0))

#define RCF_METHOD_PLACEHOLDER_(id)                                                                     \
    public:                                                                                             \
        RCF_MAKE_NEXT_DISPATCH_ID(id);                                                                  \
    private:

#include "RcfMethodGen.hpp"

// RCF_MAKE_UNIQUE_ID

BOOST_STATIC_ASSERT( sizeof(RCF::defined_) != sizeof(RCF::default_));

#define RCF_MAKE_UNIQUE_ID(func, sig)                       RCF_MAKE_UNIQUE_ID_(func, sig, __LINE__)
#define RCF_MAKE_UNIQUE_ID_(func, sig, __LINE__)            RCF_MAKE_UNIQUE_ID__(func, sig, __LINE__)
#define RCF_MAKE_UNIQUE_ID__(func, sig, Line)               rcf_unique_id_##func##_##sig##_##Line

#if RCF_MAX_METHOD_COUNT <= 35

#define RCF_MAKE_NEXT_DISPATCH_ID(next_dispatch_id)                                                                                                                             \
    typedef                                                                                                                                                                     \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 0> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 1> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 2> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 3> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 4> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 5> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 6> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 7> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 8> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 9> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<10> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<11> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<12> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<13> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<14> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<15> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<16> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<17> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<18> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<19> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<20> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<21> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<22> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<23> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<24> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<25> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<26> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<27> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<28> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<29> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<30> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<31> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<32> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<33> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<34> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::int_<35>,                                                                                                                                                       \
    boost::mpl::int_<34> >::type,                                                                                                                                               \
    boost::mpl::int_<33> >::type,                                                                                                                                               \
    boost::mpl::int_<32> >::type,                                                                                                                                               \
    boost::mpl::int_<31> >::type,                                                                                                                                               \
    boost::mpl::int_<30> >::type,                                                                                                                                               \
    boost::mpl::int_<29> >::type,                                                                                                                                               \
    boost::mpl::int_<28> >::type,                                                                                                                                               \
    boost::mpl::int_<27> >::type,                                                                                                                                               \
    boost::mpl::int_<26> >::type,                                                                                                                                               \
    boost::mpl::int_<25> >::type,                                                                                                                                               \
    boost::mpl::int_<24> >::type,                                                                                                                                               \
    boost::mpl::int_<23> >::type,                                                                                                                                               \
    boost::mpl::int_<22> >::type,                                                                                                                                               \
    boost::mpl::int_<21> >::type,                                                                                                                                               \
    boost::mpl::int_<20> >::type,                                                                                                                                               \
    boost::mpl::int_<19> >::type,                                                                                                                                               \
    boost::mpl::int_<18> >::type,                                                                                                                                               \
    boost::mpl::int_<17> >::type,                                                                                                                                               \
    boost::mpl::int_<16> >::type,                                                                                                                                               \
    boost::mpl::int_<15> >::type,                                                                                                                                               \
    boost::mpl::int_<14> >::type,                                                                                                                                               \
    boost::mpl::int_<13> >::type,                                                                                                                                               \
    boost::mpl::int_<12> >::type,                                                                                                                                               \
    boost::mpl::int_<11> >::type,                                                                                                                                               \
    boost::mpl::int_<10> >::type,                                                                                                                                               \
    boost::mpl::int_< 9> >::type,                                                                                                                                               \
    boost::mpl::int_< 8> >::type,                                                                                                                                               \
    boost::mpl::int_< 7> >::type,                                                                                                                                               \
    boost::mpl::int_< 6> >::type,                                                                                                                                               \
    boost::mpl::int_< 5> >::type,                                                                                                                                               \
    boost::mpl::int_< 4> >::type,                                                                                                                                               \
    boost::mpl::int_< 3> >::type,                                                                                                                                               \
    boost::mpl::int_< 2> >::type,                                                                                                                                               \
    boost::mpl::int_< 1> >::type,                                                                                                                                               \
    boost::mpl::int_< 0> >::type next_dispatch_id;                                                                                                                              \
    friend RCF::defined_ RCF_make_next_dispatch_id_func(DummyThisT *, ThisT *, next_dispatch_id *)

#elif RCF_MAX_METHOD_COUNT <= 100

#define RCF_MAKE_NEXT_DISPATCH_ID(next_dispatch_id)                                                                                                                             \
    typedef                                                                                                                                                                     \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 0> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 1> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 2> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 3> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 4> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 5> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 6> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 7> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 8> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_< 9> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<10> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<11> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<12> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<13> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<14> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<15> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<16> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<17> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<18> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<19> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<20> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<21> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<22> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<23> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<24> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<25> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<26> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<27> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<28> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<29> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<30> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<31> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<32> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<33> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<34> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<35> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<36> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<37> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<38> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<39> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<40> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<41> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<42> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<43> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<44> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<45> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<46> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<47> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<48> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<49> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<50> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<51> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<52> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<53> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<54> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<55> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<56> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<57> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<58> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<59> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<60> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<61> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<62> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<63> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<64> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<65> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<66> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<67> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<68> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<69> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<70> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<71> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<72> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<73> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<74> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<75> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<76> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<77> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<78> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<79> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<80> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<81> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<82> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<83> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<84> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<85> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<86> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<87> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<88> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<89> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<90> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<91> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<92> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<93> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<94> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<95> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<96> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<97> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<98> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::if_< boost::mpl::bool_< (sizeof(RCF_make_next_dispatch_id_func((DummyThisT *) 0, (ThisT *) 0, (boost::mpl::int_<99> *) 0)) == sizeof(RCF::defined_)) >,         \
    boost::mpl::int_<100>,                                                                                                                                                      \
    boost::mpl::int_<99> >::type,                                                                                                                                               \
    boost::mpl::int_<98> >::type,                                                                                                                                               \
    boost::mpl::int_<97> >::type,                                                                                                                                               \
    boost::mpl::int_<96> >::type,                                                                                                                                               \
    boost::mpl::int_<95> >::type,                                                                                                                                               \
    boost::mpl::int_<94> >::type,                                                                                                                                               \
    boost::mpl::int_<93> >::type,                                                                                                                                               \
    boost::mpl::int_<92> >::type,                                                                                                                                               \
    boost::mpl::int_<91> >::type,                                                                                                                                               \
    boost::mpl::int_<90> >::type,                                                                                                                                               \
    boost::mpl::int_<89> >::type,                                                                                                                                               \
    boost::mpl::int_<88> >::type,                                                                                                                                               \
    boost::mpl::int_<87> >::type,                                                                                                                                               \
    boost::mpl::int_<86> >::type,                                                                                                                                               \
    boost::mpl::int_<85> >::type,                                                                                                                                               \
    boost::mpl::int_<84> >::type,                                                                                                                                               \
    boost::mpl::int_<83> >::type,                                                                                                                                               \
    boost::mpl::int_<82> >::type,                                                                                                                                               \
    boost::mpl::int_<81> >::type,                                                                                                                                               \
    boost::mpl::int_<80> >::type,                                                                                                                                               \
    boost::mpl::int_<79> >::type,                                                                                                                                               \
    boost::mpl::int_<78> >::type,                                                                                                                                               \
    boost::mpl::int_<77> >::type,                                                                                                                                               \
    boost::mpl::int_<76> >::type,                                                                                                                                               \
    boost::mpl::int_<75> >::type,                                                                                                                                               \
    boost::mpl::int_<74> >::type,                                                                                                                                               \
    boost::mpl::int_<73> >::type,                                                                                                                                               \
    boost::mpl::int_<72> >::type,                                                                                                                                               \
    boost::mpl::int_<71> >::type,                                                                                                                                               \
    boost::mpl::int_<70> >::type,                                                                                                                                               \
    boost::mpl::int_<69> >::type,                                                                                                                                               \
    boost::mpl::int_<68> >::type,                                                                                                                                               \
    boost::mpl::int_<67> >::type,                                                                                                                                               \
    boost::mpl::int_<66> >::type,                                                                                                                                               \
    boost::mpl::int_<65> >::type,                                                                                                                                               \
    boost::mpl::int_<64> >::type,                                                                                                                                               \
    boost::mpl::int_<63> >::type,                                                                                                                                               \
    boost::mpl::int_<62> >::type,                                                                                                                                               \
    boost::mpl::int_<61> >::type,                                                                                                                                               \
    boost::mpl::int_<60> >::type,                                                                                                                                               \
    boost::mpl::int_<59> >::type,                                                                                                                                               \
    boost::mpl::int_<58> >::type,                                                                                                                                               \
    boost::mpl::int_<57> >::type,                                                                                                                                               \
    boost::mpl::int_<56> >::type,                                                                                                                                               \
    boost::mpl::int_<55> >::type,                                                                                                                                               \
    boost::mpl::int_<54> >::type,                                                                                                                                               \
    boost::mpl::int_<53> >::type,                                                                                                                                               \
    boost::mpl::int_<52> >::type,                                                                                                                                               \
    boost::mpl::int_<51> >::type,                                                                                                                                               \
    boost::mpl::int_<50> >::type,                                                                                                                                               \
    boost::mpl::int_<49> >::type,                                                                                                                                               \
    boost::mpl::int_<48> >::type,                                                                                                                                               \
    boost::mpl::int_<47> >::type,                                                                                                                                               \
    boost::mpl::int_<46> >::type,                                                                                                                                               \
    boost::mpl::int_<45> >::type,                                                                                                                                               \
    boost::mpl::int_<44> >::type,                                                                                                                                               \
    boost::mpl::int_<43> >::type,                                                                                                                                               \
    boost::mpl::int_<42> >::type,                                                                                                                                               \
    boost::mpl::int_<41> >::type,                                                                                                                                               \
    boost::mpl::int_<40> >::type,                                                                                                                                               \
    boost::mpl::int_<39> >::type,                                                                                                                                               \
    boost::mpl::int_<38> >::type,                                                                                                                                               \
    boost::mpl::int_<37> >::type,                                                                                                                                               \
    boost::mpl::int_<36> >::type,                                                                                                                                               \
    boost::mpl::int_<35> >::type,                                                                                                                                               \
    boost::mpl::int_<34> >::type,                                                                                                                                               \
    boost::mpl::int_<33> >::type,                                                                                                                                               \
    boost::mpl::int_<32> >::type,                                                                                                                                               \
    boost::mpl::int_<31> >::type,                                                                                                                                               \
    boost::mpl::int_<30> >::type,                                                                                                                                               \
    boost::mpl::int_<29> >::type,                                                                                                                                               \
    boost::mpl::int_<28> >::type,                                                                                                                                               \
    boost::mpl::int_<27> >::type,                                                                                                                                               \
    boost::mpl::int_<26> >::type,                                                                                                                                               \
    boost::mpl::int_<25> >::type,                                                                                                                                               \
    boost::mpl::int_<24> >::type,                                                                                                                                               \
    boost::mpl::int_<23> >::type,                                                                                                                                               \
    boost::mpl::int_<22> >::type,                                                                                                                                               \
    boost::mpl::int_<21> >::type,                                                                                                                                               \
    boost::mpl::int_<20> >::type,                                                                                                                                               \
    boost::mpl::int_<19> >::type,                                                                                                                                               \
    boost::mpl::int_<18> >::type,                                                                                                                                               \
    boost::mpl::int_<17> >::type,                                                                                                                                               \
    boost::mpl::int_<16> >::type,                                                                                                                                               \
    boost::mpl::int_<15> >::type,                                                                                                                                               \
    boost::mpl::int_<14> >::type,                                                                                                                                               \
    boost::mpl::int_<13> >::type,                                                                                                                                               \
    boost::mpl::int_<12> >::type,                                                                                                                                               \
    boost::mpl::int_<11> >::type,                                                                                                                                               \
    boost::mpl::int_<10> >::type,                                                                                                                                               \
    boost::mpl::int_< 9> >::type,                                                                                                                                               \
    boost::mpl::int_< 8> >::type,                                                                                                                                               \
    boost::mpl::int_< 7> >::type,                                                                                                                                               \
    boost::mpl::int_< 6> >::type,                                                                                                                                               \
    boost::mpl::int_< 5> >::type,                                                                                                                                               \
    boost::mpl::int_< 4> >::type,                                                                                                                                               \
    boost::mpl::int_< 3> >::type,                                                                                                                                               \
    boost::mpl::int_< 2> >::type,                                                                                                                                               \
    boost::mpl::int_< 1> >::type,                                                                                                                                               \
    boost::mpl::int_< 0> >::type next_dispatch_id;                                                                                                                              \
    friend RCF::defined_ RCF_make_next_dispatch_id_func(DummyThisT *, ThisT *, next_dispatch_id *)

#else

#error RCF_MAX_METHOD_COUNT > 100 is currently not implemented.

#endif // RCF_MAX_METHOD_COUNT

#endif // ! INCLUDE_RCF_IDL_HPP
