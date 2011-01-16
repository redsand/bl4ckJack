
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

#ifndef INCLUDE_RCF_RCFMETHODGEN_HPP
#define INCLUDE_RCF_RCFMETHODGEN_HPP


// Parameters - 0.

#define RCF_METHOD_R0(R,func  )                                               \
            RCF_METHOD_R0_(R,func  , RCF_MAKE_UNIQUE_ID(func, R0))

#define RCF_METHOD_R0_(R,func  , id)                                          \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                )                                                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions()                                      \
                    );                                                        \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions                         \
                )                                                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R                                                     \
                         ,                                                    \
                        V,V,V,V,V,V,V,V,V,V,V,V,V,V,V >()(                    \
                            getClientStub()                                   \
                             ,                                                \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R                                                         \
                     > &p =                                                   \
                    ::RCF::AllocateServerParameters<                          \
                        R                                                     \
                         >()(session);                                        \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        ));                                                   \
            }

#define RCF_METHOD_V0(R,func  )                                               \
            RCF_METHOD_V0_(R,func   , RCF_MAKE_UNIQUE_ID(func, V0))

#define RCF_METHOD_V0_(R,func  , id)                                          \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                )                                                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions()                                      \
                    );                                                        \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions                         \
                )                                                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V                                                     \
                         ,                                                    \
                        V,V,V,V,V,V,V,V,V,V,V,V,V,V,V >()(                    \
                            getClientStub()                                   \
                             ,                                                \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V                                                         \
                     > &p =                                                   \
                        ::RCF::AllocateServerParameters<                      \
                            V                                                 \
                             >()(session);                                    \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        );                                                    \
            }


// Parameters - 1.

#define RCF_METHOD_R1(R,func , A1)                                            \
            RCF_METHOD_R1_(R,func , A1, RCF_MAKE_UNIQUE_ID(func, R1))

#define RCF_METHOD_R1_(R,func , A1, id)                                       \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1);                                                      \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1 ,                                                  \
                        V,V,V,V,V,V,V,V,V,V,V,V,V,V >()(                      \
                            getClientStub() ,                                 \
                            a1 ,                                              \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1 > &p =                                                 \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1 >()(session);                                      \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get()));                                         \
            }

#define RCF_METHOD_V1(R,func , A1)                                            \
            RCF_METHOD_V1_(R,func  , A1, RCF_MAKE_UNIQUE_ID(func, V1))

#define RCF_METHOD_V1_(R,func , A1, id)                                       \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1);                                                      \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1 ,                                                  \
                        V,V,V,V,V,V,V,V,V,V,V,V,V,V >()(                      \
                            getClientStub() ,                                 \
                            a1 ,                                              \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1 > &p =                                                 \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1 >()(session);                                  \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get());                                          \
            }


// Parameters - 2.

#define RCF_METHOD_R2(R,func , A1,A2)                                         \
            RCF_METHOD_R2_(R,func , A1,A2, RCF_MAKE_UNIQUE_ID(func, R2))

#define RCF_METHOD_R2_(R,func , A1,A2, id)                                    \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2);                                                   \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2 ,                                               \
                        V,V,V,V,V,V,V,V,V,V,V,V,V >()(                        \
                            getClientStub() ,                                 \
                            a1,a2 ,                                           \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2 > &p =                                              \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2 >()(session);                                   \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get()));                                         \
            }

#define RCF_METHOD_V2(R,func , A1,A2)                                         \
            RCF_METHOD_V2_(R,func  , A1,A2, RCF_MAKE_UNIQUE_ID(func, V2))

#define RCF_METHOD_V2_(R,func , A1,A2, id)                                    \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2);                                                   \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2 ,                                               \
                        V,V,V,V,V,V,V,V,V,V,V,V,V >()(                        \
                            getClientStub() ,                                 \
                            a1,a2 ,                                           \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2 > &p =                                              \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2 >()(session);                               \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get());                                          \
            }


// Parameters - 3.

#define RCF_METHOD_R3(R,func , A1,A2,A3)                                      \
            RCF_METHOD_R3_(R,func , A1,A2,A3, RCF_MAKE_UNIQUE_ID(func, R3))

#define RCF_METHOD_R3_(R,func , A1,A2,A3, id)                                 \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3);                                                \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3 ,                                            \
                        V,V,V,V,V,V,V,V,V,V,V,V >()(                          \
                            getClientStub() ,                                 \
                            a1,a2,a3 ,                                        \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3 > &p =                                           \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3 >()(session);                                \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get()));                                         \
            }

#define RCF_METHOD_V3(R,func , A1,A2,A3)                                      \
            RCF_METHOD_V3_(R,func  , A1,A2,A3, RCF_MAKE_UNIQUE_ID(func, V3))

#define RCF_METHOD_V3_(R,func , A1,A2,A3, id)                                 \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3);                                                \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3 ,                                            \
                        V,V,V,V,V,V,V,V,V,V,V,V >()(                          \
                            getClientStub() ,                                 \
                            a1,a2,a3 ,                                        \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3 > &p =                                           \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3 >()(session);                            \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get());                                          \
            }


// Parameters - 4.

#define RCF_METHOD_R4(R,func , A1,A2,A3,A4)                                   \
            RCF_METHOD_R4_(R,func , A1,A2,A3,A4, RCF_MAKE_UNIQUE_ID(func, R4))

#define RCF_METHOD_R4_(R,func , A1,A2,A3,A4, id)                              \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4);                                             \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4 ,                                         \
                        V,V,V,V,V,V,V,V,V,V,V >()(                            \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4 ,                                     \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4 > &p =                                        \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4 >()(session);                             \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get()));                                         \
            }

#define RCF_METHOD_V4(R,func , A1,A2,A3,A4)                                   \
            RCF_METHOD_V4_(R,func  , A1,A2,A3,A4, RCF_MAKE_UNIQUE_ID(func, V4))

#define RCF_METHOD_V4_(R,func , A1,A2,A3,A4, id)                              \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4);                                             \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4 ,                                         \
                        V,V,V,V,V,V,V,V,V,V,V >()(                            \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4 ,                                     \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),\
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4 > &p =                                        \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4 >()(session);                         \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get());                                          \
            }


// Parameters - 5.

#define RCF_METHOD_R5(R,func , A1,A2,A3,A4,A5)                                \
            RCF_METHOD_R5_(R,func , A1,A2,A3,A4,A5, RCF_MAKE_UNIQUE_ID(func, R5))

#define RCF_METHOD_R5_(R,func , A1,A2,A3,A4,A5, id)                           \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5);                                          \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5 ,                                      \
                        V,V,V,V,V,V,V,V,V,V >()(                              \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5 ,                                  \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(), \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5 > &p =                                     \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5 >()(session);                          \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get()));                                         \
            }

#define RCF_METHOD_V5(R,func , A1,A2,A3,A4,A5)                                \
            RCF_METHOD_V5_(R,func  , A1,A2,A3,A4,A5, RCF_MAKE_UNIQUE_ID(func, V5))

#define RCF_METHOD_V5_(R,func , A1,A2,A3,A4,A5, id)                           \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5);                                          \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5 ,                                      \
                        V,V,V,V,V,V,V,V,V,V >()(                              \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5 ,                                  \
                            V(),V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(), \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5 > &p =                                     \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5 >()(session);                      \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get());                                          \
            }


// Parameters - 6.

#define RCF_METHOD_R6(R,func , A1,A2,A3,A4,A5,A6)                             \
            RCF_METHOD_R6_(R,func , A1,A2,A3,A4,A5,A6, RCF_MAKE_UNIQUE_ID(func, R6))

#define RCF_METHOD_R6_(R,func , A1,A2,A3,A4,A5,A6, id)                        \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6);                                       \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6 ,                                   \
                        V,V,V,V,V,V,V,V,V >()(                                \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6 ,                               \
                            V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6 > &p =                                  \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6 >()(session);                       \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get()));                                         \
            }

#define RCF_METHOD_V6(R,func , A1,A2,A3,A4,A5,A6)                             \
            RCF_METHOD_V6_(R,func  , A1,A2,A3,A4,A5,A6, RCF_MAKE_UNIQUE_ID(func, V6))

#define RCF_METHOD_V6_(R,func , A1,A2,A3,A4,A5,A6, id)                        \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6);                                       \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6 ,                                   \
                        V,V,V,V,V,V,V,V,V >()(                                \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6 ,                               \
                            V(),V(),V(),V(),V(),V(),V(),V(),V()).r.get(),     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6 > &p =                                  \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6 >()(session);                   \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get());                                          \
            }


// Parameters - 7.

#define RCF_METHOD_R7(R,func , A1,A2,A3,A4,A5,A6,A7)                          \
            RCF_METHOD_R7_(R,func , A1,A2,A3,A4,A5,A6,A7, RCF_MAKE_UNIQUE_ID(func, R7))

#define RCF_METHOD_R7_(R,func , A1,A2,A3,A4,A5,A6,A7, id)                     \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7);                                    \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7 ,                                \
                        V,V,V,V,V,V,V,V >()(                                  \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7 ,                            \
                            V(),V(),V(),V(),V(),V(),V(),V()).r.get(),         \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7 > &p =                               \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7 >()(session);                    \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get()));                                         \
            }

#define RCF_METHOD_V7(R,func , A1,A2,A3,A4,A5,A6,A7)                          \
            RCF_METHOD_V7_(R,func  , A1,A2,A3,A4,A5,A6,A7, RCF_MAKE_UNIQUE_ID(func, V7))

#define RCF_METHOD_V7_(R,func , A1,A2,A3,A4,A5,A6,A7, id)                     \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7);                                    \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7 ,                                \
                        V,V,V,V,V,V,V,V >()(                                  \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7 ,                            \
                            V(),V(),V(),V(),V(),V(),V(),V()).r.get(),         \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7 > &p =                               \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7 >()(session);                \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get());                                          \
            }


// Parameters - 8.

#define RCF_METHOD_R8(R,func , A1,A2,A3,A4,A5,A6,A7,A8)                       \
            RCF_METHOD_R8_(R,func , A1,A2,A3,A4,A5,A6,A7,A8, RCF_MAKE_UNIQUE_ID(func, R8))

#define RCF_METHOD_R8_(R,func , A1,A2,A3,A4,A5,A6,A7,A8, id)                  \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8);                                 \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8 ,                             \
                        V,V,V,V,V,V,V >()(                                    \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8 ,                         \
                            V(),V(),V(),V(),V(),V(),V()).r.get(),             \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8 > &p =                            \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8 >()(session);                 \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get()));                                         \
            }

#define RCF_METHOD_V8(R,func , A1,A2,A3,A4,A5,A6,A7,A8)                       \
            RCF_METHOD_V8_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8, RCF_MAKE_UNIQUE_ID(func, V8))

#define RCF_METHOD_V8_(R,func , A1,A2,A3,A4,A5,A6,A7,A8, id)                  \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8);                                 \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8 ,                             \
                        V,V,V,V,V,V,V >()(                                    \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8 ,                         \
                            V(),V(),V(),V(),V(),V(),V()).r.get(),             \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8 > &p =                            \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8 >()(session);             \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get());                                          \
            }


// Parameters - 9.

#define RCF_METHOD_R9(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9)                    \
            RCF_METHOD_R9_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9, RCF_MAKE_UNIQUE_ID(func, R9))

#define RCF_METHOD_R9_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9, id)               \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9);                              \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9 ,                          \
                        V,V,V,V,V,V >()(                                      \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9 ,                      \
                            V(),V(),V(),V(),V(),V()).r.get(),                 \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9 > &p =                         \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9 >()(session);              \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get()));                                         \
            }

#define RCF_METHOD_V9(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9)                    \
            RCF_METHOD_V9_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9, RCF_MAKE_UNIQUE_ID(func, V9))

#define RCF_METHOD_V9_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9, id)               \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9)                               \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9);                              \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9)                               \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9 ,                          \
                        V,V,V,V,V,V >()(                                      \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9 ,                      \
                            V(),V(),V(),V(),V(),V()).r.get(),                 \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9 > &p =                         \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9 >()(session);          \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get());                                          \
            }


// Parameters - 10.

#define RCF_METHOD_R10(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10)               \
            RCF_METHOD_R10_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, RCF_MAKE_UNIQUE_ID(func, R10))

#define RCF_METHOD_R10_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, id)          \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);                          \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 ,                      \
                        V,V,V,V,V >()(                                        \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 ,                  \
                            V(),V(),V(),V(),V()).r.get(),                     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 > &p =                     \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 >()(session);          \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get()));                                        \
            }

#define RCF_METHOD_V10(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10)               \
            RCF_METHOD_V10_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, RCF_MAKE_UNIQUE_ID(func, V10))

#define RCF_METHOD_V10_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, id)          \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10);                          \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 ,                      \
                        V,V,V,V,V >()(                                        \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 ,                  \
                            V(),V(),V(),V(),V()).r.get(),                     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 > &p =                     \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 >()(session);      \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get());                                         \
            }


// Parameters - 11.

#define RCF_METHOD_R11(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11)           \
            RCF_METHOD_R11_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11, RCF_MAKE_UNIQUE_ID(func, R11))

#define RCF_METHOD_R11_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11, id)      \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11);                      \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 ,                  \
                        V,V,V,V >()(                                          \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 ,              \
                            V(),V(),V(),V()).r.get(),                         \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 > &p =                 \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 >()(session);      \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get()));                                        \
            }

#define RCF_METHOD_V11(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11)           \
            RCF_METHOD_V11_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11, RCF_MAKE_UNIQUE_ID(func, V11))

#define RCF_METHOD_V11_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11, id)      \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11);                      \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 ,                  \
                        V,V,V,V >()(                                          \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 ,              \
                            V(),V(),V(),V()).r.get(),                         \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 > &p =                 \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11 >()(session);  \
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get());                                         \
            }


// Parameters - 12.

#define RCF_METHOD_R12(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12)       \
            RCF_METHOD_R12_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, RCF_MAKE_UNIQUE_ID(func, R12))

#define RCF_METHOD_R12_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, id)  \
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12);                  \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 ,              \
                        V,V,V >()(                                            \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 ,          \
                            V(),V(),V()).r.get(),                             \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 > &p =             \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 >()(session);  \
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get()));                                        \
            }

#define RCF_METHOD_V12(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12)       \
            RCF_METHOD_V12_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, RCF_MAKE_UNIQUE_ID(func, V12))

#define RCF_METHOD_V12_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, id)  \
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12);                  \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 ,              \
                        V,V,V >()(                                            \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 ,          \
                            V(),V(),V()).r.get(),                             \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 > &p =             \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12 >()(session);\
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get());                                         \
            }


// Parameters - 13.

#define RCF_METHOD_R13(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13)   \
            RCF_METHOD_R13_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13, RCF_MAKE_UNIQUE_ID(func, R13))

#define RCF_METHOD_R13_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13, id)\
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13);              \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 ,          \
                        V,V >()(                                              \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13 ,      \
                            V(),V()).r.get(),                                 \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 > &p =         \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 >()(session);\
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get()));                                        \
            }

#define RCF_METHOD_V13(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13)   \
            RCF_METHOD_V13_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13, RCF_MAKE_UNIQUE_ID(func, V13))

#define RCF_METHOD_V13_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13, id)\
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13);              \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 ,          \
                        V,V >()(                                              \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13 ,      \
                            V(),V()).r.get(),                                 \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 > &p =         \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 >()(session);\
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get());                                         \
            }


// Parameters - 14.

#define RCF_METHOD_R14(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14)\
            RCF_METHOD_R14_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14, RCF_MAKE_UNIQUE_ID(func, R14))

#define RCF_METHOD_R14_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14, id)\
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14);          \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 ,      \
                        V >()(                                                \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 ,  \
                            V()).r.get(),                                     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 > &p =     \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 >()(session);\
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get(),                                          \
                        p.a14.get()));                                        \
            }

#define RCF_METHOD_V14(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14)\
            RCF_METHOD_V14_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14, RCF_MAKE_UNIQUE_ID(func, V14))

#define RCF_METHOD_V14_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14, id)\
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14);          \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 ,      \
                        V >()(                                                \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 ,  \
                            V()).r.get(),                                     \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 > &p =     \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14 >()(session);\
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get(),                                          \
                        p.a14.get());                                         \
            }


// Parameters - 15.

#define RCF_METHOD_R15(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15)\
            RCF_METHOD_R15_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15, RCF_MAKE_UNIQUE_ID(func, R15))

#define RCF_METHOD_R15_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15, id)\
        public:                                                               \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<R > func(                                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14,                             \
                ::RCF::RemoveOut<A15 >::type a15)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);      \
            }                                                                 \
            ::RCF::FutureImpl<R > func(                                       \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14,                             \
                ::RCF::RemoveOut<A15 >::type a15)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<R >(                                   \
                    ::RCF::AllocateClientParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15    \
                         >()(                                                 \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 \
                            ).r.get(),                                        \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    R ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15 > &p = \
                    ::RCF::AllocateServerParameters<                          \
                        R ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15 >()(session);\
                p.r.set(                                                      \
                    session.getAutoSend(),                                    \
                    t.func(                                                   \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get(),                                          \
                        p.a14.get(),                                          \
                        p.a15.get()));                                        \
            }

#define RCF_METHOD_V15(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15)\
            RCF_METHOD_V15_(R,func  , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15, RCF_MAKE_UNIQUE_ID(func, V15))

#define RCF_METHOD_V15_(R,func , A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15, id)\
        public:                                                               \
            BOOST_STATIC_ASSERT(( boost::is_same<R, void>::value ));          \
            RCF_MAKE_NEXT_DISPATCH_ID(id);                                    \
            ::RCF::FutureImpl<V> func(                                        \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14,                             \
                ::RCF::RemoveOut<A15 >::type a15)                             \
            {                                                                 \
                return func(                                                  \
                    ::RCF::CallOptions() ,                                    \
                    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);      \
            }                                                                 \
            ::RCF::FutureImpl<V> func(                                        \
                const ::RCF::CallOptions &callOptions ,                       \
                ::RCF::RemoveOut<A1 >::type a1,                               \
                ::RCF::RemoveOut<A2 >::type a2,                               \
                ::RCF::RemoveOut<A3 >::type a3,                               \
                ::RCF::RemoveOut<A4 >::type a4,                               \
                ::RCF::RemoveOut<A5 >::type a5,                               \
                ::RCF::RemoveOut<A6 >::type a6,                               \
                ::RCF::RemoveOut<A7 >::type a7,                               \
                ::RCF::RemoveOut<A8 >::type a8,                               \
                ::RCF::RemoveOut<A9 >::type a9,                               \
                ::RCF::RemoveOut<A10 >::type a10,                             \
                ::RCF::RemoveOut<A11 >::type a11,                             \
                ::RCF::RemoveOut<A12 >::type a12,                             \
                ::RCF::RemoveOut<A13 >::type a13,                             \
                ::RCF::RemoveOut<A14 >::type a14,                             \
                ::RCF::RemoveOut<A15 >::type a15)                             \
            {                                                                 \
                getClientStub().setAsync(false);                              \
                return RCF::FutureImpl<V>(                                    \
                    ::RCF::AllocateClientParameters<                          \
                        V ,                                                   \
                        A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15    \
                         >()(                                                 \
                            getClientStub() ,                                 \
                            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 \
                            ).r.get(),                                        \
                    getClientStub(),                                          \
                    ::RCF::getInterfaceName( (Interface *) NULL),             \
                    id::value,                                                \
                    callOptions.apply(getClientStub()));                      \
            }                                                                 \
                                                                              \
        private:                                                              \
            template<typename T>                                              \
            void invoke(                                                      \
                const id &,                                                   \
                ::RCF::RcfSession &session,                                   \
                T &t)                                                         \
            {                                                                 \
                ::RCF::ServerParameters<                                      \
                    V ,                                                       \
                    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15 > &p = \
                        ::RCF::AllocateServerParameters<                      \
                            V ,                                               \
                            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15 >()(session);\
                RCF_UNUSED_VARIABLE(p);                                       \
                t.func(                                                       \
                        p.a1.get(),                                           \
                        p.a2.get(),                                           \
                        p.a3.get(),                                           \
                        p.a4.get(),                                           \
                        p.a5.get(),                                           \
                        p.a6.get(),                                           \
                        p.a7.get(),                                           \
                        p.a8.get(),                                           \
                        p.a9.get(),                                           \
                        p.a10.get(),                                          \
                        p.a11.get(),                                          \
                        p.a12.get(),                                          \
                        p.a13.get(),                                          \
                        p.a14.get(),                                          \
                        p.a15.get());                                         \
            }

#endif // ! INCLUDE_RCF_RCFMETHODGEN_HPP