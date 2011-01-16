
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

#ifndef INCLUDE_RCF_TOOLS_HPP
#define INCLUDE_RCF_TOOLS_HPP

// Various utilities

#include <stdlib.h>
#include <time.h>

#include <deque>
#include <iosfwd>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include <boost/bind.hpp>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/current_function.hpp>
#include <boost/noncopyable.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <RCF/Export.hpp>
#include <RCF/MinMax.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/util/UnusedVariable.hpp>
#include <RCF/util/Platform/OS/BsdSockets.hpp> // GetErrorString()

// Logging mechanism
#include <RCF/util/Log.hpp>

namespace RCF {
    static const int LogNameRcf = 1;
    static const int LogLevel_1 = 1; // Error and exceptions.
    static const int LogLevel_2 = 2; // Larger scale setup/teardown.
    static const int LogLevel_3 = 3; // Messages sent and received (RCF level), RCF client and session lifetime.
    static const int LogLevel_4 = 4; // Messages sent and received (network level), network client and session lifetime.

    typedef util::LoggerPtr         LoggerPtr;
    typedef util::Logger            Logger;
    typedef util::LogManager        LogManager;
    typedef util::LogEntry          LogEntry;
    typedef util::LogToStdout       LogToStdout;
    typedef util::LogToFile         LogToFile;
    typedef util::LogToFunc         LogToFunc;
    typedef util::LogFormatFunctor  LogFormatFunctor;

#ifdef BOOST_WINDOWS
    typedef util::LogToDebugWindow  LogToDebugWindow;
#endif

} // namespace RCF

#define RCF_LOG_1() UTIL_LOG(RCF::LogNameRcf, RCF::LogLevel_1)
#define RCF_LOG_2() UTIL_LOG(RCF::LogNameRcf, RCF::LogLevel_2)
#define RCF_LOG_3() UTIL_LOG(RCF::LogNameRcf, RCF::LogLevel_3)
#define RCF_LOG_4() UTIL_LOG(RCF::LogNameRcf, RCF::LogLevel_4)

// Assertion mechanism
#include <RCF/util/Assert.hpp>
#define RCF_ASSERT(x) UTIL_ASSERT(x, RCF::AssertionFailureException(), RCF::LogNameRcf, RCF::LogLevel_1)

#define RCF_ASSERT_EQ(a,b)      RCF_ASSERT(a == b)(a)(b)
#define RCF_ASSERT_NEQ(a,b)     RCF_ASSERT(a != b)(a)(b)

#define RCF_ASSERT_LT(a,b)      RCF_ASSERT(a < b)(a)(b)
#define RCF_ASSERT_LTEQ(a,b)    RCF_ASSERT(a <= b)(a)(b)

#define RCF_ASSERT_GT(a,b)      RCF_ASSERT(a > b)(a)(b)
#define RCF_ASSERT_GTEQ(a,b)    RCF_ASSERT(a >= b)(a)(b)

// Throw mechanism
#include <RCF/util/Throw.hpp>
#define RCF_THROW(e)          UTIL_THROW(e, RCF::LogNameRcf, RCF::LogLevel_1)

// Verification mechanism
#include <RCF/util/Throw.hpp>
#define RCF_VERIFY(cond, e) UTIL_VERIFY(cond, e, RCF::LogNameRcf, RCF::LogLevel_1)


// Scope guard mechanism
#include <boost/multi_index/detail/scope_guard.hpp>

namespace RCF 
{
    class Exception;
}

// assorted tracing conveniences
#ifndef __BORLANDC__
namespace std {
#endif

    // Trace std::vector
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
    {
        os << "(";
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
        os << ")";
        return os;
    }

    // Trace std::deque
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const std::deque<T> &d)
    {
        os << "(";
        std::copy(d.begin(), d.end(), std::ostream_iterator<T>(os, ", "));
        os << ")";
        return os;
    }

    // Trace type_info
    RCF_EXPORT std::ostream &operator<<(std::ostream &os, const std::type_info &ti);

    // Trace exception
    RCF_EXPORT std::ostream &operator<<(std::ostream &os, const std::exception &e);

    // Trace exception
    RCF_EXPORT std::ostream &operator<<(std::ostream &os, const RCF::Exception &e);

#ifndef __BORLANDC__
} // namespace std
#endif

namespace RCF {

    // Time in ms since ca 1970, modulo 65536 s (turns over every ~18.2 hrs).
    RCF_EXPORT unsigned int getCurrentTimeMs();

    // Generate a timeout value for the given ending time.
    // Returns zero if endTime <= current time <= endTime+10%of timer resolution, otherwise returns a nonzero duration in ms.
    // Timer resolution as above (18.2 hrs).
    RCF_EXPORT unsigned int generateTimeoutMs(unsigned int endTimeMs);

} // namespace RCF

// narrow/wide string utilities
#include <RCF/util/Tchar.hpp>
namespace RCF {

    typedef util::tstring tstring;

} // namespace RCF


namespace RCF {

    // null deleter, for use with for shared_ptr
    class NullDeleter
    {
    public:
        template<typename T>
        void operator()(T)
        {}
    };

    class SharedPtrIsNull
    {
    public:
        template<typename T>
        bool operator()(boost::shared_ptr<T> spt) const
        {
            return spt.get() == NULL;
        }
    };

} // namespace RCF

#include <RCF/util/DefaultInit.hpp>

namespace RCF {

    RCF_EXPORT std::string toString(const std::exception &e);

} // namespace RCF

// destructor try/catch blocks
#define RCF_DTOR_BEGIN                              \
    try {

#define RCF_DTOR_END                                \
    }                                               \
    catch (const std::exception &e)                 \
    {                                               \
        if (!util::detail::uncaught_exception())    \
        {                                           \
            throw;                                  \
        }                                           \
        else                                        \
        {                                           \
            RCF_LOG_1()(e);                         \
        }                                           \
    }

// vc6 issues

#if defined(_MSC_VER) && _MSC_VER == 1200

typedef unsigned long ULONG_PTR;

namespace std {

    std::ostream &operator<<(std::ostream &os, __int64);

    std::ostream &operator<<(std::ostream &os, unsigned __int64);

    std::istream &operator>>(std::istream &os, __int64 &);

    std::istream &operator>>(std::istream &os, unsigned __int64 &);

}

#endif

#if (__BORLANDC__ >= 0x560) && defined(_USE_OLD_RW_STL)
#include <boost/thread.hpp>
#include <boost/thread/tss.hpp>
#include <libs/thread/src/thread.cpp>
#include <libs/thread/src/tss.cpp>

static void DummyFuncToGenerateTemplateSpecializations(void)
{
    // http://lists.boost.org/boost-users/2005/06/12412.php

    // This forces generation of
    // boost::thread::thread(boost::function0<void>&)

    boost::function0<void> A = NULL;
    boost::thread B(A);

    // This forces generation of
    // boost::detail::tss::init(boost::function1<void, void *,
    // std::allocator<boost::function_base> > *)
    // but has the consequence of requiring the deliberately undefined function
    // tss_cleanup_implemented

    boost::function1<void, void*>* C = NULL;
    boost::detail::tss D(C);
}

#endif

#if defined(_MSC_VER) && _MSC_VER >= 1310
// need this for 64 bit builds with asio 0.3.7 (boost 1.33.1)
#include <boost/mpl/equal.hpp>
#include <boost/utility/enable_if.hpp>
namespace boost {
    template<typename T>
    inline std::size_t hash_value(
        T t,
        typename boost::enable_if< boost::mpl::equal<T, std::size_t> >::type * = 0)
    {
        return t;
    }
}
#endif

#if defined(_MSC_VER) && _MSC_VER < 1310
#define RCF_PFTO_HACK long
#else
#define RCF_PFTO_HACK
#endif

// Auto linking on VC++
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "mswsock.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "crypt32.lib")
#endif

namespace RCF {

    struct Void {};

    // Bizarre compiler errors when trying to use erase-remove with Borland 5.6,
    // so here's a manual implementation instead.

#if defined(__BORLANDC__) && __BORLANDC__ <= 0x560

    template<typename Container, typename Element>
    void eraseRemove(Container & container, const Element & element)
    {
        std::size_t i = 0;
        while( i < container.size() )
        {
            if (container[i] == element)
            {
                for (std::size_t j = i; j < container.size() - 1; ++j)
                {
                    container[j] = container[j+1];
                }
                container.pop_back();
            }
            else
            {
                ++i;
            }
        }        
    }


#else

    template<typename Container, typename Element>
    void eraseRemove(Container & container, const Element & element)
    {
        container.erase(
            std::remove(
                container.begin(),
                container.end(),
                element),
            container.end());
    }

#endif

    RCF_EXPORT boost::uint64_t fileSize(const std::string & path);

} // namespace RCF

namespace boost {
    
    template<typename T>
    inline bool operator==(
        const boost::weak_ptr<T> & lhs, 
        const boost::weak_ptr<T> & rhs)
    {
        return ! (lhs < rhs) && ! (rhs < lhs);
    }

    template<typename T>
    inline bool operator!=(
        const boost::weak_ptr<T> & lhs, 
        const boost::weak_ptr<T> & rhs)
    {
        return ! (lhs == rhs);
    }

} // namespace boost

#endif // ! INCLUDE_RCF_TOOLS_HPP
