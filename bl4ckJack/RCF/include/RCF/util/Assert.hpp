
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

#ifndef INCLUDE_UTIL_ASSERT_HPP
#define INCLUDE_UTIL_ASSERT_HPP

#include <cassert>
#include <exception>
#include <iostream>

#include <boost/current_function.hpp>

#include "Throw.hpp"
#include "VariableArgMacro.hpp"

#if defined(_MSC_VER) && !defined(NDEBUG)
#include <crtdbg.h>
#endif

namespace util {

    class AssertFunctor : public VariableArgMacroFunctor
    {
    public:

        AssertFunctor() : mExpr(NULL)
        {
        }

        AssertFunctor(const char * expr) : mExpr(expr)
        {
        }

#if defined(_MSC_VER) && !defined(NDEBUG)
#pragma warning(push)
#pragma warning(disable: 4995) // 'sprintf': name was marked as #pragma deprecated
#pragma warning(disable: 4996) // 'sprintf': This function or variable may be unsafe.

        ~AssertFunctor()
        {
            const char * msg = 
                "%s\n"
                "Values: %s\n"
                "Function: %s";

            std::string values(mArgs->str(), static_cast<std::size_t>(mArgs->pcount()));

            char szBuffer[512] = {0};
            sprintf(szBuffer, "%s(%d): Assert failed. Expression: %s.\n", mFile, mLine, mExpr);
            OutputDebugStringA(szBuffer);
            int ret = _CrtDbgReport(_CRT_ASSERT, mFile, mLine, "", msg, mExpr, values.c_str(), mFunc);
            if (ret == 1)
            {
                DebugBreak();
            }
        }

#pragma warning(pop)
#else

        ~AssertFunctor()
        {
            std::string values(mArgs->str(), static_cast<std::size_t>(mArgs->pcount()));
            
            std::cout 
                << mFile << ":" << mLine 
                << ": Assertion failed. " << mExpr 
                << " . Values: " << values << std::endl;

            assert(0 && "See line above for assertion details.");
        }

#endif

        const char * mExpr;
    };

    class VarArgAbort
    {
    public:
        VarArgAbort()
        {
            abort();
        }

        template<typename T>
        VarArgAbort &operator()(const T &)
        {
            return *this;
        }
    };


#if 0
#define UTIL_ASSERT_DEBUG(cond, e, logName, logLevel)                       \
    if (cond) ;                                                             \
    else util::VarArgAssert(__FILE__, __LINE__, #cond)
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4355 )  // warning C4355: 'this' : used in base member initializer list
#endif

#if defined(__GNUC__) && (__GNUC__ < 3 || (__GNUC__ == 3 && __GNUC_MINOR__ < 4))
#define UTIL_ASSERT_DEBUG_GCC_33_HACK (const util::VariableArgMacro<util::ThrowFunctor> &)
#else
#define UTIL_ASSERT_DEBUG_GCC_33_HACK
#endif

    DECLARE_VARIABLE_ARG_MACRO( UTIL_ASSERT_DEBUG, util::AssertFunctor );
    #define UTIL_ASSERT_DEBUG(expr, e, logName, logLevel)                                    \
        if (expr);    \
        else\
            UTIL_ASSERT_DEBUG_GCC_33_HACK                                              \
            util::VariableArgMacro<util::AssertFunctor>(#expr)    \
                .init(                                                          \
                    "",                                                         \
                    "",                                                         \
                    __FILE__,                                                   \
                    __LINE__,                                                   \
                    BOOST_CURRENT_FUNCTION)                                     \
                .cast( (util::VariableArgMacro<util::AssertFunctor> *) NULL)     \
                .UTIL_ASSERT_DEBUG_A



    #define UTIL_ASSERT_DEBUG_A(x)               UTIL_ASSERT_DEBUG_OP(x, B)
    #define UTIL_ASSERT_DEBUG_B(x)               UTIL_ASSERT_DEBUG_OP(x, A)
    #define UTIL_ASSERT_DEBUG_OP(x, next)        UTIL_ASSERT_DEBUG_A.notify_((x), #x).UTIL_ASSERT_DEBUG_ ## next

#ifdef _MSC_VER
#pragma warning( pop )
#endif




    #ifdef RCF_ALWAYS_ABORT_ON_ASSERT

    #define UTIL_ASSERT_RELEASE(cond, e, logName, logLevel)                     \
        if (cond) ;                                                             \
        else util::VarArgAbort()

    #else

    #define UTIL_ASSERT_RELEASE(cond, e, logName, logLevel)                     \
        if (cond) ;                                                             \
        else UTIL_THROW(e, logName, logLevel)(cond)

    #endif

    #define UTIL_ASSERT_NULL(cond, E)                                           \
        DUMMY_VARIABLE_ARG_MACRO()

    #ifdef NDEBUG
    #define UTIL_ASSERT UTIL_ASSERT_RELEASE
    #else
    #define UTIL_ASSERT UTIL_ASSERT_DEBUG
    #endif

} // namespace util

#endif //! INCLUDE_UTIL_ASSERT_HPP
