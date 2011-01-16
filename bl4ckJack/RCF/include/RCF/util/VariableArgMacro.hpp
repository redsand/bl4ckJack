
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

#ifndef INCLUDE_UTIL_VARIABLEARGMACRO_HPP
#define INCLUDE_UTIL_VARIABLEARGMACRO_HPP

#include <strstream>
#include <typeinfo>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>

#include "Platform/OS/GetCurrentTime.hpp"
#include "Platform/OS/ThreadId.hpp"

#include <RCF/Export.hpp>
#include <RCF/util/ThreadLibrary.hpp>

namespace util {

    class RCF_EXPORT VariableArgMacroFunctor : boost::noncopyable
    {
    public:

        VariableArgMacroFunctor();
        virtual ~VariableArgMacroFunctor();

        VariableArgMacroFunctor &init(
            const std::string &label,
            const std::string &msg,
            const char *file,
            int line,
            const char *func);

        template<typename T>
        void notify(const T &t, const char *name)
        {
            *mArgs << name << "=" << t << ", ";
        }

        void notify(size_t t, const char *name)
        {
            *mArgs << name << "=" << static_cast<unsigned int>(t) << ", ";
        }

        // TODO: it seems this workaround is needed in stlport 5.1.3, but not in 4.6.2
#if defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)
        void notify(boost::int64_t t, const char *name)
        {
            *mArgs << name << "=" << t << ", ";
        }
#endif

        // TODO: fix this properly
#ifdef _UNICODE
        void notify(const wchar_t *, const char *) {}
        void notify(const std::wstring &, const char *) {}
#endif

        template<typename T> T &cast()
        {
            return dynamic_cast<T &>(*this);
        }

        template<typename T> T &cast(T *)
        {
            return dynamic_cast<T &>(*this);
        }

        std::ostrstream * mHeader;
        std::ostrstream * mArgs;

        const char * mFile;
        int mLine;
        const char * mFunc;
    };

    class DummyVariableArgMacroObject
    {
    public:
        template<typename T>
        DummyVariableArgMacroObject &operator()(const T &)
        {
            return *this;
        }
    };

    #define DUMMY_VARIABLE_ARG_MACRO() \
        if (false) ::util::DummyVariableArgMacroObject()

} // namespace util


#define DECLARE_VARIABLE_ARG_MACRO(macro_name, functor)                                             \
    DECLARE_VARIABLE_ARG_MACRO_(macro_name, macro_name##_A, macro_name##_B, functor)

#define DECLARE_VARIABLE_ARG_MACRO_( macro_name, macro_name_A, macro_name_B, functor)               \
    template<typename Functor>                                                                      \
    class VariableArgMacro;                                                                         \
                                                                                                    \
    template<> class                                                                                \
    VariableArgMacro< functor > : public functor                                                    \
    {                                                                                               \
    public:                                                                                         \
        VariableArgMacro() :                                                                        \
            macro_name_A(*this), macro_name_B(*this)                                                \
        {}                                                                                          \
        template<typename T1>                                                                       \
        VariableArgMacro(const T1 &t1) :                                                            \
            functor(t1), macro_name_A(*this), macro_name_B(*this)                                   \
        {}                                                                                          \
        template<typename T1, typename T2>                                                          \
        VariableArgMacro(const T1 &t1, const T2 &t2) :                                              \
            functor(t1,t2), macro_name_A(*this), macro_name_B(*this)                                \
        {}                                                                                          \
        template<typename T1, typename T2>                                                          \
        VariableArgMacro(const T1 &t1, T2 &t2) :                                                    \
            functor(t1,t2), macro_name_A(*this), macro_name_B(*this)                                \
        {}                                                                                          \
        template<typename T1, typename T2, typename T3>                                             \
        VariableArgMacro(const T1 &t1, const T2 &t2, const T3 & t3) :                               \
            functor(t1,t2,t3), macro_name_A(*this), macro_name_B(*this)                             \
        {}                                                                                          \
        template<typename T1, typename T2, typename T3, typename T4, typename T5>                   \
        VariableArgMacro(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5) :                                       \
            functor(t1,t2,t3,t4,t5), macro_name_A(*this), macro_name_B(*this)                       \
        {}                                                                                          \
        ~VariableArgMacro()                                                                         \
        {}                                                                                          \
        VariableArgMacro &macro_name_A;                                                             \
        VariableArgMacro &macro_name_B;                                                             \
        template<typename T>                                                                        \
        VariableArgMacro< functor > &notify_(const T &t, const char *name)                          \
        {                                                                                           \
            notify(t,name); return *this;                                                           \
        }                                                                                           \
        VariableArgMacro< functor > &notify_(size_t t, const char *name)                            \
        {                                                                                           \
            notify(t,name); return *this;                                                            \
        }                                                                                           \
    }

#define DECLARE_VARIABLE_ARG_MACRO_T1( macro_name, functor )                                        \
    DECLARE_VARIABLE_ARG_MACRO_T1_(macro_name, macro_name##_A, macro_name##_B, functor)

#define DECLARE_VARIABLE_ARG_MACRO_T1_( macro_name, macro_name_A, macro_name_B, functor)            \
    template<typename Functor>                                                                      \
    class VariableArgMacro;                                                                         \
                                                                                                    \
    template<typename U>                                                                            \
    class VariableArgMacro< functor<U> > : public functor<U>                                        \
    {                                                                                               \
    public:                                                                                         \
        VariableArgMacro() :                                                                        \
            macro_name_A(*this), macro_name_B(*this)                                                \
        {}                                                                                          \
        template<typename T1>                                                                       \
        VariableArgMacro(const T1 &t1) :                                                            \
            functor<U>(t1), macro_name_A(*this), macro_name_B(*this)                                \
        {}                                                                                          \
        template<typename T1, typename T2>                                                          \
        VariableArgMacro(const T1 &t1, const T2 &t2) :                                              \
            functor<U>(t1,t2), macro_name_A(*this), macro_name_B(*this)                             \
        {}                                                                                          \
        ~VariableArgMacro()                                                                         \
        {}                                                                                          \
        VariableArgMacro &macro_name_A;                                                             \
        VariableArgMacro &macro_name_B;                                                             \
        template<typename T>                                                                        \
        VariableArgMacro< functor<U> > &notify_(const T &t, const char *name)                       \
        {                                                                                           \
            notify(t,name);                                                                         \
            return *this;                                                                           \
        }                                                                                           \
        VariableArgMacro< functor<U> > &notify_(size_t t, const char *name)                         \
        {                                                                                           \
            notify(t,name);                                                                            \
            return *this;                                                                           \
        }                                                                                           \
    }

#define DECLARE_VARIABLE_ARG_MACRO_T2( macro_name, functor )                                        \
    DECLARE_VARIABLE_ARG_MACRO_T2_(macro_name, macro_name##_A, macro_name##_B, functor)

#define DECLARE_VARIABLE_ARG_MACRO_T2_( macro_name, macro_name_A, macro_name_B, functor)            \
    template<typename Functor>                                                                      \
    class VariableArgMacro;                                                                         \
                                                                                                    \
    template<typename U, typename V>                                                                \
    class VariableArgMacro< functor<U,V> > : public functor<U,V>                                    \
    {                                                                                               \
    public:                                                                                         \
        VariableArgMacro() :                                                                        \
            macro_name_A(*this), macro_name_B(*this)                                                \
        {}                                                                                          \
        template<typename T1>                                                                       \
        VariableArgMacro(const T1 &t1) :                                                            \
            functor<U,V>(t1), macro_name_A(*this), macro_name_B(*this)                              \
        {}                                                                                          \
        template<typename T1, typename T2>                                                          \
        VariableArgMacro(const T1 &t1, const T2 &t2) :                                              \
            functor<U,V>(t1,t2), macro_name_A(*this), macro_name_B(*this)                           \
        {}                                                                                          \
        ~VariableArgMacro()                                                                         \
        {}                                                                                          \
        VariableArgMacro &macro_name_A;                                                             \
        VariableArgMacro &macro_name_B;                                                             \
        template<typename T>                                                                        \
        VariableArgMacro< functor<U,V> > &notify_(const T &t, const char *name)                     \
        {                                                                                           \
            notify(t,name);                                                                         \
            return *this;                                                                           \
        }                                                                                           \
        VariableArgMacro< functor<U,V> > &notify_(size_t t, const char *name)                       \
        {                                                                                           \
            notify(t,name);                                                                            \
            return *this;                                                                           \
        }                                                                                           \
    }

#endif // ! INCLUDE_UTIL_VARIABLEARGMACRO_HPP
