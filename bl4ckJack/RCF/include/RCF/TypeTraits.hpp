
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

#ifndef INCLUDE_RCF_TYPETRAITS_HPP
#define INCLUDE_RCF_TYPETRAITS_HPP

#include <boost/type_traits.hpp>

namespace RCF {

    template<typename T>
    struct IsFundamental : public boost::is_fundamental<T>
    {};

#if defined(_MSC_VER) && _MSC_VER < 1310

    template<>
    struct IsFundamental<long double> : boost::mpl::true_
    {};

    template<>
    struct IsFundamental<__int64> : boost::mpl::true_
    {};

    template<>
    struct IsFundamental<unsigned __int64> : boost::mpl::true_
    {};

#endif

    template<typename T>
    struct IsConst : public boost::is_const<T>
    {};

    template<typename T>
    struct IsPointer : public boost::is_pointer<T>
    {};

    template<typename T>
    struct IsReference : public boost::is_reference<T>
    {};

    template<typename T>
    struct RemovePointer : public boost::remove_pointer<T>
    {};

    template<typename T>
    struct RemoveReference : public boost::remove_reference<T>
    {};

    template<typename T>
    struct RemoveCv : public boost::remove_cv<T>
    {};

    template<typename T>
    struct Out
    {
    };

    template<typename T>
    struct RemoveOut
    {
        typedef T type;
    };

    template<typename T>
    struct IsOut : public boost::mpl::false_
    {
    };

#if !defined(_MSC_VER) || _MSC_VER > 1200

    template<typename T>
    struct RemoveOut< Out<T> >
    {
        typedef T type;
    };

    template<typename T>
    struct IsOut< Out<T> >
    {
        typedef boost::mpl::true_ type;
        enum { value = type::value };
    };

#endif


} // namespace RCF

namespace SF {

    template<typename T> struct GetIndirection;

} // namespace SF

#if defined(_MSC_VER) && _MSC_VER <= 1310

// Note for vc6 users: this macro needs to be applied to a type before
// the type is used as a parameter in a RCF interface.

#define RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(T)                           \
namespace RCF {                                                                     \
                                                                                    \
    template<>                                                                      \
    struct IsReference<T > : public boost::mpl::false_                              \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsReference<T &> : public boost::mpl::true_                              \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsReference<const T &> : public boost::mpl::true_                        \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveReference<T &> : public boost::mpl::identity<T >                   \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveReference<const T &> : public boost::mpl::identity<const T >       \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T > : public boost::mpl::false_                                \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T *> : public boost::mpl::true_                                \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T const *> : public boost::mpl::true_                          \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T * const> : public boost::mpl::true_                          \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T * const *> : public boost::mpl::true_                        \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T const * const> : public boost::mpl::true_                    \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsPointer<T const * const *> : public boost::mpl::true_                  \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T > : public boost::mpl::identity<T >                      \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T *> : public boost::mpl::identity<T >                     \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T * const> : public boost::mpl::identity<T >               \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T const *> : public boost::mpl::identity<const T >         \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T const * const> : public boost::mpl::identity<const T >   \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T * const *> : public boost::mpl::identity<T * const >     \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemovePointer<T const * const *> : public boost::mpl::identity<T const * const >   \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveCv<T > : public boost::mpl::identity<T >                           \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveCv<const T > : public boost::mpl::identity<T >                     \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveCv<volatile T > : public boost::mpl::identity<T >                  \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveCv<const volatile T > : public boost::mpl::identity<T >            \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveOut< Out< T > > : public boost::mpl::identity< T >                 \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveOut< Out< T & > > : public boost::mpl::identity< T & >             \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct RemoveOut< Out< const T & > > : public boost::mpl::identity< const T & > \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsOut< Out< T > > : public boost::mpl::true_                             \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsOut< Out< T & > > : public boost::mpl::true_                           \
    {};                                                                             \
                                                                                    \
    template<>                                                                      \
    struct IsOut< Out< const T & > > : public boost::mpl::true_                     \
    {};                                                                             \
                                                                                    \
}

#else

#define RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(T)

#endif

#endif // ! INCLUDE_RCF_TYPETRAITS_HPP
