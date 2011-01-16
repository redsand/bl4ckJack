
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

#ifndef INCLUDE_RCF_TEST_PRINTTESTHEADER
#define INCLUDE_RCF_TEST_PRINTTESTHEADER

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <RCF/Config.hpp>
#include <RCF/util/AutoBuild.hpp>
#include <boost/version.hpp>

#ifdef RCF_USE_BOOST_ASIO
#include <boost/asio/version.hpp>
#include <RCF/Asio.hpp>
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4996 )  // warning C4996: '' was declared deprecated
#endif

inline void printTestHeader(const char *file)
{
    std::cout << "\n*********************\n";

    std::cout << "Compiler: ";

#if defined(_MSC_VER) && _MSC_VER == 1200
    std::cout << "Visual C++ 6";
#elif defined(_MSC_VER) && _MSC_VER == 1310
    std::cout << "Visual C++ 7.1";
#elif defined(_MSC_VER) && _MSC_VER == 1400
    std::cout << "Visual C++ 8.0";
#elif defined(_MSC_VER) && _MSC_VER == 1500
    std::cout << "Visual C++ 9.0";
#elif defined(_MSC_VER) && _MSC_VER == 1600
    std::cout << "Visual C++ 10.0";
#elif defined(_MSC_VER) 
    std::cout << "Visual C++ ??? - " << "_MSC_VER is " << _MSC_VER;
#endif

#if defined(__GNUC__)
    std::cout << "gcc " << __GNUC__ << "." << __GNUC_MINOR__;
#endif

#if defined(__BORLANDC__)
    std::cout << "Borland C++ - __BORLANDC__ is " << __BORLANDC__;
#endif

    std::cout << std::endl;
    std::cout << "Architecture (bits): " << 8*sizeof(void*) << std::endl;

    std::cout << "\n*********************\n";
    std::cout << file << std::endl;
    time_t now = time(NULL);
    std::cout << "Time now: " << std::string(ctime(&now));
    std::cout << "Defines:" << std::endl;

    std::cout << "BOOST_VERSION: " << BOOST_VERSION << std::endl;

#ifdef RCF_USE_BOOST_ASIO
    std::cout << "BOOST_ASIO_VERSION: " << BOOST_ASIO_VERSION << std::endl;
#endif

#ifdef RCF_USE_BOOST_ASIO
#if defined(BOOST_ASIO_HAS_IOCP)
    std::cout << "BOOST_ASIO_HAS_IOCP" << std::endl;
#elif defined(BOOST_ASIO_HAS_EPOLL)
    std::cout << "BOOST_ASIO_HAS_EPOLL" << std::endl;
#elif defined(BOOST_ASIO_HAS_KQUEUE)
    std::cout << "BOOST_ASIO_HAS_KQUEUE" << std::endl;
#elif defined(BOOST_ASIO_HAS_DEV_POLL)
    std::cout << "BOOST_ASIO_HAS_DEV_POLL" << std::endl;
#else
    std::cout << "Boost.Asio - using select()" << std::endl;
#endif
#endif

    std::cout << "RCF_MAX_METHOD_COUNT: " << RCF_MAX_METHOD_COUNT << std::endl;

#ifdef RCF_MULTI_THREADED
    std::cout << "RCF_MULTI_THREADED" << std::endl;
#endif

#ifdef RCF_SINGLE_THREADED
    std::cout << "RCF_SINGLE_THREADED" << std::endl;
#endif

#ifdef RCF_USE_BOOST_THREADS
    std::cout << "RCF_USE_BOOST_THREADS" << std::endl;
#endif

#ifdef RCF_USE_BOOST_ASIO
    std::cout << "RCF_USE_BOOST_ASIO" << std::endl;
#endif

#ifdef RCF_USE_SF_SERIALIZATION
    std::cout << "RCF_USE_SF_SERIALIZATION" << std::endl;
#endif

#ifdef RCF_USE_BOOST_SERIALIZATION
    std::cout << "RCF_USE_BOOST_SERIALIZATION" << std::endl;
#endif

#ifdef RCF_USE_BOOST_XML_SERIALIZATION
    std::cout << "RCF_USE_BOOST_XML_SERIALIZATION" << std::endl;
#endif

#ifdef RCF_USE_ZLIB
    std::cout << "RCF_USE_ZLIB" << std::endl;
#endif

#ifdef RCF_USE_OPENSSL
    std::cout << "RCF_USE_OPENSSL" << std::endl;
#endif

#ifdef BOOST_SP_ENABLE_DEBUG_HOOKS
    std::cout << "BOOST_SP_ENABLE_DEBUG_HOOKS" << std::endl;
#endif

    std::cout << "RCF_TEMP_DIR: " << RCF_TEMP_DIR << std::endl;
    
    std::cout << "*********************\n\n";
}

#ifdef _MSC_VER
#pragma warning( pop )
#endif


#endif // ! INCLUDE_RCF_TEST_PRINTTESTHEADER
