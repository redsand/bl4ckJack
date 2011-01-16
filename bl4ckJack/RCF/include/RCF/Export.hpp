
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

#ifndef INCLUDE_RCF_EXPORT_HPP
#define INCLUDE_RCF_EXPORT_HPP

#if defined(__GNUC__) && __GNUC__ >= 4 
    #ifdef RCF_BUILD_DLL
        #define RCF_EXPORT __attribute__((visibility("default")))
    #else
        #define RCF_EXPORT __attribute__((visibility("default")))
    #endif
#elif defined(__GNUC__)
    #ifdef RCF_BUILD_DLL
        #define RCF_EXPORT
    #else
        #define RCF_EXPORT
    #endif
#else
    #ifdef RCF_BUILD_DLL
        #define RCF_EXPORT __declspec(dllexport)
    #else
        #define RCF_EXPORT
    #endif
#endif


#if defined(RCF_BUILD_DLL) && !defined(RCF_NO_AUTO_INIT_DEINIT)
#ifdef _MSC_VER
#pragma message("Warning: DLL builds of RCF should define RCF_NO_AUTO_INIT_DEINIT and explicitly call RCF::init()/RCF::deinit().")
#endif
#endif

#if defined(RCF_BUILD_DLL) && defined(_MSC_VER) && !defined(_DLL)
#error "Error: DLL builds of RCF require dynamic runtime linking. Select one of the DLL options in Properties -> C/C++ -> Code Generation -> Runtime Library."
#endif

#endif // ! INCLUDE_RCF_EXPORT_HPP
