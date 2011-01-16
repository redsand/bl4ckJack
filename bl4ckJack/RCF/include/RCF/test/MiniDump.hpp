
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

#ifndef INCLUDE_RCF_TEST_MINIDUMP_HPP
#define INCLUDE_RCF_TEST_MINIDUMP_HPP

#include <DbgHelp.h>

#include <RCF/util/CommandLine.hpp>

void createMiniDump(EXCEPTION_POINTERS * pep = NULL);

void setMiniDumpExceptionFilter();

#endif
