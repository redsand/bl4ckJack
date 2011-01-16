
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

#include <iostream>

#include <RCF/test/Test.hpp>

bool gInstrumented = false;
bool gExpectAllocations = true;
std::size_t gnAllocations = 0;

// User-defined operator new.
void *operator new(size_t bytes)
{
    if (gInstrumented)
    {
        
        if (!gExpectAllocations)
        {
            // Only flag the first unexpected allocation, so we don't end up 
            // with thousands of failures.

            gInstrumented = false;
            RCF_CHECK(gExpectAllocations);
            std::cout << "Unexpected memory allocation." << std::endl;
        }
        ++gnAllocations;
    }
    return malloc(bytes);
}

void operator delete (void *pv) throw()
{
    free(pv);
}

void *operator new [](size_t bytes)
{
    if (gInstrumented)
    {

        if (!gExpectAllocations)
        {
            // Only flag the first unexpected allocation, so we don't end up 
            // with thousands of failures.

            gInstrumented = false;
            RCF_CHECK(gExpectAllocations);
            std::cout << "Unexpected memory allocation." << std::endl;
        }
        ++gnAllocations;
    }
    return malloc(bytes);
}

// User-defined operator delete.
void operator delete [](void *pv) throw()
{
    free(pv);
}
