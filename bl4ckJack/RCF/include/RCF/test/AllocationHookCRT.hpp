
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

#include <crtdbg.h>
#include <iostream>

#include <RCF/test/Test.hpp>

#ifdef NDEBUG
#error CRT allocator hook only possible in debug builds.
#endif

bool gInstrumented = false;
bool gExpectAllocations = true;
std::size_t gnAllocations = 0;

static _CRT_ALLOC_HOOK  pfnOldCrtAllocHook  = NULL;

int crtAllocationHook(
    int allocType, 
    void    *userData, 
    size_t size, 
    int blockType, 
    long    requestNumber, 
    const unsigned char *filename, // Can't be UNICODE
    int lineNumber)
{
    if (    gInstrumented
        &&  (allocType == _HOOK_ALLOC || allocType == _HOOK_REALLOC)
        && !gExpectAllocations)
    {
        // Only flag the first unexpected allocation, so we don't end up 
        // with thousands of failures.

        gInstrumented = false;
        RCF_CHECK(gExpectAllocations);
        std::cout << "Unexpected memory allocation." << std::endl;

        // If we do want to track further allocations, uncomment this.
        //gInstrumented = true;

    }

    if (allocType == _HOOK_ALLOC || allocType == _HOOK_REALLOC)
    {
        ++gnAllocations;
    }

    return pfnOldCrtAllocHook(
        allocType, 
        userData, 
        size, 
        blockType, 
        requestNumber, 
        filename, 
        lineNumber);
}

void setupHook()
{
    pfnOldCrtAllocHook = _CrtSetAllocHook(crtAllocationHook);
}

bool dummy = (setupHook(), false);
