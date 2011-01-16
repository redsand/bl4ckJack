
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

#include <RCF/Random.hpp>

#include <sys/timeb.h>
#include <sys/types.h>

#include <stdlib.h>
#include <time.h>

#include <RCF/InitDeinit.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    Mutex * gpRandMutex = NULL;

    int generateRandomNumber()
    {
        Lock lock(*gpRandMutex); RCF_UNUSED_VARIABLE(lock);
        return rand();
    }

    RCF_ON_INIT_DEINIT_NAMED( 
        gpRandMutex = new Mutex; ,
        delete gpRandMutex; gpRandMutex = NULL, 
        generateRandomNumberInit)

}
