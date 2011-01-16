
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

#ifndef INCLUDE_RCF_INITDEINIT_HPP
#define INCLUDE_RCF_INITDEINIT_HPP

#include <RCF/util/InitDeinit.hpp>

#include <RCF/Export.hpp>

/// RCF_ON_INIT - macro for specifying code that should be run upon RCF initialization
/// \param statements Code (statements) to execute.
#define RCF_ON_INIT(statements)                 UTIL_ON_INIT(statements)

/// RCF_ON_INIT - macro for specifying code that should be run upon RCF deinitialization
/// \param statements Code (statements) to execute.
#define RCF_ON_DEINIT(statements)               UTIL_ON_DEINIT(statements)

/// RCF_ON_INIT - macro for specifying code that should be run upon RCF initialization and deinitialization
/// \param initStatements Code (statements) to execute upon initialization.
/// \param deinitStatements Code (statements) to execute upon deinitialization.
#define RCF_ON_INIT_DEINIT(initStatements,deinitStatements)       UTIL_ON_INIT_DEINIT(initStatements,deinitStatements)

#define RCF_ON_INIT_NAMED(x, name)              UTIL_ON_INIT_NAMED(x, name)
#define RCF_ON_DEINIT_NAMED(x, name)            UTIL_ON_DEINIT_NAMED(x, name)
#define RCF_ON_INIT_DEINIT_NAMED(x,y, name)     UTIL_ON_INIT_DEINIT_NAMED(x,y, name)

namespace RCF {

    /// Initializes RCF framework.
    RCF_EXPORT void init();

    /// Deinitializes RCF framework.
    RCF_EXPORT void deinit();

    class RCF_EXPORT RcfInitDeinit
    {
    public:
        RcfInitDeinit();
        ~RcfInitDeinit();
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_INITDEINIT_HPP
