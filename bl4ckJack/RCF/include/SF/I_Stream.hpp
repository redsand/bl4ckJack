
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

#ifndef INCLUDE_SF_I_STREAM_HPP
#define INCLUDE_SF_I_STREAM_HPP

#include <typeinfo>

#include <RCF/Export.hpp>

#include <SF/PortableTypes.hpp>

namespace SF {

    //*************************************************************************
    // Stream interfaces

    class DataPtr;
    class Node;

    typedef std::pair<void *, const std::type_info *> ObjectId;

    class RCF_EXPORT I_Encoding
    {
    public:
        virtual ~I_Encoding() 
        {}

        virtual UInt32 getCount(
            DataPtr &               data, 
            const std::type_info &  type) = 0;

        virtual void toData(
            DataPtr &               data, 
            void *                  pvObject, 
            const std::type_info &  type, 
            int                     nCount) = 0;

        virtual void toObject(
            DataPtr &               data, 
            void *                  pvObject, 
            const std::type_info &  type, 
            int                     nCount) = 0;
    };

}

#endif // ! INCLUDE_SF_I_STREAM_HPP
