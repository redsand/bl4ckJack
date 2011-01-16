
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

#include <RCF/ServerStub.hpp>

#include <iterator>

#include <RCF/RcfClient.hpp>

namespace RCF {

    void ServerStub::invoke(
        const std::string &         subInterface,
        int                         fnId,
        RcfSession &                session)
    {
        // no mutex here, since there is never anyone writing to mInvokeFunctorMap

        RCF_VERIFY(
            mInvokeFunctorMap.find(subInterface) != mInvokeFunctorMap.end(),
            Exception(_RcfError_UnknownInterface(subInterface)))
            (subInterface)(fnId)(mInvokeFunctorMap.size())(mMergedStubs.size());

        mInvokeFunctorMap[subInterface](fnId, session);
    }

    void ServerStub::merge(RcfClientPtr rcfClientPtr)
    {
        InvokeFunctorMap &invokeFunctorMap =
            rcfClientPtr->getServerStub().mInvokeFunctorMap;

        std::copy(
            invokeFunctorMap.begin(),
            invokeFunctorMap.end(),
            std::insert_iterator<InvokeFunctorMap>(
                mInvokeFunctorMap,
                mInvokeFunctorMap.begin()));

        invokeFunctorMap.clear();

        mMergedStubs.push_back(rcfClientPtr);
    }

} // namespace RCF
