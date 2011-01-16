
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

#ifndef INCLUDE_RCF_AMD_HPP
#define INCLUDE_RCF_AMD_HPP

#include <boost/any.hpp>

#include <RCF/Export.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ThreadLocalData.hpp>

// Temporary hack for now
#include <RCF/TcpIocpServerTransport.hpp>

namespace RCF {

    class I_Parameters;

    class RCF_EXPORT AmdImpl
    {
    public:

        AmdImpl()
        {
            mRcfSessionPtr = RCF::getCurrentRcfSessionPtr()->shared_from_this();
            mRcfSessionPtr->mAutoSend = false;

            mpParametersUntyped = mRcfSessionPtr->mpParameters;

            IocpSessionState & sessionState = 
                dynamic_cast<IocpSessionState &>(
                    mRcfSessionPtr->getSessionState());

            mSessionStatePtr = sessionState.sharedFromThis();
        }

        void commit()
        {
            mRcfSessionPtr->sendResponse();
            mpParametersUntyped = NULL;
            mRcfSessionPtr.reset();
            mSentry = boost::any();

            mSessionStatePtr.reset();
        }

        void commit(const std::exception &e)
        {
            mRcfSessionPtr->sendResponseException(e);
            mpParametersUntyped = NULL;
            mRcfSessionPtr.reset();
            mSentry = boost::any();

            mSessionStatePtr.reset();
        }

    private:
        boost::any          mSentry;
        RcfSessionPtr       mRcfSessionPtr;
        
        // Temporary hack to keep the session state alive...
        IocpSessionStatePtr mSessionStatePtr;

    protected:
        I_Parameters *      mpParametersUntyped;

    };

    template<
        typename R, 
        typename A1  = Void,
        typename A2  = Void,
        typename A3  = Void,
        typename A4  = Void,
        typename A5  = Void,
        typename A6  = Void,
        typename A7  = Void,
        typename A8  = Void,
        typename A9  = Void,
        typename A10 = Void,
        typename A11 = Void,
        typename A12 = Void,
        typename A13 = Void,
        typename A14 = Void,
        typename A15 = Void>
    class Amd : public AmdImpl
    {
    public:

        typedef ServerParameters<
            R, 
            A1, A2, A3, A4, A5, A6, A7, A8,
            A9, A10, A11, A12, A13, A14, A15> ParametersT;

        Amd()
        {
            RCF_ASSERT( dynamic_cast<ParametersT *>(mpParametersUntyped) );
        }

        ParametersT &parameters()
        {
            return * static_cast<ParametersT *>(mpParametersUntyped);;
        }
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_AMD_HPP


