
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

#ifndef INCLUDE_RCF_SCHANNEL_HPP
#define INCLUDE_RCF_SCHANNEL_HPP

#include <RCF/SspiFilter.hpp>
#include <RCF/util/Tchar.hpp>

#include <schnlsp.h>

// missing stuff in mingw headers
#ifdef __MINGW32__
#ifndef SP_PROT_NONE
#define SP_PROT_NONE                    0
#endif
#endif // __MINGW32__

namespace RCF {

    static const ULONG DefaultSchannelContextRequirements = 
        ASC_REQ_SEQUENCE_DETECT |
        ASC_REQ_REPLAY_DETECT   |
        ASC_REQ_CONFIDENTIALITY |
        ASC_REQ_EXTENDED_ERROR  |
        ASC_REQ_ALLOCATE_MEMORY |
        ASC_REQ_STREAM;

    class RCF_EXPORT SchannelServerFilter : public SspiServerFilter
    {
    public:
        SchannelServerFilter(
            boost::shared_ptr<const CERT_CONTEXT> localCert, 
            DWORD enabledProtocols,
            ULONG contextRequirements);

        const FilterDescription &           getFilterDescription() const;
    };

    typedef boost::shared_ptr<const CERT_CONTEXT> CertContextPtr;

    class RCF_EXPORT SchannelFilterFactory : public FilterFactory
    {
    public:

        SchannelFilterFactory(
            CertContextPtr cert, 
            DWORD enabledProtocols = SP_PROT_TLS1_SERVER,
            ULONG contextRequirements = DefaultSchannelContextRequirements);

        FilterPtr                           createFilter();
        const FilterDescription &           getFilterDescription();

    private:
        CertContextPtr mLocalCert;
        ULONG mContextRequirements;
        DWORD mEnabledProtocols;
    };

    class RCF_EXPORT SchannelClientFilter : public SspiClientFilter
    {
    public:
        SchannelClientFilter(
            DWORD enabledProtocols = SP_PROT_NONE,
            ULONG contextRequirements = DefaultSchannelContextRequirements);

        void                setManualCertValidation(
                                CertValidationCallback certValidationCallback);

        void                setAutoCertValidation(
                                const tstring & serverName);

        void                setClientCertificate(
                                CertContextPtr certContext);

        PCCERT_CONTEXT      getServerCertificate();

        const FilterDescription &           getFilterDescription() const;
    };

    typedef SchannelClientFilter        SchannelFilter;

    // Certificate utility classes.

    class RCF_EXPORT PfxCertificate
    {
    public:

        PfxCertificate(
            const std::string & pathToCert, 
            const tstring & password,
            const tstring & certName,
            DWORD dwFindType);

        ~PfxCertificate();

        void addToStore(
            DWORD dwFlags, 
            const std::string & storeName);

        CertContextPtr getCertContext();

    private:

        HCERTSTORE mPfxStore;
        CertContextPtr mCertContextPtr;
    };

    class RCF_EXPORT StoreCertificate
    {
    public:

        StoreCertificate(
            DWORD dwStoreFlags,
            const std::string & storeName,
            const tstring & certName,
            DWORD dwFindType);

        ~StoreCertificate();

        void removeFromStore();

        CertContextPtr getCertContext();

    private:

        HCERTSTORE mStore;
        CertContextPtr mCertContextPtr;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_SCHANNEL_HPP
