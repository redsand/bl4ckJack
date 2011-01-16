
#include <RCF/test/TestMinimal.hpp>

#include <RCF/RCF.hpp>
#include <RCF/FilterService.hpp>
#include <RCF/OpenSslEncryptionFilter.hpp>
#include <RCF/Schannel.hpp>
#include <RCF/SspiFilter.hpp>

#include <RCF/util/AutoBuild.hpp>

RCF_BEGIN(I_X, "I_X")
RCF_METHOD_R1(std::string, echo, const std::string &)
RCF_METHOD_V0(void, echoUserData)
RCF_END(I_X)

class X
{
public:
    std::string echo(const std::string & s)
    {
        return s;
    }

    void echoUserData()
    {
        RCF::RcfSession & session = RCF::getCurrentRcfSession();
        std::string userData = session.getRequestUserData();
        session.setResponseUserData(userData);
    }
};

void myCallback(RCF::SchannelFilter & schannelFilter, const RCF::tstring & target)
{
    PCCERT_CONTEXT pRemoteCert = schannelFilter.getServerCertificate();

    // Build the certificate chain.
    PCCERT_CHAIN_CONTEXT     pChainContext = NULL;

    LPSTR rgszUsages[] = 
    {  
        szOID_PKIX_KP_SERVER_AUTH,
        szOID_SERVER_GATED_CRYPTO,
        szOID_SGC_NETSCAPE 
    };

    DWORD cUsages = sizeof(rgszUsages) / sizeof(LPSTR);

    CERT_CHAIN_PARA          ChainPara;

    ZeroMemory(&ChainPara, sizeof(ChainPara));
    ChainPara.cbSize = sizeof(ChainPara);
    ChainPara.RequestedUsage.dwType = USAGE_MATCH_TYPE_OR;
    ChainPara.RequestedUsage.Usage.cUsageIdentifier     = cUsages;
    ChainPara.RequestedUsage.Usage.rgpszUsageIdentifier = rgszUsages;

    BOOL bRet = CertGetCertificateChain(
        NULL,
        pRemoteCert,
        NULL,
        pRemoteCert->hCertStore,
        &ChainPara,
        0,
        NULL,
        &pChainContext);

    DWORD dwErr = GetLastError();

    RCF_VERIFY(
        bRet, 
        RCF::Exception(
            RCF::_RcfError_Sspi(), 
            dwErr, 
            RCF::RcfSubsystem_Os, 
            "CertGetCertificateChain() failed"));

    boost::shared_ptr<const CERT_CHAIN_CONTEXT> chainContext(
        pChainContext, 
        CertFreeCertificateChain);

    // Validate the certificate chain.
    DWORD dwCertFlags = 0;
    std::wstring serverName;
    serverName.assign(target.begin(), target.end());

    HTTPSPolicyCallbackData  polHttps;
    CERT_CHAIN_POLICY_PARA   PolicyPara;
    CERT_CHAIN_POLICY_STATUS PolicyStatus;

    ZeroMemory(&polHttps, sizeof(HTTPSPolicyCallbackData));
    polHttps.cbStruct           = sizeof(HTTPSPolicyCallbackData);
    polHttps.dwAuthType         = AUTHTYPE_SERVER;
    polHttps.fdwChecks          = dwCertFlags;
    polHttps.pwszServerName     = (WCHAR *) serverName.c_str();

    memset(&PolicyPara, 0, sizeof(PolicyPara));
    PolicyPara.cbSize            = sizeof(PolicyPara);
    PolicyPara.pvExtraPolicyPara = &polHttps;

    memset(&PolicyStatus, 0, sizeof(PolicyStatus));
    PolicyStatus.cbSize = sizeof(PolicyStatus);

    bRet = CertVerifyCertificateChainPolicy(
        CERT_CHAIN_POLICY_SSL,
        pChainContext,
        &PolicyPara,
        &PolicyStatus);

    dwErr = GetLastError();

    RCF_VERIFY(
        bRet, 
        RCF::Exception(
            RCF::_RcfError_Sspi(), 
            dwErr, 
            RCF::RcfSubsystem_Os, 
            "CertVerifyCertificateChainPolicy() failed"));
    
    if (PolicyStatus.dwError)
    {
        // We typically go in here because the CA certificate isn't installed in
        // the Trusted Root Certificates store.

        RCF_VERIFY(
            PolicyStatus.dwError == CERT_E_UNTRUSTEDROOT,
            RCF::Exception("Unexpected certificate validation result."))
            (PolicyStatus.dwError);
    }
}

int test_main(int argc, char ** argv)
{
    RCF::RcfInitDeinit initDeinit;

    util::CommandLine::getSingleton().parse(argc, argv);

    RCF::tstring certAName = RCF_T("localhost");
    RCF::tstring caCertAName = RCF_T("RCF CA A");

    RCF::PfxCertificate caCertA(
        RCF_TEMP_DIR "caCertA.p12", 
        RCF_T(""), 
        caCertAName, 
        CERT_FIND_SUBJECT_STR);

    RCF::RcfServer server( RCF::TcpEndpoint(0) );

    RCF::FilterServicePtr fsPtr( new RCF::FilterService());

    fsPtr->addFilterFactory( RCF::FilterFactoryPtr( 
        new RCF::NtlmFilterFactory() ));


    // Test that certificate handle outlives the certificate class.
    {
        RCF::PfxCertificate certA(
            RCF_TEMP_DIR "certA.p12", 
            RCF_T(""), 
            certAName, 
            CERT_FIND_SUBJECT_STR);

        fsPtr->addFilterFactory( RCF::FilterFactoryPtr( 
            new RCF::SchannelFilterFactory(certA.getCertContext()) ));
    }

    server.addService(fsPtr);
    
    server.start();

    int port = server.getIpServerTransport().getPort();
    RCF::TcpEndpoint ep(port);

    X x;
    server.bind( (I_X *) 0, x);

    typedef boost::shared_ptr<RCF::SchannelFilter> SchannelFilterPtr;

    // Remove the CA from the local machine Root store, in case it was already there.
    RCF::StoreCertificate(
        CERT_SYSTEM_STORE_LOCAL_MACHINE, 
        "Root", 
        RCF_T("RCF CA A"), 
        CERT_FIND_SUBJECT_STR).removeFromStore();

    caCertA.addToStore(CERT_SYSTEM_STORE_LOCAL_MACHINE, "Root");

    {
        RcfClient<I_X> client(ep);

        SchannelFilterPtr filterPtr( new RCF::SchannelFilter() );
        
        filterPtr->setManualCertValidation( boost::bind(myCallback, _1, certAName) );

        {
            // If the client wants to authenticate itself to the server as well...
            RCF::PfxCertificate certA(
                RCF_TEMP_DIR "certA.p12", 
                RCF_T(""), 
                certAName, 
                CERT_FIND_SUBJECT_STR);

            filterPtr->setClientCertificate(certA.getCertContext());
        }

        client.getClientStub().requestTransportFilters(filterPtr);

        std::string s0 = "asdf";
        std::string s1 = client.echo(s0);
        RCF_CHECK_EQ(s0 , s1);

        s1 = client.echo(s0);

        for (int i=0; i<50; ++i)
        {
            s1 = client.echo(s0);
        }

        // Test oneway calls.
        for (int i=0; i<50; ++i)
        {
            client.echo(RCF::Oneway, s0);
        }
        s1 = client.echo(s0+s0);
        RCF_CHECK_EQ(s1 , s0+s0);

        // Test large messages.
        server.getServerTransport().setMaxMessageLength(1024*1024);
        client.getClientStub().getTransport().setMaxMessageLength(1024*1024);

        s0.clear();
        std::size_t len = 70000;
        for (std::size_t i=0; i<len; ++i)
        {
            s0 += 'a';
        }
        s1 = client.echo(s0);
        RCF_CHECK_EQ(s1 , s0);
    }

    {
        RcfClient<I_X> client(ep);

        SchannelFilterPtr filterPtr( new RCF::SchannelFilter() );
        filterPtr->setAutoCertValidation( certAName );
        client.getClientStub().requestTransportFilters(filterPtr);

        std::string s0 = "asdf";
        std::string s1 = client.echo(s0);
        RCF_CHECK_EQ(s0 , s1);
    }

    // Remove the CA from the local machine Root store. Now validation should fail.
    RCF::StoreCertificate(
        CERT_SYSTEM_STORE_LOCAL_MACHINE,
        "Root",
        RCF_T("RCF CA A"),
        CERT_FIND_SUBJECT_STR).removeFromStore();

    {
        RcfClient<I_X> client(ep);

        SchannelFilterPtr filterPtr( new RCF::SchannelFilter() );
        filterPtr->setAutoCertValidation( certAName );
        client.getClientStub().requestTransportFilters(filterPtr);

        std::string s0 = "asdf";
        try
        {
            std::string s1 = client.echo(s0);
            RCF_CHECK_EQ(1 , 0);
        }
        catch(const RCF::Exception & e)
        {
            RCF_CHECK_OK();
            RCF_CHECK_EQ(e.getSubSystemError() , SEC_E_UNTRUSTED_ROOT);
        }                
    }

    {
        RcfClient<I_X> client(ep);

        RCF::FilterPtr filterPtr( new RCF::NtlmFilter() );
        client.getClientStub().requestTransportFilters(filterPtr);

        std::string s0 = "asdf";
        std::string s1 = client.echo(s0);       
        RCF_CHECK_EQ(s0 , s1);

        s1 = client.echo(s0);

        for (int i=0; i<50; ++i)
        {
            s1 = client.echo(s0);
        }
    }

#ifdef RCF_USE_OPENSSL

    // OpenSSL <-> Schannel interoperability
    {
        std::vector<int> filterIds;
        filterIds.push_back(RCF::RcfFilter_OpenSsl);
        filterIds.push_back(RCF::RcfFilter_SspiSchannel);

        RcfClient<I_X> client(ep);

        client.getClientStub().setRemoteCallTimeoutMs(1000*3600);
        client.getClientStub().setConnectTimeoutMs(1000*3600);

        // Schannel -> OpenSSL
        RCF::FilterFactoryPtr filterFactoryPtr( 
            new RCF::OpenSslEncryptionFilterFactory(
                RCF_TEMP_DIR "certA.pem",
                ""));
        
        fsPtr->addFilterFactory(filterFactoryPtr, filterIds);

        SchannelFilterPtr schannelFilterPtr( new RCF::SchannelFilter() );
        schannelFilterPtr->setManualCertValidation( boost::bind(myCallback, _1, certAName) );
        client.getClientStub().requestTransportFilters(schannelFilterPtr);

        std::string s0 = "asdf";
        std::string s1 = client.echo(s0);
        RCF_CHECK_EQ(s0 , s1);

        // OpenSSL -> Schannel

        RCF::PfxCertificate certA(
            RCF_TEMP_DIR "certA.p12", 
            RCF_T(""), 
            certAName, 
            CERT_FIND_SUBJECT_STR);

        filterFactoryPtr.reset( 
            new RCF::SchannelFilterFactory(certA.getCertContext()));

        fsPtr->addFilterFactory(filterFactoryPtr, filterIds);

        typedef boost::shared_ptr<RCF::OpenSslEncryptionFilter> OpenSslEncryptionFilterPtr;
        OpenSslEncryptionFilterPtr openSslFilterPtr( new RCF::OpenSslEncryptionFilter(
            "",
            "",
            RCF_TEMP_DIR "caCertA.pem"));

        client.getClientStub().requestTransportFilters(openSslFilterPtr);

        s0 = "asdf";
        s1 = "";
        s1 = client.echo(s0);
        RCF_CHECK_EQ(s0 , s1);
    }

#endif

    // Test passing of custom user data in request headers.
    {
        std::string s0 = "Some user data";
        std::string s1;

        RcfClient<I_X> client(ep);

        client.getClientStub().setRemoteCallTimeoutMs(1000*3600);
        client.getClientStub().setConnectTimeoutMs(1000*3600);

        std::ostringstream os;
        SF::OBinaryStream(os) << s0;

        client.getClientStub().setRequestUserData(os.str());

        client.echoUserData();

        std::istringstream is( client.getClientStub().getResponseUserData() );
        SF::IBinaryStream(is) >> s1;

        RCF_CHECK_EQ(s0 , s1);
    }

    return 0;
}
