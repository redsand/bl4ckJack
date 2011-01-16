
#include <RCF/test/TestMinimal.hpp>

#include <RCF/Idl.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/IpClientTransport.hpp>
#include <RCF/IpServerTransport.hpp>

#include <RCF/test/TransportFactories.hpp>

#include <RCF/util/CommandLine.hpp>
#include <RCF/util/Platform/OS/Sleep.hpp>

#include <RCF/TcpEndpoint.hpp>
#include <RCF/UdpEndpoint.hpp>

namespace Test_IpRestriction {

    RCF_BEGIN(I_X, "I_X")
        RCF_METHOD_V0(void, f);
    RCF_METHOD_R1(std::string, g, const std::string &);
    RCF_END(I_X);

    class X
    {
    public:
        void f() {}
        std::string g(const std::string & s) { return s; }
    };

} // namespace Test_IpRestriction

int test_main(int argc, char **argv)
{
    Platform::OS::BsdSockets::disableBrokenPipeSignals();

    printTestHeader(__FILE__);

    RCF::RcfInitDeinit rcfInitDeinit;
    //RCF::AmiInitDeinit amiInitDeinit;

    using namespace Test_IpRestriction;

    util::CommandLine::getSingleton().parse(argc, argv);

    
    {
    /*
        // Some IPv4/IPv6 tests.

        // TODO: Cover UDP and asio.
        // ...

        RCF::RcfInitDeinit init;
        RCF::AmiInitDeinit amiInitDeinit;

        using RCF::IpAddress;

        {
            IpAddress addr("lindrud", 123);
            RCF_CHECK(!addr.isResolved());
            addr.resolve();
            RCF_CHECK(addr.isResolved());

            std::string s = addr.string();

            addr = IpAddress("15.146.221.3", 123);
            RCF_CHECK(!addr.isResolved());
            addr.resolve();
            RCF_CHECK(addr.isResolved());

            s = addr.string();

            addr = IpAddress();
            RCF_CHECK(addr.empty());
            RCF_CHECK(IpAddress().empty());
        }

        {
            IpAddress ip("15.146.221.3");

            IpAddress ip32("15.146.221.3");
            IpAddress ip24("15.146.221.1");
            IpAddress ip16("15.146.0.1");
            IpAddress ip8("15.1.1.1");
            IpAddress ip0("1.1.1.1");

            RCF_CHECK(ip.matches(ip32, 32));
            RCF_CHECK(ip.matches(ip24, 24));
            RCF_CHECK(ip.matches(ip16, 16));
            RCF_CHECK(ip.matches(ip8, 8));
            RCF_CHECK(ip.matches(ip0, 0));

            RCF_CHECK( ! ip.matches(ip24, 32));
            RCF_CHECK( ! ip.matches(ip16, 24));
            RCF_CHECK( ! ip.matches(ip8, 16));
            RCF_CHECK( ! ip.matches(ip0, 8));
        }
        

        std::vector< std::pair<RCF::TcpEndpoint, RCF::TcpEndpoint> > eps;

        eps.push_back( std::make_pair( 
            RCF::TcpEndpoint("0.0.0.0", 50001), 
            RCF::TcpEndpoint("15.146.221.3", 50001)) );

        eps.push_back( std::make_pair( 
            RCF::TcpEndpoint("::", 50001), 
            RCF::TcpEndpoint("2002:f92:dd03::f92:dd03", 50001)) );

        eps.push_back( std::make_pair( 
            RCF::TcpEndpoint("::", 50001), 
            RCF::TcpEndpoint( RCF::IpAddress("lindrud", 50001, RCF::IpAddress::V6)) ));

        eps.push_back( std::make_pair( 
            RCF::TcpEndpoint("::1", 50001), 
            RCF::TcpEndpoint("::1", 50001)));

        for (std::size_t i=0; i<eps.size(); ++i)
        {
            const RCF::TcpEndpoint & serverEp = eps[i].first;
            X x;
            RCF::RcfServer server(serverEp);
            server.bind<I_X>(x);
            server.start();

            const RCF::TcpEndpoint & clientEp = eps[i].second;
            {
                RcfClient<I_X> client(clientEp);
                std::string s0 = "asdf";
                std::string s1 = client.g(s0);
                RCF_CHECK_EQ(s0 , s1)(serverEp.asString())(clientEp.asString());
            }
            {
                RcfClient<I_X> client(clientEp);
                std::string s0 = "asdf";
                RCF::Future<std::string> fs1 = client.g(s0);
                fs1.wait();
                std::string s1 = *fs1;
                RCF_CHECK_EQ(s0 , s1)(serverEp.asString())(clientEp.asString());
            }
        }

        {
            X x;
            RCF::RcfServer server( RCF::TcpEndpoint(0) );
            server.bind<I_X>(x);
            server.start();

            int port = server.getIpServerTransport().getPort();

            RcfClient<I_X> client(( RCF::TcpEndpoint(port) ));
            std::string s0 = "asdf";
            std::string s1 = client.g(s0);
            RCF_CHECK_EQ(s0 , s1);
        }

        {
            X x;
            RCF::RcfServer server( RCF::UdpEndpoint(50001) );
            server.bind<I_X>(x);
            server.start();

            int port = server.getIpServerTransport().getPort();

            RcfClient<I_X> client(( RCF::UdpEndpoint(port) ));
            std::string s0 = "asdf";
            std::string s1 = client.g(s0);
            RCF_CHECK_EQ(s0 , s1);
        }

        {
            X x;
            RCF::RcfServer server( RCF::UdpEndpoint(0) );
            server.bind<I_X>(x);
            server.start();

            int port = server.getIpServerTransport().getPort();

            RcfClient<I_X> client(( RCF::UdpEndpoint(port) ));
            std::string s0 = "asdf";
            std::string s1 = client.g(s0);
            RCF_CHECK_EQ(s0 , s1);
        }
        */
    }

    for (unsigned int i=0; i<RCF::getIpTransportFactories().size(); ++i)
    {
        RCF::TransportFactoryPtr transportFactoryPtr = RCF::getIpTransportFactories()[i];
        RCF::TransportPair transports;
        RCF::ServerTransportPtr serverTransportPtr;
        RCF::ClientTransportAutoPtr clientTransportAutoPtr;

        transports = transportFactoryPtr->createTransports();
        serverTransportPtr = transports.first;
        clientTransportAutoPtr = *transports.second;

        X x;

        std::cout << transportFactoryPtr->desc() << std::endl;

        {
            // test client restriction

            transports = transportFactoryPtr->createTransports();
            serverTransportPtr = transports.first;
            clientTransportAutoPtr = *transports.second;

            using RCF::IpRule;
            using RCF::IpAddress;
            using RCF::IpAddressV4;
            using RCF::IpAddressV6;

            RCF::RcfServer server(serverTransportPtr);
            server.bind( (I_X*) 0, x);

            // Test IP rule matching.
            {
                RCF::I_IpServerTransport &ipTransport = 
                    dynamic_cast<RCF::I_IpServerTransport &>(server.getServerTransport());

                std::vector<IpRule> ips;

                // 32 significant bits.

                ips.clear();
                ips.push_back( IpRule( IpAddressV4("11.22.33.44"), 32) );
                
                ipTransport.setAllowIps(ips);
                ipTransport.setDenyIps(std::vector<IpRule>());

                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );

                ipTransport.setAllowIps(std::vector<IpRule>());
                ipTransport.setDenyIps(ips);

                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );

                // 24 significant bits.

                ips.clear();
                ips.push_back( IpRule( IpAddressV4("11.22.33.44"), 24) );

                ipTransport.setAllowIps(ips);
                ipTransport.setDenyIps(std::vector<IpRule>());

                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );

                ipTransport.setAllowIps(std::vector<IpRule>());
                ipTransport.setDenyIps(ips);

                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );


                // 16 significant bits.
                ips.clear();
                ips.push_back( IpRule( IpAddressV4("11.22.33.44"), 16) );
                
                ipTransport.setAllowIps(ips);
                ipTransport.setDenyIps(std::vector<IpRule>());

                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );

                ipTransport.setAllowIps(std::vector<IpRule>());
                ipTransport.setDenyIps(ips);

                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );


                // 8 significant bits.
                ips.clear();
                ips.push_back( IpRule( IpAddressV4("11.22.33.44"), 8) );

                ipTransport.setAllowIps(ips);
                ipTransport.setDenyIps(std::vector<IpRule>());

                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );

                ipTransport.setAllowIps(std::vector<IpRule>());
                ipTransport.setDenyIps(ips);

                RCF_CHECK(   ipTransport.isIpAllowed( IpAddress("99.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.99.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.99.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.99") ) );
                RCF_CHECK( ! ipTransport.isIpAllowed( IpAddress("11.22.33.44") ) );
            }

            std::vector<IpRule> ips;
            ips.push_back( IpRule( IpAddressV4("11.22.33.44"), 32) );
            ips.push_back( IpRule( IpAddressV4("44.33.22.11"), 32) );
            ips.push_back( IpRule( IpAddressV4("12.34.56.78"), 32) );

            RCF::I_IpServerTransport &ipTransport = 
                dynamic_cast<RCF::I_IpServerTransport &>(server.getServerTransport());

            ipTransport.setAllowIps(ips);

            server.start();

            Platform::OS::Sleep(1);

            try
            {
                RcfClient<I_X> client(clientTransportAutoPtr->clone());
                client.f();
                RCF_CHECK_FAIL();
            }
            catch(const RCF::Exception &e)
            {
                RCF_CHECK_OK();
            }

            ips = ipTransport.getAllowIps();
            ips.push_back( IpRule( IpAddressV4("127.0.0.1"), 32) );

#ifdef RCF_USE_IPV6
            ips.push_back( IpRule( IpAddressV6("::1"), 128) );
#endif

            ipTransport.setAllowIps(ips);

            RcfClient<I_X> client(clientTransportAutoPtr->clone());
            client.f();
        }

        {
            // Test set and get of ip and port number of a TCP connection.

            transports = transportFactoryPtr->createTransports();
            serverTransportPtr = transports.first;
            clientTransportAutoPtr = *transports.second;

            RCF::RcfServer server(serverTransportPtr);
            server.bind( (I_X*) 0, x);
            server.start();

            RCF::IpAddress::Type ipType = RCF::IpAddress::V4;

            {
                RcfClient<I_X> client( clientTransportAutoPtr->clone() );
                client.f();

                RCF::I_IpClientTransport &ipTransport = client.getClientStub().getIpTransport();
                
                RCF::IpAddress localIp = ipTransport.getAssignedLocalIp();
                RCF_CHECK(!localIp.getIp().empty());
                RCF_CHECK_GT(localIp.getPort() , 0);

                ipType = localIp.getType();

                client.getClientStub().disconnect();
                client.f();

                localIp = ipTransport.getAssignedLocalIp();
                RCF_CHECK(!localIp.getIp().empty());
                RCF_CHECK_GT(localIp.getPort() , 0);
            }


             // This test uses a hard coded port number and thus rather fragile.
#ifdef BOOST_WINDOWS
            {
                RcfClient<I_X> client( clientTransportAutoPtr->clone() );

                RCF::I_IpClientTransport &ipTransport = client.getClientStub().getIpTransport();

                std::string localInterface = 
                    ipType == RCF::IpAddress::V4 ? 
                        "127.0.0.1" : 
                        "::1" ;

                static int idx = 0;
                ++idx;
                int localPort = 50000 + idx;

                ipTransport.setLocalIp( RCF::IpAddress(localInterface, localPort) );

                client.f();

                RCF::IpAddress localIp = ipTransport.getAssignedLocalIp();
                RCF_CHECK_EQ(localIp.getIp() , localInterface);
                RCF_CHECK_EQ(localIp.getPort() , localPort);
            }
#endif
        }

    }

    return 0;
}
