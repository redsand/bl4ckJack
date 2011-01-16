// uncomment to enable VLD leak detection - will automatically link to required libs
//#include "vld.h"
//#include "vldapi.h"

#include <RCF/test/TestMinimal.hpp>

#include <RCF/Idl.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/test/TransportFactories.hpp>
#include <RCF/test/WithStopServer.hpp>
#include <RCF/util/CommandLine.hpp>

namespace Test_Minimal {

    RCF_BEGIN(I_Echo, "I_Echo")
        RCF_METHOD_R1(std::string, echo, const std::string &)
        RCF_METHOD_V0(void, stopServer)
    RCF_END(I_Echo)

    class Echo : public RCF::WithStopServer
    {
        
    private:
#if defined(_MSC_VER) && _MSC_VER == 1200
        friend RcfClient<I_Echo>;
#else
        friend class RcfClient<I_Echo>;
#endif        

        std::string echo(const std::string &s)
        {
            return s;
        }
    };

} // namespace Test_Minimal

int test_main(int argc, char **argv)
{

    printTestHeader(__FILE__);

    RCF::RcfInitDeinit rcfInitDeinit;

    //RCF_CHECK_EQ(1,1);
    //RCF_CHECK_EQ(1,2);
    //RCF_CHECK_NEQ(3,3);
    //
    //RCF_CHECK_LT(3,3);
    //RCF_CHECK_LT(3,4);
    //RCF_CHECK_LT(4,3);

    //RCF_CHECK_LTEQ(3,3);
    //RCF_CHECK_LTEQ(3,4);
    //RCF_CHECK_LTEQ(4,3);

    //RCF_CHECK_GT(3,3);
    //RCF_CHECK_GT(4,3);
    //RCF_CHECK_GT(3,4);

    //RCF_CHECK_GTEQ(3,3);
    //RCF_CHECK_GTEQ(4,3);
    //RCF_CHECK_GTEQ(3,4);

    using namespace Test_Minimal;

    util::CommandLine::getSingleton().parse(argc, argv);

    TEST_CASE("1")
    {
        for (unsigned int i=0; i<RCF::getTransportFactories().size(); ++i)
        {
            RCF::TransportFactoryPtr transportFactoryPtr = RCF::getTransportFactories()[i];
            RCF::TransportPair transports = transportFactoryPtr->createTransports();
            RCF::ServerTransportPtr serverTransportPtr( transports.first );
            RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );

            std::cout << transportFactoryPtr->desc() << std::endl;

            serverTransportPtr->setMaxMessageLength(1000*10);
            clientTransportAutoPtr->setMaxMessageLength(1000*10);

            std::string s0 = "something special";

            Echo echo;

            RCF::RcfServer server( serverTransportPtr );
            server.bind( (I_Echo*) 0, echo);
            server.start();

            // 5 calls on each serialization protocol
            for(int protocol=1; protocol<10; ++protocol)
            {
                RCF::SerializationProtocol sp = RCF::SerializationProtocol(protocol);
                if (RCF::isSerializationProtocolSupported(sp))
                {
                    std::cout
                        << "Serialization protocol: "
                        << RCF::getSerializationProtocolName(sp)
                        << std::endl;

                    RcfClient<I_Echo> client( clientTransportAutoPtr->clone());
                    client.getClientStub().setSerializationProtocol(sp);
                    for (int k=0; k<5; ++k)
                    {
                        std::string s = client.echo(s0);
                        RCF_CHECK_EQ(s0 , s);

                        std::size_t requestSize = client.getClientStub().getTransport().getLastRequestSize();
                        std::size_t responseSize = client.getClientStub().getTransport().getLastResponseSize();

                        RCF_CHECK_GT(requestSize, 0);
                        RCF_CHECK_GT(responseSize, 0);
                    }
                }
            }

            // all calls on the same connection
            RcfClient<I_Echo> client( clientTransportAutoPtr->clone() );

            for (int j=0; j<100; ++j)
            {
                std::string s = client.echo(s0);
                RCF_CHECK_EQ(s0 , s);
            }

            // new connection for each call
            for (int j=0; j<100; ++j)
            {
                std::string s = RcfClient<I_Echo>( clientTransportAutoPtr->clone() ).echo(s0);
                RCF_CHECK_EQ(s0 , s);
            }

            RcfClient<I_Echo>(clientTransportAutoPtr->clone()).stopServer();
        }
    }

    TEST_CASE("2")
    {
        for (unsigned int i=0; i<RCF::getTransportFactories().size(); ++i)
        {
            TEST_CASE(i)
            {
                RCF::TransportFactoryPtr transportFactoryPtr = RCF::getTransportFactories()[i];
                RCF::TransportPair transports = transportFactoryPtr->createTransports();
                RCF::ServerTransportPtr serverTransportPtr( transports.first );
                RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );

                Echo echo;
                RCF::RcfServer server( serverTransportPtr );
                server.bind( (I_Echo*) 0, echo);
                server.start();
                server.stop();
            }
        }
    }

    TEST_CASE("3")
    {
        RCF::RcfServer server( RCF::TcpEndpoint(0));
        Echo echo;
        server.bind( (I_Echo*) 0, echo);
        server.start();

        int port = dynamic_cast<RCF::I_IpServerTransport &>(
            server.getServerTransport()).getPort();
        std::string s0 = "asdf";
        std::string s1 = RcfClient<I_Echo>( RCF::TcpEndpoint(port)).echo(s0);
        RCF_CHECK_EQ(s1 , s0);

        server.stop();
    }

    // Uncomment to see if shared_ptr cycle detection is working
    //struct X { boost::shared_ptr<X> mpx; };
    //boost::shared_ptr<X> px(new X());
    //px->mpx = px;

    return 0;
}
