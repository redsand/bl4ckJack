
#include <iostream>
#include <sstream>
#include <string>

#include <RCF/test/TestMinimal.hpp>

#include <RCF/Idl.hpp>
#include <RCF/PingBackService.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/test/TransportFactories.hpp>
#include <RCF/util/CommandLine.hpp>
#include <RCF/util/Platform/OS/Sleep.hpp>

#include <RCF/UdpEndpoint.hpp>

#include <SF/memory.hpp>

namespace Test_ClientTimeout {

    RCF_BEGIN(I_X, "I_X")
        RCF_METHOD_R2(std::string, echo, std::string, unsigned int)
        RCF_METHOD_R0(boost::uint32_t, getPingTimestamp)
    RCF_END(I_X);

    class X
    {
    public:
        std::string echo(const std::string &s, unsigned int sec)
        {
            Platform::OS::Sleep(sec);
            return s;
        }

        boost::uint32_t getPingTimestamp()
        {
            return RCF::getCurrentRcfSession().getPingTimestamp();
        }
    };

    class RestoreClientTransportGuard
    {
    public:

        RestoreClientTransportGuard(RCF::ClientStub &client, RCF::ClientStub &clientTemp) :
          mClient(client),
              mClientTemp(clientTemp)
          {}

          ~RestoreClientTransportGuard()
          {
              RCF_DTOR_BEGIN
                  mClient.setTransport(mClientTemp.releaseTransport());
              RCF_DTOR_END
          }

    private:
        RCF::ClientStub &mClient;
        RCF::ClientStub &mClientTemp;
    };

    void testTimerHeap()
    {
        RCF::Heap<int> timerHeap;

        int Numbers[] = {
            1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        };

        std::size_t count = sizeof(Numbers) / sizeof(Numbers[0]);

        std::random_shuffle(Numbers, Numbers + count);

        for (std::size_t i = 0; i < count ; ++i)
        {
            timerHeap.add( Numbers[i] );
        }

        // Removing from the top until it's empty, using top() and pop().
        for (std::size_t i = count; i >= 1; --i)
        {
            RCF_CHECK_EQ(timerHeap.size() , i);
            int top = timerHeap.top();
            RCF_CHECK_EQ(top , i );
            timerHeap.pop();
        }
        
        RCF_CHECK( timerHeap.empty() );

        // Removing from the top until it's empty, using remove().
        for (std::size_t i = 0; i<count ; ++i)
        {
            timerHeap.add( Numbers[i] );
        }

        for (std::size_t i = count; i >= 1; --i)
        {
            RCF_CHECK_EQ(timerHeap.size() , i);
            RCF_CHECK_EQ(timerHeap.top() , i);
            timerHeap.remove(i);
        }

        RCF_CHECK( timerHeap.empty() );

        // Removing random elements until it's half empty, using remove().
        for (std::size_t i = 0; i<count ; ++i)
        {
            timerHeap.add( Numbers[i] );
        }

        for (std::size_t i = 0; i < count/2; ++i)
        {
            RCF_CHECK_EQ(timerHeap.size() , count - i);
            timerHeap.remove( Numbers[i] );
        }

        // Pull remaining elements off the top and check that they are ordered.
        int previousVal = 0;
        while ( !timerHeap.empty() )
        {
            int val = timerHeap.top();
            timerHeap.pop();

            RCF_CHECK_EQ(previousVal , 0 || previousVal > val );
            previousVal = val;
        }
    }

} // namespace Test_ClientTimeout

int test_main(int argc, char **argv)
{

    Platform::OS::BsdSockets::disableBrokenPipeSignals();

    printTestHeader(__FILE__);

    RCF::RcfInitDeinit rcfInitDeinit;

    using namespace Test_ClientTimeout;

    util::CommandLine::getSingleton().parse(argc, argv);

    testTimerHeap();

#if defined(BOOST_WINDOWS) && defined(SIO_UDP_CONNRESET)
    {
        int port = 50002;

        RCF::UdpEndpoint ep(port);
        RcfClient<I_X> client(ep);
        client.getClientStub().setRemoteCallTimeoutMs(2*1000);
        boost::uint32_t t0 = RCF::getCurrentTimeMs();
        try
        {
            client.echo("", 0);
            RCF_CHECK_FAIL();
        }
        catch(const RCF::Exception & e)
        {
            RCF_CHECK_OK();
        }
        boost::uint32_t t1 = RCF::getCurrentTimeMs();
        RCF_CHECK_GT(t1 - t0, 1800);

        RCF::RcfServer server(ep);

        X x;
        server.bind( (I_X *) NULL, x);
        server.start();

        std::string s0 = "asdf";
        std::string s1 = client.echo(s0, 0);
        RCF_CHECK_EQ(s0 , s1);
    }
#endif

    for (int i=0; i<RCF::getTransportFactories().size(); ++i)
    {
        RCF::TransportFactoryPtr transportFactoryPtr = RCF::getTransportFactories()[i];
        RCF::TransportPair transports = transportFactoryPtr->createTransports();
        RCF::ServerTransportPtr serverTransportPtr( transports.first );
        RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );

        if (!transportFactoryPtr->isConnectionOriented())
        {
            continue;
        }

        std::cout << transportFactoryPtr->desc() << std::endl;

        RCF::RcfServer server( serverTransportPtr );

        RCF::PingBackServicePtr pbsPtr( new RCF::PingBackService() );
        server.addService(pbsPtr);

        // Multi threaded server is necessary for ping backs.
        server.setThreadPool( RCF::ThreadPoolPtr( new RCF::ThreadPool(5)) );

        server.start();

        X x;
        server.bind( (I_X *) 0, x);

        RcfClient<I_X> client( clientTransportAutoPtr );
        client.echo("", 0);

        // Test pings.

        boost::uint32_t pingTimeStamp = client.getPingTimestamp();
        RCF_CHECK_EQ(pingTimeStamp , 0 );

        boost::uint32_t t0 = Platform::OS::getCurrentTimeMs();
        client.getClientStub().ping();
        boost::uint32_t t1 = Platform::OS::getCurrentTimeMs();
        
        pingTimeStamp = client.getPingTimestamp();

        RCF_CHECK( t0 <= pingTimeStamp && pingTimeStamp <= t1);


        // Test ping backs.

        // Ping backs enabled.
        client.getClientStub().setPingBackIntervalMs(1000);
        RCF_CHECK_EQ(client.getClientStub().getPingBackCount() , 0 );
        client.echo("", 3);
        RCF_CHECK_GT( client.getClientStub().getPingBackCount(), 1 );

        // Ping backs disabled.
        client.getClientStub().setPingBackIntervalMs(0);
        client.echo("", 2);
        RCF_CHECK_EQ(client.getClientStub().getPingBackCount() , 0 );

        // Ping back service not added.
        server.removeService(pbsPtr);
        client.getClientStub().setPingBackIntervalMs(2000);

        try
        {
            client.echo("", 2);
        }
        catch(RCF::RemoteException & e)
        {
            RCF_CHECK_EQ(e.getErrorId() , RCF::RcfError_NoPingBackService);
        }

        // Ping back service added.
        server.addService(pbsPtr);
        client.echo("", 2);
        RCF_CHECK_GT( client.getClientStub().getPingBackCount(), 0 );


        try
        {
            // Pingbacks won't come back more frequently than once a second,
            // so this should cause the remote call to fail immediately.
            client.getClientStub().setPingBackIntervalMs(100);
            client.echo("", 2);
            RCF_CHECK_FAIL();
        }
        catch(const RCF::Exception & e)
        {
            RCF_CHECK_OK();
            RCF_CHECK_EQ(e.getErrorId() , RCF::RcfError_PingBackTimeout);
        }

        // NB: the above call results in an immediate exception, but there will
        // be a server thread held up in X::echo for 2 seconds. For the next call
        // to succeed, without a delay, we need to have configured at least 3 
        // threads in the server (two for parallel calls and one for pingbacks).

        // Requesting ping backs from an older server.
        server.setRuntimeVersion(4);
        client.getClientStub().setRuntimeVersion(4);
        try
        {
            client.echo("", 2);
        }
        catch(const RCF::RemoteException & re)
        {
            RCF_CHECK_FAIL();
            std::cout << re.what() << std::endl;
        }

        RCF_CHECK_EQ(client.getClientStub().getPingBackCount() , 0 );

        server.setRuntimeVersion( RCF::getDefaultRuntimeVersion() );

    }

    for (int i=0; i<RCF::getTransportFactories().size(); ++i)
    {
        RCF::TransportFactoryPtr transportFactoryPtr = RCF::getTransportFactories()[i];
        RCF::TransportPair transports = transportFactoryPtr->createTransports();
        RCF::ServerTransportPtr serverTransportPtr( transports.first );
        RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );

        if (!transportFactoryPtr->isConnectionOriented())
        {
            continue;
        }

        std::cout << transportFactoryPtr->desc() << std::endl;

        X x;
        RCF::RcfServer server(serverTransportPtr);
        server.bind( (I_X *) 0, x);
        server.start();

        RcfClient<I_X> client(clientTransportAutoPtr);

        // this call will timeout
        client.getClientStub().setRemoteCallTimeoutMs(1*1000);
        try
        {
            std::string s = client.echo("abc", 2);
            RCF_CHECK_FAIL();
        }
        catch (const RCF::Exception &e)
        {
            RCF_CHECK_OK();
            std::cout << e.what();
        }
       
        // this one won't
        client.getClientStub().setRemoteCallTimeoutMs(10*1000);
        std::string s = client.echo("def", 2);

        // Whether or not this check succeeds depends on the client transport 
        // implementation. A UDP transport might well read the response of the
        // previous call.
        if (transportFactoryPtr->isConnectionOriented())
        {
            RCF_CHECK_EQ(s , "def");
        }

        // Following tests are too dependent on platform-local configuration.
        
        // test connection timeout

        server.stop();

        client.getClientStub().setRemoteCallTimeoutMs(15*1000);
        client.getClientStub().setConnectTimeoutMs(2*1000);
        unsigned int t0 = RCF::getCurrentTimeMs();
        try
        {
            std::string x = client.echo("asdf", 10);
            RCF_CHECK_FAIL();
        }
        catch (const RCF::Exception &e)
        {
            RCF_CHECK_OK();
        }
        unsigned int t1 = RCF::getCurrentTimeMs();
        std::cout << "t1-t0 = " << t1-t0 << std::endl;
        RCF_CHECK_LT(t1-t0, 4*1000);

        server.start();

        client.getClientStub().setRemoteCallTimeoutMs(15*1000);
        client.getClientStub().setConnectTimeoutMs(2*1000);
        t0 = RCF::getCurrentTimeMs();
        try
        {
            client.echo("asdf", 5);
            RCF_CHECK_OK();
        }
        catch (const RCF::Exception &e)
        {
            RCF_CHECK_FAIL();
        }
        t1 = RCF::getCurrentTimeMs();
        std::cout << "t1-t0 = " << t1-t0 << std::endl;
        RCF_CHECK_GTEQ(t1-t0, 4900); // leave 100ms margin for OS timer discrepancies...
    }
   
    return 0;
}
