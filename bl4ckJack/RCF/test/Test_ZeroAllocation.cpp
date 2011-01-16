
#include <string>
#include <strstream>

#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/version.hpp>

#include <RCF/test/TestMinimal.hpp>

#include <RCF/AsyncFilter.hpp>
#include <RCF/FilterService.hpp>
#include <RCF/Idl.hpp>
#include <RCF/ObjectPool.hpp>
#include <RCF/OpenSslEncryptionFilter.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/ZlibCompressionFilter.hpp>

#include <RCF/test/TransportFactories.hpp>
#include <RCF/util/CommandLine.hpp>
#include <RCF/util/Profile.hpp>

#include <SF/AdlWorkaround.hpp>

#ifdef BOOST_WINDOWS
#include <RCF/SspiFilter.hpp>
#endif

#ifdef RCF_USE_BOOST_ASIO
#include <RCF/AsioServerTransport.hpp>
#endif

#ifdef RCF_USE_PROTOBUF
#include "protobuf/Person.pb.h"
#include "protobuf/Person.pb.cc"

#endif

#if defined(_MSC_VER) && !defined(NDEBUG)
// For MSVC debug builds, we use CRT allocation hooks, which will catch all forms of allocations.
#include <RCF/test/AllocationHookCRT.hpp>
#else
// Otherwise we override global operator new and friends.
#include <RCF/test/AllocationHook.hpp>
#endif

namespace Test_ZeroAllocation {

    class ContainsByteBuffer
    {
    public:
        RCF::ByteBuffer mByteBuffer;

        void serialize(SF::Archive &archive)
        {
            archive & mByteBuffer;
        }

        template<typename Archive>
        void serialize(Archive &, const unsigned int)
        {
            RCF_ASSERT(0);
        }
    };

    class Echo
    {
    public:
        std::string echo(const std::string &s)
        {
            return s;
        }

        RCF::ByteBuffer echo(RCF::ByteBuffer byteBuffer1, const std::string &s, RCF::ByteBuffer byteBuffer2)
        {
            void *pv1 = byteBuffer1.getPtr() ;
            std::size_t pvlen1 = byteBuffer1.getLength() ;

            void *pv2 = byteBuffer2.getPtr() ;
            std::size_t pvlen2 = byteBuffer2.getLength() ;

            return byteBuffer2;
        }

        RCF::ByteBuffer echo(RCF::ByteBuffer byteBuffer)
        {
            return byteBuffer;
        }

        ContainsByteBuffer echo(ContainsByteBuffer c)
        {
            return c;
        }

        void echo(std::string & s0, const std::string & s1, std::string & s2)
        {
            RCF_CHECK(s2.empty());

            s0 = s1;
            s2 = s1;
        }

        std::string echo(std::string s0, std::string & s1, const std::string & s2, std::string & s3)
        {
            return s0;
        }

#ifdef RCF_USE_PROTOBUF

        void echo(Person & s0, const Person & s1, Person & s2)
        {
            s0 = s1;
            s2 = s1;
        }

#endif

    };

    

} // namespace Test_ZeroAllocation

RCF_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(Test_ZeroAllocation::ContainsByteBuffer)

namespace Test_ZeroAllocation {

    RCF_BEGIN(I_Echo, "I_Echo")
        RCF_METHOD_R1(std::string, echo, const std::string &)
        RCF_METHOD_R3(RCF::ByteBuffer, echo, RCF::ByteBuffer, std::string, RCF::ByteBuffer)
        RCF_METHOD_R1(RCF::ByteBuffer, echo, RCF::ByteBuffer)
        RCF_METHOD_R1(ContainsByteBuffer, echo, ContainsByteBuffer)

        RCF_METHOD_V3(void, echo, std::string &, const std::string &, RCF::Out<std::string &>)

        RCF_METHOD_R4(std::string, echo, std::string, std::string &, const std::string &, RCF::Out<std::string &>)

#ifdef RCF_USE_PROTOBUF
        RCF_METHOD_V3(void, echo, Person &, const Person &, RCF::Out<Person &>)
#endif

    RCF_END(I_Echo)

} // namespace Test_ZeroAllocation

#ifdef RCF_USE_PROTOBUF

bool isEqual(const google::protobuf::Message & lhs, const google::protobuf::Message & rhs)
{
    std::ostringstream os1;
    bool ok1 = lhs.SerializeToOstream(&os1);
    std::string s1 = os1.str();

    std::ostringstream os2;
    bool ok2 = rhs.SerializeToOstream(&os2);
    std::string s2 = os2.str();

    return ok1 && ok2 && s1 == s2;
}

#endif

void clearString(std::string * pStr)
{
    // VC6 doesn't have std::string::clear().
    *pStr = "";
}

int test_main(int argc, char **argv)
{
    gInstrumented = true;

    printTestHeader(__FILE__);

    if (BOOST_VERSION <= 103301)
    {
        std::cout 
            << "Boost version is 1.33.1 or earlier. Zero-allocation on critical paths "
            << "requires allocator support in boost::shared_ptr, which is only available "
            << "in Boost 1.34.0 and later."
            << std::endl;

        return 0;
    }

    RCF::RcfInitDeinit rcfInitDeinit;

    using namespace Test_ZeroAllocation;

    util::CommandLineOption<std::string>    clScert("scert", RCF_TEMP_DIR "ssCert2.pem", "OpenSSL server certificate");
    util::CommandLineOption<std::string>    clSpwd("spwd", "mt2316", "OpenSSL server certificate password");
    util::CommandLineOption<std::string>    clCcert("ccert", RCF_TEMP_DIR "ssCert1.pem", "OpenSSL client certificate");
    util::CommandLineOption<std::string>    clCpwd("cpwd", "mt2316", "OpenSSL client certificate password");
    util::CommandLine::getSingleton().parse(argc, argv);

#if defined(BOOST_WINDOWS) && defined(__MINGW32__) && __GNUC__ == 3 && __GNUC_MINOR__ == 4
    RCF_CHECK_EQ(1 , 0 && "Zero-allocation not working in gcc 3.4 for unknown reasons");
    return 0;
#endif

    for (unsigned int i=0; i<RCF::getTransportFactories().size(); ++i)
    {
        RCF::TransportFactoryPtr transportFactoryPtr( RCF::getTransportFactories()[i] );
        RCF::TransportPair transports = transportFactoryPtr->createTransports();
        RCF::ServerTransportPtr serverTransportPtr( transports.first );
        RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );
        bool transportFiltersSupported = transportFactoryPtr->supportsTransportFilters();

        std::cout << transportFactoryPtr->desc() << std::endl;
        std::string transportDesc = "Transport " + boost::lexical_cast<std::string>(i) + ": ";

        serverTransportPtr->setMaxMessageLength(-1);
        clientTransportAutoPtr->setMaxMessageLength(-1);

#ifdef RCF_USE_BOOST_ASIO

        RCF::AsioServerTransport * pAsioTransport = 
            dynamic_cast<RCF::AsioServerTransport *>(serverTransportPtr.get());

        if (pAsioTransport)
        {
            if (BOOST_VERSION <= 104200)
            {
                // Timer implementation in older asio versions causes ongoing allocations.
                // Eventually we should remove the deadline timer the asio cycle() function.

                std::cout << "This version of Boost.Asio will not pass zero allocation test." << std::endl;
                continue;
            }
        }

#endif

        Echo echo;
        RCF::RcfServer server(serverTransportPtr);
        server.bind( (I_Echo*) 0, echo);

        RCF::FilterServicePtr filterServicePtr(new RCF::FilterService());
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::IdentityFilterFactory()));
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::XorFilterFactory()));
#ifdef RCF_USE_ZLIB
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::ZlibStatefulCompressionFilterFactory()));
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::ZlibStatelessCompressionFilterFactory()));
#endif
#ifdef RCF_USE_OPENSSL
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::OpenSslEncryptionFilterFactory(clScert, clSpwd)));
#endif
#ifdef BOOST_WINDOWS
        filterServicePtr->addFilterFactory( RCF::FilterFactoryPtr( new RCF::NtlmFilterFactory()));
#endif
        server.addService(filterServicePtr);
       
        server.start();
       
        // make sure all allocations have taken place
        Platform::OS::SleepMs(1000);

        RcfClient<I_Echo> client(clientTransportAutoPtr->clone());
       
        client.getClientStub().setRemoteCallTimeoutMs(1000*60*60);

        {
            std::size_t nAllocations = gnAllocations;
            std::auto_ptr<int> apn(new int(17));
            apn.reset();
            RCF_CHECK_NEQ(gnAllocations , nAllocations);
        }

        {
            std::string s = "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdf";
            RCF::ByteBuffer byteBuffer0( (char*) s.c_str(), s.length());

            std::vector<RCF::FilterPtr> filters;

            // with no transport or payload filters

            // prime the pump
            client.echo(byteBuffer0);
            Platform::OS::SleepMs(1000);

            {
                std::string s0 = byteBuffer0.string();
                gExpectAllocations = false;
                RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                gExpectAllocations = true;

                std::string s1 = byteBuffer1.string();
                RCF_CHECK_EQ(s0 , s1);
            }

            RCF::MarshalingProtocol mp = client.getClientStub().getMarshalingProtocol();

            client.getClientStub().setMarshalingProtocol(RCF::Mp_Rcf);
            client.echo(byteBuffer0);

            {
                util::Profile profile(transportDesc + "1000 calls, RCF marshaling, no dynamic allocations");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }

#ifdef RCF_USE_PROTOBUF

            client.getClientStub().setMarshalingProtocol(RCF::Mp_Protobuf);
            client.echo(byteBuffer0);

            {
                util::Profile profile(transportDesc + "1000 calls, Protobuf marshaling, no dynamic allocations");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }

#endif // RCF_USE_PROTOBUF

            client.getClientStub().setMarshalingProtocol(mp);

            {
                // Check that return value, value, cref, ref and outref marshaling don't cause allocations.

                std::string s0;
                std::string s1;
                std::string s2;
                std::string s3;
                std::string s4;

                s4 = client.echo(s0, s1, s2, s3);
                Sleep(1000);

                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    s4 = client.echo(s0, s1, s2, s3);

                    RCF_CHECK(s0.empty() && s1.empty() && s2.empty() && s3.empty() && s4.empty());
                }
                gExpectAllocations = true;

            }

            // Can't compile calls to ObjectPool::enableCaching<>, with vc6.
#if !defined(_MSC_VER) || _MSC_VER > 1200
            // Test custom object allocators.
            {
                RCF::ObjectPool & pool = RCF::getObjectPool();

                // Enable server side caching of std::string.
                pool.enableCaching<std::string>( 
                    10, 
                    boost::bind(&clearString, _1));

                std::string s0;
                std::string s1;
                std::string s2;


                // VC9 has std::string internal buffer of 15. We'd like to be
                //  a little bigger than that.
                s0 = "";
                s1 = "123456781234567812345678";
                s2 = "bla";

                // First time through.
                client.echo(s0, s1, s2);
                Sleep(1000);

                {
                    util::Profile profile(transportDesc + "1000 calls, RCF marshaling, custom std::string allocator, no dynamic allocations");

                    gExpectAllocations = false;
                    for(unsigned int i=0; i<1000; ++i)
                    {
                        s0 = "";
                        s1 = "123456781234567812345678";
                        s2 = "bla";

                        client.echo(s0, s1, s2);

                        RCF_CHECK(s0 == s1 && s2 == s1);
                    }
                    gExpectAllocations = true;
                }

                // Disable server side caching of std::string.
                pool.disableCaching<std::string>();
            }
#endif

#ifdef RCF_USE_PROTOBUF

            // Test custom object allocators with Protocol Buffers.
            {
                // First loop - without object caching.
                // Second loop - with object caching.
                for (int j=0; j<2; ++j)
                {
                    if (j==1)
                    {
                        // Configure server side caching of Person.
                        RCF::ObjectPool & pool = RCF::getObjectPool();
                        RCF::WriteLock lock(pool.mObjPoolMutex);

                        RCF::TypeInfo ti( typeid(Person) );
                        pool.mObjPool[ti].reset( new RCF::ObjectPool::ObjList() );
                        pool.mObjPool[ti]->mMaxSize = 10;

                        pool.mObjPool[ti]->mOps.reset( 
                            new RCF::ObjectPool::Ops<Person>(
                            boost::bind(&Person::Clear, _1)) );
                    }

                    Person p0;
                    Person p1;
                    Person p2;

                    p0.set_id(0);
                    p0.set_name("");
                    p0.set_email("");

                    p1.set_id(1234);
                    p1.set_name("The archbishop of Canterbury.");
                    p1.set_email("archbishop@canterbury.com");

                    p2.set_id(0);
                    p2.set_name("");
                    p2.set_email("");

                    // First time through.
                    client.echo(p0, p1, p2);
                    RCF_CHECK(isEqual(p0, p1) && isEqual(p2, p1));

                    Sleep(1000);

                    {
                        util::Profile profile(transportDesc + "1000 calls, RCF marshaling, custom Protobuf Person allocator, no dynamic allocations");

                        bool shouldExpectAllocations = (j == 0); 
                        gExpectAllocations = shouldExpectAllocations;
                        
                        for(unsigned int k=0; k<1000; ++k)
                        {

                            p0.set_id(0);
                            p0.set_name("");
                            p0.set_email("");

                            p1.set_id(1234);
                            p1.set_name("The archbishop of Canterbury.");
                            p1.set_email("archbishop@canterbury.com");

                            p2.set_id(0);
                            p2.set_name("");
                            p2.set_email("");


                            client.echo(p0, p1, p2);

                            gExpectAllocations = true;
                            RCF_CHECK(isEqual(p0, p1) && isEqual(p2, p1));
                            gExpectAllocations = shouldExpectAllocations;
                        }
                        gExpectAllocations = true;
                    }

                    if (j == 1)
                    {
                        // Disable server side caching of Person.
                        RCF::ObjectPool & pool = RCF::getObjectPool();
                        RCF::WriteLock lock(pool.mObjPoolMutex);

                        RCF::TypeInfo ti( typeid(Person) );
                        pool.mObjPool[ti]->mMaxSize = 0;
                        pool.mObjPool[ti]->clear();
                    }
                }
            }

#endif

            // with both transport and payload filters

            if (transportFiltersSupported)
            {
                filters.clear();
                filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
                filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
                filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
                client.getClientStub().requestTransportFilters(filters);
            }

            filters.clear();
            filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
            filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
            filters.push_back( RCF::FilterPtr( new RCF::XorFilter()));
            client.getClientStub().setMessageFilters(filters);

            // prime the pump
            client.echo(byteBuffer0);
            Platform::OS::SleepMs(1000);

            for (int i=0; i<3; ++i)
            {
                // byteBuffer0 will be transformed in place, so we need to be a bit careful with the before/after comparison.
                // s0 and s1 will change on each pass.

                std::string s0 = byteBuffer0.string();
                gExpectAllocations = false;
                RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                gExpectAllocations = true;
                std::string s1 = byteBuffer1.string();
                RCF_CHECK_EQ(s0 , s1);
            }

            {
                util::Profile profile(transportDesc + "1000 calls, no dynamic allocations, 3 transport filters + 3 payload filters");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }
           
            filters.clear();
#ifdef RCF_USE_ZLIB
            filters.push_back( RCF::FilterPtr( new RCF::ZlibStatelessCompressionFilter()));
#endif
            client.getClientStub().setMessageFilters(filters);
            client.echo(byteBuffer0);

            {
                util::Profile profile(transportDesc + "1000 calls, no dynamic allocations, <zlib stateless> payload filters");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }
           

            if (transportFiltersSupported)
            {
               
                filters.clear();
#ifdef RCF_USE_ZLIB
                filters.push_back( RCF::FilterPtr( new RCF::ZlibStatefulCompressionFilter()));
#endif
#ifdef RCF_USE_OPENSSL
                filters.push_back( RCF::FilterPtr( new RCF::OpenSslEncryptionFilter(clCcert, clCpwd)));
#endif
                client.getClientStub().requestTransportFilters(filters);
               
                client.echo(byteBuffer0);

                util::Profile profile(transportDesc + "1000 calls, <zlib stateful><OpenSSL> transport filter");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }

#ifdef BOOST_WINDOWS

            if (transportFiltersSupported)
            {
                filters.clear();
#ifdef RCF_USE_ZLIB
                filters.push_back( RCF::FilterPtr( new RCF::ZlibStatefulCompressionFilter()));
#endif
                filters.push_back( RCF::FilterPtr( new RCF::NtlmFilter()));
                client.getClientStub().requestTransportFilters(filters);
                client.echo(byteBuffer0);

                util::Profile profile(transportDesc + "1000 calls, <zlib stateful><sspi ntlm> transport filter");
                gExpectAllocations = false;
                for(unsigned int i=0; i<1000; ++i)
                {
                    RCF::ByteBuffer byteBuffer1 = client.echo(byteBuffer0);
                }
                gExpectAllocations = true;
            }

#endif

            client.getClientStub().clearTransportFilters();

            {
                // try serialization (as opposed to marshaling) of ByteBuffer
                ContainsByteBuffer c1;
                c1.mByteBuffer = byteBuffer0;
                ContainsByteBuffer c2 = client.echo(c1);

                gExpectAllocations = false;
                c2.mByteBuffer.clear();
                c2 = client.echo(c1);
                gExpectAllocations = true;
            }

            // ByteBuffer serialization only supported for SF. This test crashes when run with Boost.Serialization.
            if (false)
            {
                // try serialization (not marshaling) of ByteBuffer with all serialization protocols
                for(int protocol=1; protocol<10; ++protocol)
                {
                    RCF::SerializationProtocol sp = RCF::SerializationProtocol(protocol);
                    if (RCF::isSerializationProtocolSupported(sp))
                    {
                        client.getClientStub().setSerializationProtocol(sp);

                        ContainsByteBuffer c1;
                        c1.mByteBuffer = byteBuffer0;
                        ContainsByteBuffer c2 = client.echo(c1);
                        RCF_CHECK_EQ(c2.mByteBuffer.getLength() , c1.mByteBuffer.getLength());

                        // will get memory allocations here when using boost serialization
                        c2.mByteBuffer.clear();
                        c2 = client.echo(c1);
                        RCF_CHECK_EQ(c2.mByteBuffer.getLength() , c1.mByteBuffer.getLength());
                    }
                }
            }
        }
        server.stop();
    }

    return 0;
}
