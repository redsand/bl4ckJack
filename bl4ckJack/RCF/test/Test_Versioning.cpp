
#include <boost/bind.hpp>

#include <RCF/test/TestMinimal.hpp>

#include <RCF/Idl.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/TcpEndpoint.hpp>
#include <RCF/UdpEndpoint.hpp>
#include <RCF/Version.hpp>
#include <RCF/test/TransportFactories.hpp>
#include <RCF/util/CommandLine.hpp>

#include <SF/deque.hpp>
#include <SF/list.hpp>
#include <SF/map.hpp>
#include <SF/set.hpp>
#include <SF/vector.hpp>

#ifdef RCF_USE_BOOST_SERIALIZATION
#include <boost/serialization/deque.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#endif

namespace Test_Versioning {

    enum MyEnum { One, Two, Three };

    typedef std::vector<std::pair<int,int> >    VectorPairIntInt;
    typedef std::map<int,int>                   MapIntInt;
    typedef boost::shared_ptr<std::string>      StringPtr;

    class Echo
    {
    public:
        std::string echo(const std::string &s)
        {
            return s;
        }

        std::string echo(const std::string &s, boost::uint32_t version)
        {
            int sessionVersion = RCF::getCurrentRcfSession().getRuntimeVersion();
            RCF_CHECK_EQ(sessionVersion , version);
            return s;
        }

        std::string echo(const std::string &s, boost::uint32_t runtimeVersion, boost::uint32_t archiveVersion)
        {
            RCF::RcfSession & session = RCF::getCurrentRcfSession();
            boost::uint32_t sessionRuntimeVersion = session.getRuntimeVersion();
            boost::uint32_t sessionArchiveVersion = session.getArchiveVersion();
            RCF_CHECK_EQ(sessionRuntimeVersion, runtimeVersion);
            RCF_CHECK_EQ(sessionArchiveVersion, archiveVersion);
            return s;
        }

        void echo(const RCF::ByteBuffer & bufferIn, RCF::ByteBuffer & bufferOut, boost::uint32_t version)
        {
            boost::uint32_t sessionVersion = RCF::getCurrentRcfSession().getRuntimeVersion();
            RCF_CHECK_EQ(sessionVersion , version);
            bufferOut = bufferIn;
        }

        MyEnum                      f1(MyEnum myEnum)                       { return myEnum; }
        int                         f1(int n)                               { return n; }

        std::vector<std::string>    f2(const std::vector<std::string> & v)  { return v; }
        std::list<std::string>      f2(const std::list<std::string> & v)    { return v; }
        
        std::vector<std::string>    f3(const std::vector<std::string> & v)  { return v; }
        std::set<std::string>       f3(const std::set<std::string> & v)     { return v; }
        
        std::vector<std::string>    f4(const std::vector<std::string> & v)  { return v; }
        std::deque<std::string>     f4(const std::deque<std::string> & v)   { return v; }
        
        std::map<int, int>          f5(const std::map<int, int> & v)        { return v; }
        VectorPairIntInt            f5(const VectorPairIntInt & v)          { return v; }

        std::string                 f6(const std::string & s)               { return s; }

        std::string                 f7(const std::string & s)               { return s; }
        std::string                 f7(std::string * s)                     { return s ? *s : NULL ; }

        std::string                 f8(const std::string & s)               { return s; }
        StringPtr                   f8(StringPtr s)                         { return s; }

        std::string                 f9(std::string s)                       { return s; }
        StringPtr                   f9(StringPtr s)                         { return s; }

        std::vector<char>           f10(const std::vector<char> & s)        { return s; }
        std::string                 f10(const std::string & s)              { return s; }

        std::vector<char>           f11(const std::vector<char> & s)        { return s; }
        RCF::ByteBuffer             f11(RCF::ByteBuffer s)                  { return s; }
    };

    RCF_BEGIN(I_Echo, "I_Echo")
        RCF_METHOD_R2(std::string, echo, const std::string &, boost::uint32_t)
        RCF_METHOD_R3(std::string, echo, const std::string &, boost::uint32_t, boost::uint32_t)
        RCF_METHOD_V3(void, echo, const RCF::ByteBuffer &, RCF::ByteBuffer &, boost::uint32_t)
    RCF_END(I_Echo)

    RCF_BEGIN(I_EchoA, "I_Echo")
        RCF_METHOD_R2(std::string, echo, const std::string &, int)

        RCF_METHOD_R1(int,                      f1, int)
        RCF_METHOD_R1(std::vector<std::string>, f2, const std::vector<std::string> &)
        RCF_METHOD_R1(std::vector<std::string>, f3, const std::vector<std::string> &)
        RCF_METHOD_R1(std::vector<std::string>, f4, const std::vector<std::string> &)
        RCF_METHOD_R1(MapIntInt,                f5, const MapIntInt &)

        RCF_METHOD_R1(std::string,              f6, const std::string &)
        RCF_METHOD_R1(std::string,              f7, const std::string &)
        RCF_METHOD_R1(std::string,              f8, const std::string &)        
        RCF_METHOD_R1(std::string,              f9, std::string)

        RCF_METHOD_R1(std::vector<char>,        f10, const std::vector<char> &)
        RCF_METHOD_R1(std::vector<char>,        f11, const std::vector<char> &)

    RCF_END(I_EchoA)

    RCF_BEGIN(I_EchoB, "I_Echo")
        RCF_METHOD_R2(std::string, echo, const std::string &, int)

        RCF_METHOD_R1(MyEnum,                   f1, MyEnum)
        RCF_METHOD_R1(std::list<std::string>,   f2, const std::list<std::string> &)
        RCF_METHOD_R1(std::set<std::string>,    f3, const std::set<std::string> &)
        RCF_METHOD_R1(std::deque<std::string>,  f4, const std::deque<std::string> &)
        RCF_METHOD_R1(VectorPairIntInt,         f5, const VectorPairIntInt &)

        RCF_METHOD_R1(std::string,              f6, std::string)
        RCF_METHOD_R1(std::string,              f7, std::string *)
        RCF_METHOD_R1(StringPtr,                f8, StringPtr)
        RCF_METHOD_R1(StringPtr,                f9, StringPtr)

        RCF_METHOD_R1(std::string,              f10, const std::string &)
        RCF_METHOD_R1(RCF::ByteBuffer,          f11, const RCF::ByteBuffer &)

    RCF_END(I_EchoB)

} // namespace Test_Minimal

BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(Test_Versioning::MyEnum)

// TODO: make this into ByteBuffer::operator==()
bool equals(const RCF::ByteBuffer & lhs, const RCF::ByteBuffer & rhs)
{
    if (lhs.isEmpty() && rhs.isEmpty())
    {
        return true;
    }
    else if (!lhs.isEmpty() && !rhs.isEmpty())
    {
        return 
            lhs.getLength() == rhs.getLength()
            && 0 == memcmp(lhs.getPtr(), rhs.getPtr(), lhs.getLength());
    }
    else
    {
        return false;
    }
}

void testClientServerVersioning(
    RCF::TcpEndpoint tcpEndpoint, 
    RCF::RcfServer &server)
{
    using namespace Test_Versioning;

    std::string s0 = "something special";

    Echo echo;
    server.bind( (I_Echo*) 0, echo);
    server.bind( (I_EchoA*) 0, echo);
    server.bind( (I_EchoB*) 0, echo);

    RcfClient<I_Echo> client(tcpEndpoint);

    boost::uint32_t versionCurrent = RCF::getDefaultRuntimeVersion();

    std::string s1 = client.echo(s0, versionCurrent);
    RCF_CHECK_EQ(s1 , s0);
    RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionCurrent);

    for (boost::uint32_t versionOld = 1; versionOld <= versionCurrent; ++versionOld)
    {
        for (boost::uint32_t versionNew = versionOld; versionNew <= versionCurrent; ++versionNew)
        {
            // 1. client old, server new
            client.getClientStub().setRuntimeVersion(versionOld);
            server.setRuntimeVersion(versionNew);

            s1 = client.echo(s0, versionOld);
            RCF_CHECK_EQ(s1, s0);
            RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionOld);

            // 2. client new, server old
            client.getClientStub().setRuntimeVersion(versionNew);
            server.setRuntimeVersion(versionOld);
            RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionNew);
            s1 = client.echo(s0, versionOld);
            RCF_CHECK_EQ(s1, s0);
            RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionOld);

            client.getClientStub().setAutoVersioning(false);
            client.getClientStub().setRuntimeVersion(versionNew);
            try
            {
                client.echo(s0, versionNew);
                RCF_CHECK_EQ(versionOld , versionNew );
            }
            catch(const RCF::VersioningException &e)
            {
                RCF_CHECK_NEQ(versionOld , versionNew);
                RCF_CHECK_EQ(e.getRuntimeVersion() , versionOld);
                RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionNew);
                client.getClientStub().setRuntimeVersion(versionOld);
                s1 = client.echo(s0, versionOld);
                RCF_CHECK_EQ(s1, s0);
                RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionOld);
            }
            client.getClientStub().setAutoVersioning(true);

            // 3. client old, server old
            client.getClientStub().setRuntimeVersion(versionOld);
            server.setRuntimeVersion(versionOld);
            s1 = client.echo(s0, versionOld);
            RCF_CHECK_EQ(s1, s0);
            RCF_CHECK_EQ(client.getClientStub().getRuntimeVersion() , versionOld);
        }
    }

    // Check automatic archive versioning.

    server.bind( (I_Echo*) 0, echo);

    boost::uint32_t runtimeVersion = RCF::getDefaultRuntimeVersion();
    
    client.getClientStub().setAutoVersioning(true);
    client.getClientStub().disconnect();

    client.getClientStub().setRuntimeVersion(runtimeVersion);
    server.setRuntimeVersion(runtimeVersion);

    // Server reverting to older archive version.
    client.getClientStub().setArchiveVersion(5);
    server.setArchiveVersion(7);
    client.echo(s0, runtimeVersion, 5);

    // Client reverting to older archive version.
    client.getClientStub().setArchiveVersion(7);
    server.setArchiveVersion(5);
    client.echo(s0, runtimeVersion, 5);
}

void testSerializationVersioning( 
    RCF::TcpEndpoint tcpEndpoint, 
    RCF::RcfServer &server)
{
    using namespace Test_Versioning;

    int f1_arg_a = 17;
    MyEnum f1_arg_b = Two;

    std::vector<std::string> f2_arg_a;
    f2_arg_a.push_back("a");
    f2_arg_a.push_back("b");

    std::list<std::string> f2_arg_b;
    f2_arg_b.push_back("a");
    f2_arg_b.push_back("b");

    std::vector<std::string> & f3_arg_a = f2_arg_a;

    std::set<std::string> f3_arg_b;
    f3_arg_b.insert("a");
    f3_arg_b.insert("b");

    std::vector<std::string> & f4_arg_a = f2_arg_a;

    std::deque<std::string> f4_arg_b;
    f4_arg_b.push_back("a");
    f4_arg_b.push_back("b");

    std::map<int, int> f5_arg_a;
    f5_arg_a[1] = 2;
    f5_arg_a[3] = 4;

    std::vector<std::pair<int, int> > f5_arg_b;
    f5_arg_b.push_back( std::make_pair( int(1), int(2)) );
    f5_arg_b.push_back( std::make_pair( int(3), int(4)) );

    std::string f6_arg_a = "a";
    std::string f6_arg_b = "a";

    std::string f7_arg_a = "a";
    std::string f7_arg_b_ = "a";
    std::string * f7_arg_b = & f7_arg_b_;

    std::string f8_arg_a = "a";
    boost::shared_ptr<std::string> f8_arg_b( new std::string("a") );

    std::string f9_arg_a = "a";
    boost::shared_ptr<std::string> f9_arg_b( new std::string("a") );

    char * f10_arg_a_ = "asdf";
    std::vector<char> f10_arg_a(f10_arg_a_, f10_arg_a_ + strlen(f10_arg_a_));
    std::string f10_arg_b(f10_arg_a_, f10_arg_a_ + strlen(f10_arg_a_));

    char * f11_arg_a_ = "asdf";
    std::vector<char> f11_arg_a(f11_arg_a_, f11_arg_a_ + strlen(f11_arg_a_));
    RCF::ByteBuffer f11_arg_b( f11_arg_a );

    RcfClient<I_EchoA> clientA(tcpEndpoint);
    RcfClient<I_EchoB> clientB(tcpEndpoint);

    clientA.getClientStub().setAutoVersioning(false);
    clientB.getClientStub().setAutoVersioning(false);

    boost::uint32_t versionCurrent = RCF::getDefaultRuntimeVersion();

    Echo echo;

    // Expected OK: clients A and B calling an A server.
    server.bind( (I_EchoA*) 0, echo);

    RCF_CHECK(clientA.f1(f1_arg_a) == f1_arg_a);
    RCF_CHECK(clientB.f1(f1_arg_b) == f1_arg_b);

    RCF_CHECK(clientA.f2(f2_arg_a) == f2_arg_a );
    RCF_CHECK(clientB.f2(f2_arg_b) == f2_arg_b );

    RCF_CHECK(clientA.f3(f3_arg_a) == f3_arg_a );
    RCF_CHECK(clientB.f3(f3_arg_b) == f3_arg_b );

    RCF_CHECK(clientA.f4(f4_arg_a) == f4_arg_a );
    RCF_CHECK(clientB.f4(f4_arg_b) == f4_arg_b );

    RCF_CHECK(clientA.f5(f5_arg_a) == f5_arg_a );
    RCF_CHECK(clientB.f5(f5_arg_b) == f5_arg_b );

    RCF_CHECK(clientA.f6(f6_arg_a) == f6_arg_a );
    RCF_CHECK(clientB.f6(f6_arg_b) == f6_arg_b );

    RCF_CHECK(clientA.f7(f7_arg_a) == f7_arg_a );
    RCF_CHECK(clientB.f7(f7_arg_b) == *f7_arg_b );

    RCF_CHECK(clientA.f8(f8_arg_a) == f8_arg_a );
    RCF_CHECK(*(clientB.f8(f8_arg_b).get()) == *f8_arg_b );

    RCF_CHECK(clientA.f9(f9_arg_a) == f9_arg_a );
    RCF_CHECK(*(clientB.f9(f9_arg_b).get()) == *f9_arg_b );

    RCF_CHECK(clientA.f10(f10_arg_a) == f10_arg_a );
    RCF_CHECK(clientB.f10(f10_arg_b) == f10_arg_b );

    RCF_CHECK(clientA.f11(f11_arg_a) == f11_arg_a );
    RCF_CHECK(clientB.f11(f11_arg_b) == f11_arg_b );

    // Expected failure: enum-int compatibility, client B calling an A server on RCF runtime version 1
    clientB.getClientStub().setRuntimeVersion(1);
    try 
    {
        clientB.f1(f1_arg_b);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::RemoteException &e)
    {
        RCF_CHECK_OK();
        clientB.getClientStub().disconnect();
        clientB.getClientStub().setRuntimeVersion(2);
        clientB.f1(f1_arg_b);
    }

    // Expected failure: ByteBuffer-vector compatibility, client B calling an A server on RCF runtime version 3
    clientB.getClientStub().setRuntimeVersion(3);
    try 
    {
        clientB.f11(f11_arg_b);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::Exception &e)
    {
        RCF_CHECK_OK();
        clientB.getClientStub().disconnect();
        clientB.getClientStub().setRuntimeVersion(4);
        clientB.f11(f11_arg_b);
    }

    clientB.getClientStub().setRuntimeVersion(versionCurrent);

    // Expected failure: server deserializing a reference from a null pointer.
    try
    {
        boost::shared_ptr<std::string> sps;
        clientB.f8(sps);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::RemoteException &e)
    {
        RCF_CHECK_OK();
        RCF_CHECK_EQ(e.getErrorId() , RCF::RcfError_DeserializationNullPointer );
    }

    // Expected failure: server deserializing a value from a null pointer.
    try
    {
        boost::shared_ptr<std::string> sps;
        clientB.f9(sps);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::RemoteException &e)
    {
        RCF_CHECK_OK();
        RCF_CHECK_EQ(e.getErrorId() , RCF::RcfError_DeserializationNullPointer );
    }


    // Expected OK: clients A and B calling a B server.
    server.bind( (I_EchoB*) 0, echo);

    RCF_CHECK(clientA.f1(f1_arg_a) == f1_arg_a);
    RCF_CHECK(clientB.f1(f1_arg_b) == f1_arg_b);

    RCF_CHECK(clientA.f2(f2_arg_a) == f2_arg_a );
    RCF_CHECK(clientB.f2(f2_arg_b) == f2_arg_b );

    RCF_CHECK(clientA.f3(f3_arg_a) == f3_arg_a );
    RCF_CHECK(clientB.f3(f3_arg_b) == f3_arg_b );

    RCF_CHECK(clientA.f4(f4_arg_a) == f4_arg_a );
    RCF_CHECK(clientB.f4(f4_arg_b) == f4_arg_b );

    RCF_CHECK(clientA.f5(f5_arg_a) == f5_arg_a );
    RCF_CHECK(clientB.f5(f5_arg_b) == f5_arg_b );

    RCF_CHECK(clientA.f6(f6_arg_a) == f6_arg_a );
    RCF_CHECK(clientB.f6(f6_arg_b) == f6_arg_b );

    RCF_CHECK(clientA.f7(f7_arg_a) == f7_arg_a );
    RCF_CHECK(clientB.f7(f7_arg_b) == *f7_arg_b );

    RCF_CHECK(clientA.f8(f8_arg_a) == f8_arg_a );
    RCF_CHECK(*(clientB.f8(f8_arg_b).get()) == *f8_arg_b );

    RCF_CHECK(clientA.f9(f9_arg_a) == f9_arg_a );
    RCF_CHECK(*(clientB.f9(f9_arg_b).get()) == *f9_arg_b );

    RCF_CHECK(clientA.f10(f10_arg_a) == f10_arg_a );
    RCF_CHECK(clientB.f10(f10_arg_b) == f10_arg_b );

    RCF_CHECK(clientA.f11(f11_arg_a) == f11_arg_a );
    RCF_CHECK( equals(clientB.f11(f11_arg_b), f11_arg_b) );

    // Expected failure: enum-int compatibility, client A calling a B server on RCF runtime version 1
    clientA.getClientStub().setRuntimeVersion(1);
    try 
    {
        clientA.f1(f1_arg_a);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::RemoteException &e)
    {
        RCF_CHECK_OK();
        clientA.getClientStub().disconnect();
        clientA.getClientStub().setRuntimeVersion(2);
        clientA.f1(f1_arg_a);
    }

    // Expected failure: ByteBuffer-vector compatibility, client A calling a B server on RCF runtime version 3
    clientA.getClientStub().setRuntimeVersion(3);
    try 
    {
        clientA.f11(f11_arg_a);
        RCF_CHECK_FAIL();
    }
    catch (const RCF::Exception &e)
    {
        RCF_CHECK_OK();
        clientA.getClientStub().disconnect();
        clientA.getClientStub().setRuntimeVersion(4);
        clientA.f11(f11_arg_a);
    }

    clientA.getClientStub().setRuntimeVersion(versionCurrent);

    {
        // Check ByteBuffer serialization in runtime version 2.

        Echo echo;
        server.bind( (I_Echo *) 0, echo);

        int runtimeVersion = 2;
        RcfClient<I_Echo> client(tcpEndpoint);
        client.getClientStub().setRuntimeVersion(runtimeVersion);

        RCF::ByteBuffer bufferIn;
        RCF::ByteBuffer bufferOut;
        client.getClientStub().setRemoteCallTimeoutMs(1000*3600);
        client.echo(bufferIn, bufferOut, runtimeVersion);
        RCF_CHECK(bufferOut == bufferIn);

    }
}

namespace Test_Versioning {

    RCF_BEGIN(I_X1, "I_X")

        RCF_METHOD_R0(int,          f1)
        RCF_METHOD_R1(int,          f2, int)
        RCF_METHOD_R1(std::string,  f3, const std::string &)

        RCF_METHOD_V1(void,         f4_v, int)
        RCF_METHOD_R1(int,          f5, int)

    RCF_END(I_X)
    
    RCF_BEGIN(I_X2, "I_X")

        RCF_METHOD_R1(int,          f1, int)
        RCF_METHOD_R2(int,          f2, int, int)
        RCF_METHOD_R2(std::string,  f3, const std::string &, const std::string &)

        RCF_METHOD_R1(int,          f4_r, int)
        RCF_METHOD_R2(int,          f5, int, int &)

    RCF_END(I_X2)

    class X
    {
    public:
        int     f1()                    { return 0; }
        int     f1(int n1)              { return n1; }

        int     f2(int n1)              { return n1; }
        int     f2(int n1, int n2)      { return n1 + n2; }

        void    f4_v(int n1)            {}
        int     f4_r(int n1)            { return n1; }

        int     f5(int n1)              { return n1; }
        int     f5(int n1, int & n2)    { n2 = n1; return n1; }

        std::string f3(const std::string & s1)                          { return s1; }
        std::string f3(const std::string & s1, const std::string & s2)  { return s1 + s2; }
    };

};

void testInterfaceVersioning( 
    RCF::TcpEndpoint tcpEndpoint, 
    RCF::RcfServer &server)
{
    using namespace Test_Versioning;

    RcfClient<I_X1> client1(tcpEndpoint);
    RcfClient<I_X2> client2(tcpEndpoint);

    X x;

    std::string s1 = "a";
    std::string s2 = "b";
    
    int n2 = 0;

    int nRet = 0;
    std::string sRet;

    server.bind( (I_X1 *) 0, x);

    // Adding in parameters.
    nRet = client1.f1(); RCF_CHECK_EQ(0 , nRet);
    nRet = client2.f1(1); RCF_CHECK_EQ(0 , nRet);
    nRet = client1.f2(1); RCF_CHECK_EQ(1 , nRet);
    nRet = client2.f2(1,2); RCF_CHECK_EQ(1 , nRet);
    sRet = client1.f3(s1); RCF_CHECK_EQ(s1 , sRet);
    sRet = client2.f3(s1,s2); RCF_CHECK_EQ(s1 , sRet);

    // Adding out parameters.
    client1.f4_v(1);
    nRet = client2.f4_r(1); RCF_CHECK_EQ(0 , nRet);

    n2 = 100;
    nRet = client1.f5(1); RCF_CHECK_EQ(1 , nRet );
    nRet = client2.f5(1, n2); RCF_CHECK(1 == nRet && 100 == n2 )(nRet)(n2);

    server.bind( (I_X2 *) 0, x);

    // Removing in parameters.
    nRet = client1.f1(); RCF_CHECK_EQ(0 , nRet );
    nRet = client2.f1(1); RCF_CHECK_EQ(1 , nRet );
    nRet = client1.f2(1); RCF_CHECK_EQ(1 , nRet );
    nRet = client2.f2(1,2); RCF_CHECK_EQ(3 , nRet );
    sRet = client1.f3(s1); RCF_CHECK_EQ(s1 , sRet );
    sRet = client2.f3(s1,s2); RCF_CHECK_EQ(s1+s2 , sRet );

    // Removing out parameters.
    client1.f4_v(1);
    nRet = client2.f4_r(1); RCF_CHECK_EQ(1 , nRet );

    n2 = 100;
    nRet = client1.f5(1); RCF_CHECK_EQ(1 , nRet );
    nRet = client2.f5(1, n2); RCF_CHECK(1 == nRet && 1 == n2 )(nRet)(n2);
}

void testOnewayVersioning()
{
    using namespace Test_Versioning;

    Echo echo;
    RCF::RcfServer server( RCF::UdpEndpoint(0) );
    server.bind( (I_Echo *) 0, echo);
    server.start();

    int port = server.getIpServerTransport().getPort();

    RCF::UdpEndpoint udpEndpoint(port);
    RcfClient<I_Echo> client( udpEndpoint );

    std::string s0 = "asdf";
    boost::uint32_t versionCurrent = RCF::getDefaultRuntimeVersion();
    for(boost::uint32_t version = 1; version < versionCurrent; ++version)
    {
        client.getClientStub().setRuntimeVersion(version);
        RCF_CHECK_EQ(s0 , client.echo(s0, version) );
    }
}

namespace Test_Versioning {

    class Z
    {
    public:
        Z() : mArchiveVersion(-1)
        {
        }

        void serialize(SF::Archive & ar)
        {
            mArchiveVersion = ar.getArchiveVersion();
        }

        template<typename Archive>
        void serialize(Archive & ar, const unsigned int)
        {
            // Just a dummy for BSer so it compiles.
        }

        int mArchiveVersion;
    };

    RCF_BEGIN(I_Y, "I_Y")
        RCF_METHOD_R2(Z, echo, const Z &, int)
    RCF_END(I_Y)

    class Y
    {
    public:
        Z echo(const Z & z, int expectedArchiveVersion)
        {
            RCF_CHECK_EQ(z.mArchiveVersion , expectedArchiveVersion);
            return Z();
        }
    };

};

void testArchiveVersioning()
{
    using namespace Test_Versioning;

    Y y;
    RCF::RcfServer server( RCF::TcpEndpoint(0) );
    server.bind( (I_Y *) 0, y);
    server.start();

    int port = server.getIpServerTransport().getPort();

    RCF::TcpEndpoint tcpEndpoint(port);
    RcfClient<I_Y> client( tcpEndpoint );

    for (int i = 0; i<15; ++i)
    {
        client.getClientStub().setArchiveVersion(i);

        Z z1;
        Z z2 = client.echo(z1, i);

        RCF_CHECK_EQ(z1.mArchiveVersion , i );
        RCF_CHECK_EQ(z2.mArchiveVersion , i );
    }
}

int test_main(int argc, char **argv)
{
    printTestHeader(__FILE__);

    RCF::RcfInitDeinit rcfInitDeinit;

    using namespace Test_Versioning;

    util::CommandLine::getSingleton().parse(argc, argv);

    RCF::TransportPair transportPair = RCF::TcpTransportFactory().createTransports();
    RCF::ServerTransportPtr serverTransportPtr = transportPair.first;
    RCF::ClientTransportAutoPtr clientTransportAutoPtr = *transportPair.second;
    
    RCF::RcfServer server(serverTransportPtr);
    server.start();

    RCF::TcpEndpoint tcpEndpoint( server.getIpServerTransport().getPort() );

    testClientServerVersioning(tcpEndpoint, server);

    testSerializationVersioning(tcpEndpoint, server);

    testInterfaceVersioning(tcpEndpoint, server);

    testOnewayVersioning();

    testArchiveVersioning();

    return 0;
}
