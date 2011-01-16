
#include <string>
#include <vector>

#include <RCF/test/TestMinimal.hpp>

#include <RCF/Idl.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/ObjectFactoryService.hpp>

#include <RCF/test/TransportFactories.hpp>
#include <RCF/test/ThreadGroup.hpp>
#include <RCF/util/CommandLine.hpp>

#ifdef RCF_USE_BOOST_SERIALIZATION
#include <boost/serialization/vector.hpp>
#endif

#ifdef RCF_USE_SF_SERIALIZATION
#include <SF/vector.hpp>
#endif

namespace Test_General {

    namespace A {

        RCF_BEGIN( Calculator, "Calculator" )
            RCF_METHOD_PLACEHOLDER()
            RCF_METHOD_R2(double, add, double, double)
            RCF_METHOD_R2(double, sub, double, double)
            RCF_METHOD_R2(double, mul, double, double)
            RCF_METHOD_R2(double, div, double, double)
            RCF_METHOD_PLACEHOLDER()
            RCF_METHOD_V3(void, add, double, double, double&)
            RCF_METHOD_V3(void, sub, double, double, double&)
            RCF_METHOD_V3(void, mul, double, double, double&)
            RCF_METHOD_V3(void, div, double, double, double&)
            RCF_METHOD_PLACEHOLDER()
            RCF_METHOD_V1(void, M_in, double)
            RCF_METHOD_V1(void, M_plus, double)
            RCF_METHOD_PLACEHOLDER()
            RCF_METHOD_PLACEHOLDER()
            RCF_METHOD_R0(double, RcM)
            RCF_METHOD_V1(void, RcM, double&)
            RCF_METHOD_V2(void, marshalVector, const std::string &, std::vector<char> &)
            RCF_METHOD_PLACEHOLDER()
        RCF_END( Calculator )

        RCF_BEGIN( X, "X" )
            RCF_METHOD_R0( int,  fr00)
            RCF_METHOD_R1( int,  fr01, int)
            RCF_METHOD_R2( int,  fr02, int, int)
            RCF_METHOD_R3( int,  fr03, int, int, int)
            RCF_METHOD_R4( int,  fr04, int, int, int, int)
            RCF_METHOD_R5( int,  fr05, int, int, int, int, int)
            RCF_METHOD_R6( int,  fr06, int, int, int, int, int, int)
            RCF_METHOD_R7( int,  fr07, int, int, int, int, int, int, int)
            RCF_METHOD_R8( int,  fr08, int, int, int, int, int, int, int, int)
            RCF_METHOD_R9( int,  fr09, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R10(int,  fr10, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R11(int,  fr11, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R12(int,  fr12, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R13(int,  fr13, int, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R14(int,  fr14, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_R15(int,  fr15, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V0( void, fv00)
            RCF_METHOD_V1( void, fv01, int)
            RCF_METHOD_V2( void, fv02, int, int)
            RCF_METHOD_V3( void, fv03, int, int, int)
            RCF_METHOD_V4( void, fv04, int, int, int, int)
            RCF_METHOD_V5( void, fv05, int, int, int, int, int)
            RCF_METHOD_V6( void, fv06, int, int, int, int, int, int)
            RCF_METHOD_V7( void, fv07, int, int, int, int, int, int, int)
            RCF_METHOD_V8( void, fv08, int, int, int, int, int, int, int, int)
            RCF_METHOD_V9( void, fv09, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V10(void, fv10, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V11(void, fv11, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V12(void, fv12, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V13(void, fv13, int, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V14(void, fv14, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
            RCF_METHOD_V15(void, fv15, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        RCF_END( X )

        RCF_BEGIN( Y, "I_Y" )
            RCF_METHOD_R1(int, f01, int)
            RCF_METHOD_R1(int, f02, int)
            RCF_METHOD_R1(int, f03, int)
            RCF_METHOD_R1(int, f04, int)
            RCF_METHOD_R1(int, f05, int)
            RCF_METHOD_R1(int, f06, int)
            RCF_METHOD_R1(int, f07, int)
            RCF_METHOD_R1(int, f08, int)
            RCF_METHOD_R1(int, f09, int)
            RCF_METHOD_R1(int, f10, int)
            RCF_METHOD_R1(int, f11, int)
            RCF_METHOD_R1(int, f12, int)
            RCF_METHOD_R1(int, f13, int)
            RCF_METHOD_R1(int, f14, int)
            RCF_METHOD_R1(int, f15, int)
            RCF_METHOD_R1(int, f16, int)
            RCF_METHOD_R1(int, f17, int)
            RCF_METHOD_R1(int, f18, int)
            RCF_METHOD_R1(int, f19, int)
            RCF_METHOD_R1(int, f20, int)
            RCF_METHOD_R1(int, f21, int)
            RCF_METHOD_R1(int, f22, int)
            RCF_METHOD_R1(int, f23, int)
            RCF_METHOD_R1(int, f24, int)
            RCF_METHOD_R1(int, f25, int)
            RCF_METHOD_R1(int, f26, int)
            RCF_METHOD_R1(int, f27, int)
            RCF_METHOD_R1(int, f28, int)
            RCF_METHOD_R1(int, f29, int)
            RCF_METHOD_R1(int, f30, int)
            RCF_METHOD_R1(int, f31, int)
            RCF_METHOD_R1(int, f32, int)
            RCF_METHOD_R1(int, f33, int)
            RCF_METHOD_R1(int, f34, int)
            RCF_METHOD_R1(int, f35, int)
        RCF_END( Y )

        RCF_BEGIN(
            Z,
            "ReallyLongInterfaceName_"
            "01234567890123456789012345678901234567890123456789"
            "01234567890123456789012345678901234567890123456789"
            "01234567890123456789012345678901234567890123456789"
            "01234567890123456789012345678901234567890123456789"
            "01234567890123456789012345678901234567890123456789"
            "01234567890123456789012345678901234567890123456789")
                RCF_METHOD_R1(std::string, echo, const std::string &)
        RCF_END(Z)

    } //namespace A


    namespace A {
    namespace B {

    class CalculatorDivideByZeroException : public std::exception {
    public:
        CalculatorDivideByZeroException(std::string msg) : msg_(msg) {}
        ~CalculatorDivideByZeroException() throw() {}
        const char *what() const throw() { return msg_.c_str(); }
    private:
        std::string msg_;
    };

    class Calculator {
    public:
        Calculator() : M_(0) {}

        double add(double x, double y)        { return x+y; }
        double sub(double x, double y)        { return x-y; }
        double mul(double x, double y)        { return x*y; }
        double div(double x, double y)        { if (y==0) throw CalculatorDivideByZeroException("Divide by zero"); return x/y; }

        void add(double x, double y, double &z)    { z = x+y; }
        void sub(double x, double y, double &z)    { z = x-y; }
        void mul(double x, double y, double &z)    { z = x*y; }
        void div(double x, double y, double &z)    { if (y==0) throw int(5); z = x/y; }

        void M_in(double x)         { M_ = x; }
        void M_plus(double x)       { M_ += x; }
        double RcM()                { return M_; }
        void RcM(double &x)         { x = M_; }

        void marshalVector(const std::string & s, std::vector<char> & vec)
        {
            vec.clear();
            std::copy(s.begin(), s.end(), std::back_inserter(vec));
        }

    private:
        Calculator(const Calculator &);
        Calculator &operator=(const Calculator &);
        double M_;
    };

    class X {
    public:
        int fr00() { return 1; }
        int fr01(int) { return 1; }
        int fr02(int, int) { return 1; }
        int fr03(int, int, int) { return 1; }
        int fr04(int, int, int, int) { return 1; }
        int fr05(int, int, int, int, int) { return 1; }
        int fr06(int, int, int, int, int, int) { return 1; }
        int fr07(int, int, int, int, int, int, int) { return 1; }
        int fr08(int, int, int, int, int, int, int, int) { return 1; }
        int fr09(int, int, int, int, int, int, int, int, int) { return 1; }
        int fr10(int, int, int, int, int, int, int, int, int, int) { return 1; }
        int fr11(int, int, int, int, int, int, int, int, int, int, int) { return 1; }
        int fr12(int, int, int, int, int, int, int, int, int, int, int, int) { return 1; }
        int fr13(int, int, int, int, int, int, int, int, int, int, int, int, int) { return 1; }
        int fr14(int, int, int, int, int, int, int, int, int, int, int, int, int, int) { return 1; }
        int fr15(int, int, int, int, int, int, int, int, int, int, int, int, int, int, int) { return 1; }

        void fv00() {}
        void fv01(int) {}
        void fv02(int, int) {}
        void fv03(int, int, int) {}
        void fv04(int, int, int, int) {}
        void fv05(int, int, int, int, int) {}
        void fv06(int, int, int, int, int, int) {}
        void fv07(int, int, int, int, int, int, int) {}
        void fv08(int, int, int, int, int, int, int, int) {}
        void fv09(int, int, int, int, int, int, int, int, int) {}
        void fv10(int, int, int, int, int, int, int, int, int, int) {}
        void fv11(int, int, int, int, int, int, int, int, int, int, int) {}
        void fv12(int, int, int, int, int, int, int, int, int, int, int, int) {}
        void fv13(int, int, int, int, int, int, int, int, int, int, int, int, int) {}
        void fv14(int, int, int, int, int, int, int, int, int, int, int, int, int, int) {}
        void fv15(int, int, int, int, int, int, int, int, int, int, int, int, int, int, int) {}
    };

    class Y {
    public:
        int f01(int) { return 1; }
        int f02(int) { return 1; }
        int f03(int) { return 1; }
        int f04(int) { return 1; }
        int f05(int) { return 1; }
        int f06(int) { return 1; }
        int f07(int) { return 1; }
        int f08(int) { return 1; }
        int f09(int) { return 1; }
        int f10(int) { return 1; }
        int f11(int) { return 1; }
        int f12(int) { return 1; }
        int f13(int) { return 1; }
        int f14(int) { return 1; }
        int f15(int) { return 1; }
        int f16(int) { return 1; }
        int f17(int) { return 1; }
        int f18(int) { return 1; }
        int f19(int) { return 1; }
        int f20(int) { return 1; }
        int f21(int) { return 1; }
        int f22(int) { return 1; }
        int f23(int) { return 1; }
        int f24(int) { return 1; }
        int f25(int) { return 1; }
        int f26(int) { return 1; }
        int f27(int) { return 1; }
        int f28(int) { return 1; }
        int f29(int) { return 1; }
        int f30(int) { return 1; }
        int f31(int) { return 1; }
        int f32(int) { return 1; }
        int f33(int) { return 1; }
        int f34(int) { return 1; }
        int f35(int) { return 1; }
    };

    class Z
    {
    public:
        std::string echo(const std::string &s)
        {
            return s;
        }
    };


    } // namespace B
    } // namespace A

    inline bool fuzzy_eq(double d1, double d2, float tolerance = .1)
    {
        return (d1-d2)*(d1-d2) < tolerance;
    }

    void testCalculatorClient(const RCF::I_ClientTransport &clientTransport)
    {

        using namespace A;

        RcfClient<Calculator> calc1(clientTransport.clone());

        RCF::EndpointPtr endpointPtr = clientTransport.getEndpointPtr();
        calc1.getClientStub().setEndpoint(endpointPtr);

        {
            const RcfClient<Calculator> &clientConstRef = calc1;
            std::size_t timeoutMs = clientConstRef.getClientStub().getRemoteCallTimeoutMs();

            RcfClient<Calculator> &clientRef = calc1;
            clientRef.getClientStub().setRemoteCallTimeoutMs(timeoutMs);

            std::vector<RcfClient<Calculator> > clients1(10);
            std::vector<RcfClient<Calculator> > clients2(10);
            clients2.assign(clients1.begin(),clients1.end());
        }

        //bool ok = RCF::createRemoteObject<Calculator>(calc1);
        bool ok = tryCreateRemoteObject<Calculator>(calc1);
        RCF_CHECK(ok);


        RcfClient<Calculator> calc2 = calc1;
        RcfClient<Calculator> calc3;
        calc3 = calc1;
        RcfClient<Calculator> calc4(calc3);

        double ret = 0;
        double x = 0;
        double y = 0;

        x = 1.5;
        y = 2.5;
        ret = calc1.add(x,y); RCF_CHECK( fuzzy_eq(ret, x+y) );
        ret = calc1.sub(x,y); RCF_CHECK( fuzzy_eq(ret, x-y) );
        ret = calc1.mul(x,y); RCF_CHECK( fuzzy_eq(ret, x*y) );
        ret = calc1.div(x,y); RCF_CHECK( fuzzy_eq(ret, x/y) );

        x = 1.5;
        y = 2.5;
        ret = calc2.add(x,y); RCF_CHECK( fuzzy_eq(ret, x+y) );
        ret = calc2.sub(x,y); RCF_CHECK( fuzzy_eq(ret, x-y) );
        ret = calc2.mul(x,y); RCF_CHECK( fuzzy_eq(ret, x*y) );
        ret = calc2.div(x,y); RCF_CHECK( fuzzy_eq(ret, x/y) );

        x = 1.5;
        y = 2.5;
        ret = calc3.add(x,y); RCF_CHECK( fuzzy_eq(ret, x+y) );
        ret = calc3.sub(x,y); RCF_CHECK( fuzzy_eq(ret, x-y) );
        ret = calc3.mul(x,y); RCF_CHECK( fuzzy_eq(ret, x*y) );
        ret = calc3.div(x,y); RCF_CHECK( fuzzy_eq(ret, x/y) );

        x = 1.5;
        y = 2.5;
        ret = calc4.add(x,y); RCF_CHECK( fuzzy_eq(ret, x+y) );
        ret = calc4.sub(x,y); RCF_CHECK( fuzzy_eq(ret, x-y) );
        ret = calc4.mul(x,y); RCF_CHECK( fuzzy_eq(ret, x*y) );
        ret = calc4.div(x,y); RCF_CHECK( fuzzy_eq(ret, x/y) );

        x = 3.14;
        y = -2.718;
        calc4.add(x,y, ret); RCF_CHECK( fuzzy_eq(ret, x+y) );
        calc4.sub(x,y, ret); RCF_CHECK( fuzzy_eq(ret, x-y) );
        calc4.mul(x,y, ret); RCF_CHECK( fuzzy_eq(ret, x*y) );
        calc4.div(x,y, ret); RCF_CHECK( fuzzy_eq(ret, x/y) );

        for(int protocol=1; protocol<10; ++protocol)
        {
            RCF::SerializationProtocol sp = RCF::SerializationProtocol(protocol);
            if (RCF::isSerializationProtocolSupported(sp))
            {
                calc4.getClientStub().setSerializationProtocol(sp);
                std::string s = "asdf";
                std::vector<char> vec;
                calc4.marshalVector(s, vec);
                std::string s2(&vec[0], vec.size());
                RCF_CHECK_EQ(s , s2);
            }
        }

        {
            RcfClient<X> x(clientTransport.clone());
            x.fr00();
            x.fr01(1);
            x.fr02(1,1);
            x.fr03(1,1,1);
            x.fr04(1,1,1,1);
            x.fr05(1,1,1,1,1);
            x.fr06(1,1,1,1,1,1);
            x.fr07(1,1,1,1,1,1,1);
            x.fr08(1,1,1,1,1,1,1,1);
            x.fr09(1,1,1,1,1,1,1,1,1);
            x.fr10(1,1,1,1,1,1,1,1,1,1);
            x.fr11(1,1,1,1,1,1,1,1,1,1,1);
            x.fr12(1,1,1,1,1,1,1,1,1,1,1,1);
            x.fr13(1,1,1,1,1,1,1,1,1,1,1,1,1);
            x.fr14(1,1,1,1,1,1,1,1,1,1,1,1,1,1);
            x.fr15(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);

            x.fv00();
            x.fv01(1);
            x.fv02(1,1);
            x.fv03(1,1,1);
            x.fv04(1,1,1,1);
            x.fv05(1,1,1,1,1);
            x.fv06(1,1,1,1,1,1);
            x.fv07(1,1,1,1,1,1,1);
            x.fv08(1,1,1,1,1,1,1,1);
            x.fv09(1,1,1,1,1,1,1,1,1);
            x.fv10(1,1,1,1,1,1,1,1,1,1);
            x.fv11(1,1,1,1,1,1,1,1,1,1,1);
            x.fv12(1,1,1,1,1,1,1,1,1,1,1,1);
            x.fv13(1,1,1,1,1,1,1,1,1,1,1,1,1);
            x.fv14(1,1,1,1,1,1,1,1,1,1,1,1,1,1);
            x.fv15(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
        }
        {
            RcfClient<Y> y(clientTransport.clone());
            y.f01(1);
            y.f02(1);
            y.f03(1);
            y.f04(1);
            y.f05(1);
            y.f06(1);
            y.f07(1);
            y.f08(1);
            y.f09(1);
            y.f10(1);
            y.f11(1);
            y.f12(1);
            y.f13(1);
            y.f14(1);
            y.f15(1);
            y.f16(1);
            y.f17(1);
            y.f18(1);
            y.f19(1);
            y.f20(1);
            y.f21(1);
            y.f22(1);
            y.f23(1);
            y.f24(1);
            y.f25(1);
            y.f26(1);
            y.f27(1);
            y.f28(1);
            y.f29(1);
            y.f30(1);
            y.f31(1);
            y.f32(1);
            y.f33(1);
            y.f34(1);
            y.f35(1);
        }
        {
            RcfClient<Z> z(clientTransport.clone());
            std::string s = "asdf";
            RCF_CHECK_EQ(s , z.echo(s));
        }

        x = 1;
        y = 1;

        ret = 0;
        ret = calc4.add(1,1); RCF_CHECK( fuzzy_eq(ret, 2) );
        ret = calc4.add(1,1); RCF_CHECK( fuzzy_eq(ret, 2) );

        ret = 0;
        ret = calc4.add(RCF::Oneway, 1,1); RCF_CHECK( fuzzy_eq(ret, 0) );
        ret = calc4.add(RCF::Oneway, 1,1); RCF_CHECK( fuzzy_eq(ret, 0) );
        ret = calc4.add(RCF::Oneway, 1,1); RCF_CHECK( fuzzy_eq(ret, 0) );

        ret = 0;
        calc4.add(RCF::Oneway, 1,1, ret); RCF_CHECK( fuzzy_eq(ret, 0) );
        calc4.add(RCF::Oneway, 1,1, ret); RCF_CHECK( fuzzy_eq(ret, 0) );
        calc4.add(RCF::Oneway, 1,1, ret); RCF_CHECK( fuzzy_eq(ret, 0) );

        ret = 0;
        ret = calc4.add(1,1); RCF_CHECK( fuzzy_eq(ret, 2) );
        ret = calc4.add(1,1); RCF_CHECK( fuzzy_eq(ret, 2) );


        // Test propagation of remote std::exception derived exceptions
        x = 1;
        y = 0;
        try
        {
        ret = calc1.div(x,y);
        RCF_CHECK_FAIL();
        }
        catch(const RCF::RemoteException &e)
        {
            RCF_CHECK_OK();
        }

#ifdef RCF_USE_BOOST_SERIALIZATION

        try
        {
            calc1.getClientStub().setSerializationProtocol(RCF::Sp_BsText);
            ret = calc1.div(x,y);
            RCF_CHECK_FAIL();
        }
        catch(const RCF::RemoteException &e)
        {
            RCF_CHECK_OK();
        }

#endif

    /*
    #ifndef __BORLANDC__ // Scopeguard bug with Borland C++... causes theses tests to crash

        // Test propagation of remote non-std::exception derived exeptions
        x = 1;
        y = 0;
        ret = 0;
        try
        {
            calc.div(x,y, ret);
            RCF_CHECK_FAIL();
        }
        catch(const RCF::RemoteException &e)
        {
            RCF_CHECK_OK();
        }

        // check that the server is still alive....
        x = 1;
        y = 1;
        ret = 0;
        ret = calc.div(x,y);
        RCF_CHECK( fuzzy_eq(ret, 1) );

        {
            RcfClient<Calculator> mycalc(ip, port);
            x = 1;
            y = 1;
            ret = 0;
            ret = mycalc.div(x,y);
            RCF_CHECK( fuzzy_eq(ret, 1) );
        }

    #endif //! __BORLANDC__
    */
    }

    void testCalculatorClientReps(boost::shared_ptr<RCF::I_ClientTransport> clientTransportPtr, int nReps)
    {
        try
        {
            for (int i=0; i<nReps; ++i)
            {
                testCalculatorClient(*clientTransportPtr);
            }
        }
        catch(const std::exception & e)
        {
            RCF_CHECK_FAIL();
        }
    }

} // namespace Test_General

int test_main(int argc, char **argv)
{
    printTestHeader(__FILE__);

    RCF::RcfInitDeinit rcfInitDeinit;

    using namespace Test_General;

    util::CommandLine::getSingleton().parse(argc, argv);

    for (std::size_t i=0; i<RCF::getTransportFactories().size(); ++i)
    {
        RCF::TransportFactoryPtr transportFactoryPtr = RCF::getTransportFactories()[i];
        RCF::TransportPair transports = transportFactoryPtr->createTransports();
        RCF::ServerTransportPtr serverTransportPtr( transports.first );
        RCF::ClientTransportAutoPtr clientTransportAutoPtr( *transports.second );

        std::cout << transportFactoryPtr->desc() << std::endl;

        RCF::RcfServer server(serverTransportPtr);

        unsigned int numberOfTokens = 50;
        unsigned int objectTimeoutS = 60;
        RCF::ObjectFactoryServicePtr objectFactoryServicePtr( new RCF::ObjectFactoryService(numberOfTokens, objectTimeoutS) );
        objectFactoryServicePtr->bind( (A::Calculator*) 0, (A::B::Calculator**) 0);
        objectFactoryServicePtr->bind( (A::X*) 0, (A::B::X**) 0);
        server.addService( RCF::ServicePtr(objectFactoryServicePtr) );

        A::B::Calculator calculator;
        server.bind( (A::Calculator*) 0, calculator);

        A::B::X x;
        server.bind( (A::X*) 0, x);

        A::B::Y y;
        server.bind( (A::Y*) 0, y);

        A::B::Z z;
        server.bind( (A::Z*) 0, z);

        server.start();

        int nReps = 2;

        boost::shared_ptr<RCF::I_ClientTransport> clientTransportPtr(
                clientTransportAutoPtr->clone().release());

        testCalculatorClientReps(clientTransportPtr, nReps);

        server.stop();
    }

    return 0;
}
