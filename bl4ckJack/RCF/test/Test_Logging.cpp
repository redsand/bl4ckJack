
#include <RCF/test/TestMinimal.hpp>

#include <RCF/ByteBuffer.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/util/Log.hpp>

#include <RCF/RCF.hpp>

bool gInstrumented = false;
bool gExpectAllocations = true;
std::size_t gnAllocations = 0;

// User-defined operator new.
void *operator new(size_t bytes)
{
    if (gInstrumented)
    {
        RCF_CHECK(gExpectAllocations);
        if (!gExpectAllocations)
        {
            std::cout << "Unexpected memory allocation." << std::endl;
        }
        ++gnAllocations;
    }
    return malloc(bytes);
}

// User-defined operator delete.
void operator delete(void *pv) throw()
{
    free(pv);
}


// Log levels
static const int LogLevel_Error         = 0;
static const int LogLevel_Warning       = 1;
static const int LogLevel_Diagnostic    = 2;

// Log names
static const int Log_1                  = 1;
static const int Log_2                  = 2;
static const int Log_3                  = 3;
static const int Log_4                  = 4;

using util::LogManager;
using util::Logger;
using util::LoggerPtr;
using util::LogEntry;
using util::LogToStdout;
using util::LogToFile;
using util::LogToDebugWindow;
using util::LogToFunc;
using util::LogFormatFunctor;


#define LOG(name, level) UTIL_LOG(name, level)
#define LOGENTRY(name, level, entry) LogEntry entry(name, level, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION);

void sampleCode()
{
    int name = Log_2;
    int level = LogLevel_Error;
    
    LOG(name, level) << "bla";
    LOG(name, level) << "bla";

    int someVar = 17;

    LOG(name, level) << "name/value pairs: " << NAMEVALUE(someVar) << NAMEVALUE(someVar/2);
    LOG(name, level)(someVar)(someVar+someVar);



    LOG(name, level)(someVar)(someVar+someVar) << "asdf" << "qwer";

    {
        gInstrumented = true;
        gExpectAllocations = true;

        std::size_t nAllocations = gnAllocations;
        std::auto_ptr<int> apn(new int(17));
        apn.reset();
        RCF_CHECK_NEQ(gnAllocations , nAllocations);
    }

    {
        // Check that there are no heap allocations.
        gInstrumented = true;
        gExpectAllocations = false;
        LOG(name, level) << "bla";
        LOG(name, level) << "name/value pairs: " << NAMEVALUE(someVar) << NAMEVALUE(someVar/2);
        LOG(name, level)(someVar)(someVar+someVar);

        gExpectAllocations = true;
    }

    {
        LOGENTRY(name, level, entry);
        entry << std::endl << "1" << std::endl << "2";
    }
}

void loggerFunc(const RCF::ByteBuffer & output)
{
    RCF_UNUSED_VARIABLE(output);
}

void myFormatFunc(const LogEntry & logEntry, RCF::ByteBuffer & output)
{
    RCF_UNUSED_VARIABLE(logEntry);
    RCF_UNUSED_VARIABLE(output);
}

int sampleMain()
{
    int name = Log_2;
    int level = LogLevel_Error;

    // Log name
    // Log level
    // date time
    // thread id
    // __FILE__
    // __LINE__
    // __FUNCTION__
    // output
    std::string logFormat = "%C\t %X";

    LoggerPtr(new Logger(name, level, LogToStdout(), logFormat))->activate();

    LoggerPtr logger1Ptr(new Logger(name, level, LogToStdout(), logFormat));    
    logger1Ptr->activate();

    std::string odsLogFormat = "%E(%F): [Thread id: %D][Function: %G] %X";
    LoggerPtr logger2Ptr(new Logger(name, level, LogToDebugWindow(), odsLogFormat) );
    logger2Ptr->activate();

    LoggerPtr logger3Ptr(new Logger(name, level, LogToFile("c:\\log.txt"), logFormat));
    logger3Ptr->activate();

    LoggerPtr logger4Ptr(new Logger(name, level, LogToFunc(loggerFunc), logFormat));
    logger4Ptr->activate();

    LoggerPtr logger5Ptr(new Logger(name, level, LogToFunc(loggerFunc), LogFormatFunctor(myFormatFunc)));
    logger5Ptr->activate();

    sampleCode();

    logger2Ptr->deactivate();

    sampleCode();

    logger2Ptr->activate();

    sampleCode();

    logger2Ptr->setFormat("%E(%F): %X");

    sampleCode();

    LogManager::instance().deactivateAllLoggers(name);

    sampleCode();

    logger2Ptr->activate();

    sampleCode();

    LogManager::instance().deactivateAllLoggers();

    sampleCode();

    logger2Ptr->activate();

    sampleCode();

    return 0;
}

#include <RCF/ServerInterfaces.hpp>
#include <RCF/Tools.hpp>

RCF_BEGIN(I_X, "I_X")
RCF_METHOD_R1(std::string, echo, const std::string &)
RCF_END(I_X)

class X
{
public:
    std::string echo(const std::string & s)
    {
        return s;
    }
};

int test_main(int, char **)
{
    RCF::RcfInitDeinit init;

    sampleMain();

    LogManager::instance().deactivateAllLoggers();
/*
    std::string odsLogFormat = "%E(%F): [Thread id: %D][Function: %G] %X";
    LoggerPtr loggerPtr(new Logger(RCF::LogNameRcf, RCF::LogLevel_1, LogToDebugWindow(), odsLogFormat) );
    loggerPtr->activate();

    try
    {
        RCF::RcfClient<RCF::I_Null> client( RCF::TcpEndpoint(50001) );
        client.getClientStub().ping();
    }
    catch(const RCF::Exception & e)
    {
        RCF_LOG_1()(e);
    }
*/

    RCF::setDefaultConnectTimeoutMs(1000*3600);
    RCF::setDefaultRemoteCallTimeoutMs(1000*3600);

    std::string fileLogFormat = "%C\t[Thread %D]\t%X";
    LoggerPtr rcfLoggerPtr(new Logger(RCF::LogNameRcf, RCF::LogLevel_3, LogToFile("C:\\Test_Logging.log"), fileLogFormat) );
    rcfLoggerPtr->activate();

    RCF_LOG_1();
    RCF_LOG_1() << "--------------------------------------------------------------------";
    RCF_LOG_1() << "Log opened.";
    RCF_LOG_1() << "--------------------------------------------------------------------";
    RCF_LOG_1();

    X x;
    RCF::RcfServer server( RCF::TcpEndpoint(0) );
    server.bind<I_X>(x);
    server.start();

    int port = server.getIpServerTransport().getPort();

    RcfClient<I_X> client(( RCF::TcpEndpoint(port) ));
    for (std::size_t i=0; i<5; ++i)
    {
        std::string s0 = "some stuff to pass around";
        std::string s1 = client.echo(s0);
    }

    return 0;
}
