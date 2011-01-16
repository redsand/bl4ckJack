
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

#ifndef INCLUDE_RCF_TEST_TESTMINIMAL_HPP
#define INCLUDE_RCF_TEST_TESTMINIMAL_HPP

// Include valarray early so it doesn't get trampled by min/max macro definitions.
#if defined(_MSC_VER) && _MSC_VER == 1310 && defined(RCF_USE_BOOST_SERIALIZATION)
#include <valarray>
#endif

// For msvc-7.1, prevent including <locale> from generating warnings.
#if defined(_MSC_VER) && _MSC_VER == 1310
#pragma warning( disable : 4995 ) //  'func': name was marked as #pragma deprecated
#include <locale>
#pragma warning( default : 4995 )
#endif

// Weird errors with Borland C++Builder 6, in WSPiApi.h. This seems to help.
#if defined(__BORLANDC__) && __BORLANDC__ < 0x561
#include <Windows.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <WSPiApi.h>
#endif

#include <RCF/Exception.hpp>
#include <RCF/InitDeinit.hpp>
#include <RCF/Tools.hpp>

#include <RCF/test/PrintTestHeader.hpp>
#include <RCF/test/ProgramTimeLimit.hpp>
#include <RCF/test/SpCollector.hpp>
#include <RCF/test/Test.hpp>
#include <RCF/test/TransportFactories.hpp>

#include <RCF/util/Platform/OS/BsdSockets.hpp>
#include <RCF/util/Log.hpp>

#include <iostream>
#include <sstream>

#if defined(_MSC_VER) && _MSC_VER >= 1310
#include <RCF/test/MiniDump.hpp>
#endif

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

int test_main(int argc, char **argv);

int main(int argc, char **argv)
{
    Platform::OS::BsdSockets::disableBrokenPipeSignals();

    RCF::RcfInitDeinit init;

    RCF::Timer testTimer;

    RCF::initializeTransportFactories();

    std::cout << "Commandline: ";
    for (int i=0; i<argc; ++i)
    {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    bool shoudNotCatch = false;

    {
        util::CommandLineOption<std::string>    clTestCase( "testcase",     "",     "Run a specific test case.");
        util::CommandLineOption<bool>           clListTests("list",         false,  "List all test cases.");
        util::CommandLineOption<bool>           clAssert(   "assert",       false,  "Enable assert popups, and assert on test failures.");
        util::CommandLineOption<int>            clLogLevel( "loglevel",     1,      "Set RCF log level.");
        util::CommandLineOption<bool>           clLogTime(  "logtime",      false,  "Set RCF time stamp logging.");
        util::CommandLineOption<bool>           clLogTid(   "logtid",       false,  "Set RCF thread id logging.");
        util::CommandLineOption<std::string>    clLogFormat("logformat",    "",     "Set RCF log format.");
        util::CommandLineOption<bool>           clNoCatch(  "nocatch",      false,  "Don't catch exceptions at top level.");
        util::CommandLineOption<unsigned int>   clTimeLimit("timelimit",    5*60,   "Set program time limit in seconds. 0 to disable.");

#if defined(_MSC_VER) && _MSC_VER >= 1310
        util::CommandLineOption<bool>           clMinidump("minidump",      true,   "Enable minidump creation.");
#endif

        bool exitOnHelp = false;
        util::CommandLine::getSingleton().parse(argc, argv, exitOnHelp);

        // -testcase
        std::string testCase = clTestCase.get();
        if (!testCase.empty())
        {
            gTestEnv().setTestCaseToRun(testCase);
        }

        // -list
        bool list = clListTests.get();
        if (list)
        {
            gTestEnv().setEnumerationOnly();
        }

        // -assert
        bool assertOnFail = clAssert.get();
        gTestEnv().setAssertOnFail(assertOnFail);

#ifdef BOOST_WINDOWS
        if (!assertOnFail)
        {
            // Try to prevent those pesky crash dialogs from popping up.

            DWORD dwFlags = SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS;
            DWORD dwOldFlags = SetErrorMode(dwFlags);
            SetErrorMode(dwOldFlags | dwFlags);

#ifdef _MSC_VER
            // Disable CRT asserts.
            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
            _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
            _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG); 
#endif

        }
#endif


        // -loglevel
        int logName = RCF::LogNameRcf;
        int logLevel = clLogLevel.get();
        bool includeTid = clLogTid.get();
        bool includeTimestamp = clLogTime.get();

        std::string logFormat = clLogFormat.get();
        if (logFormat.empty())
        {
            if (!includeTid && !includeTimestamp)
            {
                logFormat = "%E(%F): %X";
            }
            if (includeTid && !includeTimestamp)
            {
                logFormat = "%E(%F): (Tid:%D): %X";
            }
            else if (!includeTid && includeTimestamp)
            {
                logFormat = "%E(%F): (Time:%H): %X";
            }
            else if (includeTid && includeTimestamp)
            {
                logFormat = "%E(%F): (Tid:%D)(Time::%H): %X";
            }
        }

#ifdef BOOST_WINDOWS
        util::LoggerPtr loggerPtr(new util::Logger(logName, logLevel, util::LogToDebugWindow(), logFormat) );
        loggerPtr->activate();
#else
        util::LoggerPtr loggerPtr(new util::Logger(logName, logLevel, util::LogToStdout(), logFormat) );
        loggerPtr->activate();
#endif

        // -minidump
#if defined(_MSC_VER) && _MSC_VER >= 1310
        bool enableMinidumps = clMinidump.get();
        if (enableMinidumps)
        {
            setMiniDumpExceptionFilter();
        }
#endif

        // -timelimit
        unsigned int timeLimitS = clTimeLimit.get();
        gpProgramTimeLimit = new ProgramTimeLimit(timeLimitS);

        shoudNotCatch = clNoCatch.get();
    }

    int ret = 0;
    
    bool shouldCatch = !shoudNotCatch;
    if (shouldCatch)
    {
        try
        {
            ret = test_main(argc, argv);
        }
        catch(const RCF::RemoteException & e)
        {
            std::cout << "Caught top-level exception (RCF::RemoteException): " << e.getErrorString() << std::endl;
            RCF_CHECK(1==0);
        }
        catch(const RCF::Exception & e)
        {
            std::cout << "Caught top-level exception (RCF::Exception): " << e.getErrorString() << std::endl;
            RCF_CHECK(1==0);
        }
        catch(const std::exception & e)
        {
            std::cout << "Caught top-level exception (std::exception): " << e.what() << std::endl;
            RCF_CHECK(1==0);
        }
        catch (...)
        {
            std::cout << "Caught top-level exception (...)" << std::endl;
            RCF_CHECK(1==0);
        }
    }
    else
    {
        ret = test_main(argc, argv);
    }

    std::string exitMsg;
    std::size_t failCount = gTestEnv().getFailCount();
    if (failCount)
    {
        std::ostringstream os;
        os << "*** Test Failures: " << failCount << " ***" << std::endl;
        exitMsg = os.str();
    }
    else
    {
        exitMsg = "*** All Tests Passed ***\n";
    }

    gTestEnv().printTestMessage(exitMsg);

    // Print out how long the test took.
    boost::uint32_t durationMs = testTimer.getDurationMs();
    std::cout << "Time elapsed: " << durationMs/1000 << " (s)" << std::endl;

    return ret + static_cast<int>(failCount);
}

// Minidump creation code, for Visual C++ 2003 and later.
#if defined(_MSC_VER) && _MSC_VER >= 1310
#include <RCF/test/MiniDump.cpp>
#endif

#include <RCF/../../src/RCF/test/Test.cpp>

#endif // ! INCLUDE_RCF_TEST_TESTMINIMAL_HPP
