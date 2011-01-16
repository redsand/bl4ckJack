
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

#include <RCF/test/Test.hpp>

#if (defined(_MSC_VER) && _MSC_VER == 1200) || defined(__BORLANDC__)

// VC6 and Borland 5.6 - Boost string algo won't compile, so use stricmp instead.
#include <string.h>
bool compareNoCase(const std::string & s1, const std::string & s2)
{
    return 0 == stricmp(s1.c_str(), s2.c_str());
}

#else

// Boost string algo is more portable than stricmp, strcasecmp, etc.
#include <boost/algorithm/string.hpp>
bool compareNoCase(const std::string & s1, const std::string & s2)
{
    return 
            boost::to_upper_copy(s1) 
        ==  boost::to_upper_copy(s2);
}

#endif


TestEnv gTestEnv_;

TestEnv & gTestEnv()
{
    return gTestEnv_;
}

TestHierarchy::TestHierarchy() : 
    mCaseSensitive(false), 
    mHasTestCaseRun(false),
    mEnumerateOnly(false)
{
}

void TestHierarchy::pushTestCase(const std::string & name)
{
    mCurrentTestCase.push_back(name);
}

void TestHierarchy::popTestCase()
{
    mCurrentTestCase.pop_back();        
}

std::string TestHierarchy::currentTestCase()
{
    std::string s;
    for (std::size_t i=0; i<mCurrentTestCase.size(); ++i)
    {
        s += mCurrentTestCase[i];
        if (i != mCurrentTestCase.size() - 1)
        {
            s += '/';
        }
    }
    return s;
}

void TestHierarchy::onlyRunTestCase(const std::string & testCase, bool caseSensitive)
{
    split(testCase, '/', mTestCaseToRun);
    mCaseSensitive = caseSensitive;
}

void TestHierarchy::enumerateTestCasesOnly()
{
    mEnumerateOnly = true;
}

bool TestHierarchy::shouldCurrentTestCaseRun()
{
    if (mEnumerateOnly)
    {
        std::cout << "Test case: " << currentTestCase() << std::endl;

        bool match = doesCurrentTestCaseMatch();
        return match && mCurrentTestCase.size() <= mTestCaseToRun.size();
    }
    else if (mTestCaseToRun.empty())
    {
        return true;
    }
    else
    {
        bool matches = doesCurrentTestCaseMatch();
        if (matches && mTestCaseToRun.size() == mCurrentTestCase.size())
        {
            mHasTestCaseRun = true;
        }
        return matches;
    }
}

bool TestHierarchy::didTestCaseRun()
{
    return mHasTestCaseRun;
}

void TestHierarchy::split(
    const std::string &s, 
    char delim, 
    std::vector<std::string> &elems) 
{
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

bool TestHierarchy::doesCurrentTestCaseMatch()
{
    for (std::size_t i=0; i<mCurrentTestCase.size(); ++i)
    {
        if (i < mTestCaseToRun.size())
        {
            if (mCaseSensitive)
            {
                return mCurrentTestCase[i] == mTestCaseToRun[i];
            }
            else
            {
                return compareNoCase(mCurrentTestCase[i], mTestCaseToRun[i]);               
            }
        }
    }

    return true;
}


TestCase::TestCase(const std::string & name) : mName(name), mHasRun(false)
{
    TestHierarchy & th = gTestEnv().mTestHierarchy;
    th.pushTestCase(mName);
    mRunnable = th.shouldCurrentTestCaseRun();

    if (mRunnable)
    {
        std::cout << "Entering test case " << th.currentTestCase() << "." << std::endl;
    }
}

#if defined(_MSC_VER) && _MSC_VER == 1310
// C4267: 'argument' : conversion from 'size_t' to 'unsigned int', possible loss of data
#pragma warning(push)
#pragma warning( disable: 4267 ) 
#endif

TestCase::TestCase(std::size_t n) : mRunnable(false), mHasRun(false)
{
    TestHierarchy & th = gTestEnv().mTestHierarchy;

    std::ostringstream os;
    os << n;
    mName = os.str();

    th.pushTestCase(mName);

    mRunnable = th.shouldCurrentTestCaseRun();

    if (mRunnable)
    {
        std::cout << "Entering test case " << th.currentTestCase() << "." << std::endl;
    }
}

#if defined(_MSC_VER) && _MSC_VER == 1310
#pragma warning(pop)
#endif

TestCase::~TestCase()
{
    TestHierarchy & th = gTestEnv().mTestHierarchy;

    if (mRunnable)
    {
        std::cout << "Leaving test case " << th.currentTestCase() << "." << std::endl;
    }

    th.popTestCase();
}

bool TestCase::shouldRun()
{
    return mRunnable && !mHasRun;
}

void TestCase::setHasRun()
{
    mHasRun = true;
}

#ifdef BOOST_WINDOWS

void TestEnv::printTestMessage(const std::string & msg)
{
    std::cout << msg << std::flush;
    OutputDebugStringA(msg.c_str());
}

#else

void TestEnv::printTestMessage(const std::string & msg)
{
    std::cout << msg << std::flush;
}

#endif

void TestEnv::reportTestFailure(
    const char * file, 
    int line, 
    const char * condition,
    const char * info)
{
    std::ostringstream os;
    
    os 
        << file << "(" << line << "): Test case: " 
        << mTestHierarchy.currentTestCase() 
        << std::endl;

    os
        << file << "(" << line << "): Test failed: " 
        << condition;
    
    if (info)
    {
        os << " : " << info;
    }
    
    os << std::endl;

    printTestMessage( os.str() );
    
    ++mFailCount;

    if (mAssertOnFail)
    {
        assert(0);
    }
}

TestEnv::TestEnv() : mFailCount(0), mAssertOnFail(false)
{
}

void TestEnv::setTestCaseToRun(const std::string & testCase, bool caseSensitive)
{
    mTestHierarchy.onlyRunTestCase(testCase, caseSensitive);
}

void TestEnv::setEnumerationOnly()
{
    mTestHierarchy.enumerateTestCasesOnly();
}

void TestEnv::setAssertOnFail(bool assertOnFail)
{
    mAssertOnFail = assertOnFail;
}

std::size_t TestEnv::getFailCount()
{
    return mFailCount;
}

bool TestEnv::didTestCaseRun()
{
    return mTestHierarchy.didTestCaseRun();
}
