
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

#include <RCF/util/Log.hpp>

#include <RCF/ByteBuffer.hpp>
#include <RCF/ThreadLibrary.hpp>
#include <RCF/Tools.hpp>

#include <RCF/util/InitDeinit.hpp>
#include <RCF/util/Platform/OS/ThreadId.hpp>

#include <iostream>

#ifndef BOOST_WINDOWS
#include <sys/time.h>
#endif

namespace util {

    LogManager * gpLogManager;

    util::ThreadSpecificPtr<std::ostrstream>::Val tlsUserBufferPtr;
    util::ThreadSpecificPtr<std::ostrstream>::Val tlsLoggerBufferPtr;

    LogManager::LogManager() : mLoggersMutex(util::WriterPriority)
    {
    }

    void LogManager::init()
    {
        gpLogManager = new LogManager();
    }
    
    void LogManager::deinit()
    {
        delete gpLogManager;
        gpLogManager = NULL;
    }
    
    LogManager & LogManager::instance()
    {
        return *gpLogManager;
    }

    void LogManager::deactivateAllLoggers()
    {
        RCF::WriteLock lock(mLoggersMutex);

        mLoggers.clear();
    }

    void LogManager::deactivateAllLoggers(int name)
    {
        RCF::WriteLock lock(mLoggersMutex);

        Loggers::iterator iter = mLoggers.find(name);
        if (iter != mLoggers.end())
        {
            mLoggers.erase(iter);
        }
    }

    void LogManager::activateLogger(LoggerPtr loggerPtr)
    {
        RCF::WriteLock lock(mLoggersMutex);

        int name = loggerPtr->getName();
        std::vector<LoggerPtr> & loggers = mLoggers[name];
        loggers.push_back(loggerPtr);
    }

    void LogManager::deactivateLogger(LoggerPtr loggerPtr)
    {
        RCF::WriteLock lock(mLoggersMutex);

        int name = loggerPtr->getName();
        Loggers::iterator iter = mLoggers.find(name);
        if (iter != mLoggers.end())
        {
            std::vector<LoggerPtr> & loggers = iter->second;
            RCF::eraseRemove(loggers, loggerPtr);
            if (loggers.empty())
            {
                mLoggers.erase(iter);
            }
        }
    }

    bool LogManager::isLoggerActive(LoggerPtr loggerPtr)
    {
        RCF::ReadLock lock(mLoggersMutex);

        int name = loggerPtr->getName();
        Loggers::iterator iter = mLoggers.find(name);
        if (iter != mLoggers.end())
        {
            std::vector<LoggerPtr> & loggers = iter->second;
            return std::find(loggers.begin(), loggers.end(), loggerPtr) != loggers.end();
        }
        return false;
    }

    void LogManager::writeToLoggers(const LogEntry & logEntry)
    {
        RCF::ReadLock lock(mLoggersMutex);

        int name = logEntry.mName;
        int level = logEntry.mLevel;

        Loggers::iterator iter = mLoggers.find(name);
        if (iter != mLoggers.end())
        {
            std::vector<LoggerPtr> & loggers = iter->second;
            for (std::size_t i=0; i<loggers.size(); ++i)
            {
                if (loggers[i]->getLevel() >= level)
                {
                    loggers[i]->write(logEntry);
                }
            }
        }
    }

    bool LogManager::isEnabled(int name, int level)
    {
        RCF::ReadLock lock(mLoggersMutex);

        Loggers::iterator iter = mLoggers.find(name);
        if (iter != mLoggers.end())
        {
            std::vector<LoggerPtr> & loggers = iter->second;
            for (std::size_t i=0; i<loggers.size(); ++i)
            {
                if (loggers[i]->getLevel() >= level)
                {
                    return true;
                }
            }
        }
        return false;
    }

    LogToStdout::LogToStdout(bool flushAfterEachWrite) : mFlush(flushAfterEachWrite)
    {
    }

    LogTarget * LogToStdout::clone() const
    {
        return new LogToStdout(*this);
    }

    Mutex LogToStdout::sIoMutex;

    void LogToStdout::write(const RCF::ByteBuffer & output)
    {
        output.getPtr()[output.getLength() - 2] = '\n';

        {
            Lock lock(sIoMutex);
            std::cout.write(output.getPtr(), static_cast<std::streamsize>(output.getLength() - 1));
            if (mFlush)
            {
                std::cout.flush();
            }
        }

        output.getPtr()[output.getLength() - 2] = '\0';
    }

#ifdef BOOST_WINDOWS

    LogTarget * LogToDebugWindow::clone() const
    {
        return new LogToDebugWindow(*this);
    }

    void LogToDebugWindow::write(const RCF::ByteBuffer & output)
    {
        output.getPtr()[output.getLength() - 2] = '\n';
        OutputDebugStringA(output.getPtr());
        output.getPtr()[output.getLength() - 2] = '\0';
    }

    LogToEventLog::LogToEventLog(const std::string & appName, int eventLogLevel) : 
        mhEventLog(NULL), 
        mAppName(appName),
        mEventLogLevel(eventLogLevel)
    {
        mhEventLog = RegisterEventSourceA(NULL, mAppName.c_str());
        if (!mhEventLog)
        {
            // TODO: error handling.
        }
    }

    LogToEventLog::~LogToEventLog()
    {
        DeregisterEventSource(mhEventLog);
    }

    LogTarget * LogToEventLog::clone() const
    {
        return new LogToEventLog(mAppName, mEventLogLevel);
    }

    void LogToEventLog::write(const RCF::ByteBuffer & output)
    {
        const char * parms[1] = { output.getPtr() };

        BOOL ret = ReportEventA(
            mhEventLog, 
            (WORD) mEventLogLevel, 
            0, 
            0, 
            0, 
            1, 
            0, 
            parms, 
            NULL);

        RCF_UNUSED_VARIABLE(ret);
    }

#endif

    LogToFile::LogToFile(const std::string & filePath, bool flushAfterEachWrite) : 
        mFilePath(filePath), 
        mOpened(false), 
        mFlush(flushAfterEachWrite)
    {
    }

    LogToFile::LogToFile(const LogToFile & rhs) : 
        mFilePath(rhs.mFilePath), 
        mOpened(false), 
        mFlush(rhs.mFlush)
    {
    }

    LogTarget * LogToFile::clone() const
    {
        return new LogToFile(*this);
    }

    void LogToFile::write(const RCF::ByteBuffer & output)
    {
        if (!mOpened)
        {
            mFout.open(mFilePath.c_str(), std::ios::app);
            if (!mFout.is_open())
            {
                throw std::runtime_error("Unable to open log file.");
            }
            mOpened = true;
        }

        output.getPtr()[output.getLength() - 2] = '\n';

        mFout.write(output.getPtr(), static_cast<std::streamsize>(output.getLength() - 1));

        if (mFlush)
        {
            mFout.flush();
        }

        output.getPtr()[output.getLength() - 2] = '\0';
    }

    LogToFunc::LogToFunc(LogFunctor logFunctor) : mLogFunctor(logFunctor)
    {
    }

    LogTarget * LogToFunc::clone() const
    {
        return new LogToFunc(*this);
    }

    void LogToFunc::write(const RCF::ByteBuffer & output)
    {
        mLogFunctor(output);
    }

#ifdef BOOST_WINDOWS

    boost::uint32_t getCurrentMsValue()
    {
        SYSTEMTIME st; 
        GetSystemTime(&st);
        return st.wMilliseconds;
    }

#else

    boost::uint32_t getCurrentMsValue()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);     
        return tv.tv_usec/1000;
    }

#endif

    LogEntry::LogEntry(int name, int level) : 
        mName(name), 
        mLevel(level),
        mFile(NULL),
        mLine(0),
        mFunc(NULL),
        mThreadId( Platform::OS::GetCurrentThreadId() ),
        mTime(0),
        mTimeMs(0)
    {
        // Current time.
        mTime = time(NULL);
        mTimeMs = getCurrentMsValue();

        if (!tlsUserBufferPtr.get())
        {
            tlsUserBufferPtr.reset( new std::ostrstream() );
        }

        mpOstream = tlsUserBufferPtr.get();
        mpOstream->clear();
        mpOstream->rdbuf()->freeze(false);
        mpOstream->rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);
    }

    LogEntry::LogEntry(int name, int level, const char * szFile, int line, const char * szFunc) : 
        mName(name), 
        mLevel(level), 
        mFile(szFile), 
        mLine(line),
        mFunc(szFunc),
        mThreadId( Platform::OS::GetCurrentThreadId() ),
        mTime(0),
        mTimeMs(0)
    {
        // Current time.
        mTime = time(NULL);
        mTimeMs = getCurrentMsValue();

        if (!tlsUserBufferPtr.get())
        {
            tlsUserBufferPtr.reset( new std::ostrstream() );
        }

        mpOstream = tlsUserBufferPtr.get();
        mpOstream->clear();
        mpOstream->rdbuf()->freeze(false);
        mpOstream->rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);
    }

    LogEntry::~LogEntry()
    {
        *mpOstream << std::ends;
        LogManager::instance().writeToLoggers(*this);
    }

    Logger::Logger(int name, int level, const LogTarget& logTarget, const std::string logFormat) :
        mName(name), 
        mLevel(level), 
        mTargetPtr( logTarget.clone() ), 
        mFormat(logFormat)
    {
    }

    Logger::Logger(int name, int level, const LogTarget& logTarget, LogFormatFunctor logFormatFunctor) :
        mName(name), 
        mLevel(level), 
        mTargetPtr( logTarget.clone() ), 
        mFormatFunctor(logFormatFunctor)
    {
    }

    void Logger::setName(int name)
    {
        if (isActive())
        {
            deactivate();
            mName = name;
            activate();
        }
        else
        {
            mName = name;
        }
    }

    void Logger::setLevel(int level)
    {
        if (isActive())
        {
            deactivate();
            mLevel = level;
            activate();
        }
        else
        {
            mLevel = level;
        }
    }

    void Logger::setTarget(const LogTarget & logTarget)
    {
        if (isActive())
        {
            deactivate();
            mTargetPtr.reset( logTarget.clone() );
            activate();
        }
        else
        {
            mTargetPtr.reset( logTarget.clone() );
        }
    }

    void Logger::setFormat(const std::string & logFormat)
    {
        if (isActive())
        {
            deactivate();
            mFormat = logFormat;
            activate();
        }
        else
        {
            mFormat = logFormat;
        }
    }

    int Logger::getName() const
    {
        return mName;
    }

    int Logger::getLevel() const
    {
        return mLevel;
    }

    const LogTarget& Logger::getTarget() const
    {
        return *mTargetPtr;
    }

    std::string Logger::getFormat() const
    {
        return mFormat;
    }

    // A: Log name
    // B: Log level
    // C: date time
    // D: thread id
    // E: __FILE__
    // F: __LINE__
    // G: __FUNCTION__
    // H: time in ms since RCF initialization
    // X: output

    // %E(%F): [Thread id: %D][Log: %A][Log level: %B]

    void Logger::write(const LogEntry & logEntry)
    {
        // Format the log entry info into a string.
        RCF::ByteBuffer output;
        
        if (mFormatFunctor)
        {
            mFormatFunctor(logEntry, output);
        }
        else
        {
            std::size_t len = static_cast<std::size_t>(logEntry.mpOstream->pcount());
            RCF::ByteBuffer logEntryOutput(logEntry.mpOstream->str(), len);
            
            if (!tlsLoggerBufferPtr.get())
            {
                tlsLoggerBufferPtr.reset( new std::ostrstream() );
            }

            std::ostrstream & os = *tlsLoggerBufferPtr;
            os.clear();
            os.rdbuf()->freeze(false);
            os.rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);

            tm * ts = NULL;
            char timeBuffer[80] = {0};

            std::size_t pos = 0;
            while (pos < mFormat.length())
            {
                if (mFormat[pos] == '%' && pos < mFormat.length())
                {
                    switch (mFormat[pos+1])
                    {
                    case '%': 
                        os << '%';                                    
                        break;

                    case 'A': 
                        os << logEntry.mName;                            
                        break;

                    case 'B': 
                        os << logEntry.mLevel;                        
                        break;

                    case 'C': 

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4995) // 'sprintf': name was marked as #pragma deprecated
#pragma warning(disable: 4996) // 'sprintf'/'localtime': This function or variable may be unsafe.
#endif

                        ts = localtime(&logEntry.mTime);

                        sprintf(
                            timeBuffer, 
                            "%d%02d%02d %02d:%02d:%02d:%03d", 
                            1900+ts->tm_year, 
                            1+ts->tm_mon, 
                            ts->tm_mday, 
                            ts->tm_hour, 
                            ts->tm_min, 
                            ts->tm_sec,
                            logEntry.mTimeMs);

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
                        os << timeBuffer;
                        break;

                    case 'D': 
                        os << logEntry.mThreadId;                        
                        break;

                    case 'E': 
                        os << (logEntry.mFile ? logEntry.mFile : "File Not Specified");    
                        break;

                    case 'F': 
                        os << logEntry.mLine;                            
                        break;

                    case 'G': 
                        os << logEntry.mFunc;                            
                        break;

                    case 'H':
                        os << RCF::getCurrentTimeMs();
                        break;

                    case 'X': 
                        os.write(logEntryOutput.getPtr(), static_cast<std::streamsize>(logEntryOutput.getLength()-1));                
                        break;

                    default:
                        RCF_ASSERT(0);
                    }

                    pos += 2;
                    continue;
                }
                else
                {
                    os << mFormat[pos];
                    pos += 1;
                }
            }

            // Terminate the string with two zeros, that way the log targets can insert a newline if they want.
            os << std::ends << std::ends;

            output = RCF::ByteBuffer(os.str(), static_cast<std::size_t>(os.pcount()));
        }

        // Pass the string to the log target.
        if (output)
        {
            mTargetPtr->write(output);
        }
    }

    void Logger::activate()
    {
        LogManager::instance().activateLogger( shared_from_this() );
    }

    void Logger::deactivate()
    {
        LogManager::instance().deactivateLogger( shared_from_this() );
    }

    bool Logger::isActive()
    {
        return LogManager::instance().isLoggerActive( shared_from_this() );
    }

    UTIL_ON_INIT_NAMED( LogManager::init(), LogInitialize )
    UTIL_ON_DEINIT_NAMED( LogManager::deinit(), LogDeinitialize )

} // namespace util


namespace util {

    util::ThreadSpecificPtr<std::ostrstream>::Val tlsVarArgBuffer1Ptr;
    util::ThreadSpecificPtr<std::ostrstream>::Val tlsVarArgBuffer2Ptr;

    VariableArgMacroFunctor::VariableArgMacroFunctor() :
        mFile(NULL),
        mLine(0),
        mFunc(NULL)
    {
        if (!tlsVarArgBuffer1Ptr.get())
        {
            tlsVarArgBuffer1Ptr.reset( new std::ostrstream() );
            *tlsVarArgBuffer1Ptr << std::ends;
        }

        if (!tlsVarArgBuffer2Ptr.get())
        {
            tlsVarArgBuffer2Ptr.reset( new std::ostrstream() );
            *tlsVarArgBuffer2Ptr << std::ends;
        }

        mHeader = tlsVarArgBuffer1Ptr.get();
        mHeader->clear();
        mHeader->rdbuf()->freeze(false);
        mHeader->rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);

        mArgs = tlsVarArgBuffer2Ptr.get();
        mArgs->clear();
        mArgs->rdbuf()->freeze(false);
        mArgs->rdbuf()->pubseekoff(0, std::ios::beg, std::ios::out);
    }

    VariableArgMacroFunctor::~VariableArgMacroFunctor()
    {}

    VariableArgMacroFunctor & VariableArgMacroFunctor::init(
        const std::string &label,
        const std::string &msg,
        const char *file,
        int line,
        const char *func)
    {
        mFile = file;
        mLine = line;
        mFunc = func;

        unsigned int timestamp = Platform::OS::getCurrentTimeMs();
        Platform::OS::ThreadId threadid = Platform::OS::GetCurrentThreadId();
       
        *mHeader
            << file << "(" << line << "): "
            << func << ": "
            << ": Thread-id=" << threadid
            << " : Timestamp(ms)=" << timestamp << ": "
            << label << msg << ": "
            << std::ends;

        return *this;
    }

} // namespace util

