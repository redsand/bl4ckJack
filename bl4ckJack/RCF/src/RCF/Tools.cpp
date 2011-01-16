
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

#include <RCF/Tools.hpp>

#include <RCF/Exception.hpp>
#include <RCF/InitDeinit.hpp>

#include <RCF/util/Platform/OS/GetCurrentTime.hpp>

#ifndef __BORLANDC__
namespace std {
#endif

    // Logging of type_info.
    std::ostream &operator<<(std::ostream &os, const std::type_info &ti)
    {
        return os << ti.name();
    }

    // Logging of exception.
    std::ostream &operator<<(std::ostream &os, const std::exception &e)
    {
        os << RCF::toString(e);
        return os;
    }

    // Logging of RCF::Exception.
    std::ostream &operator<<(std::ostream &os, const RCF::Exception &e)
    {
        os << RCF::toString(e);
        return os;
    }

#ifndef __BORLANDC__
} // namespace std
#endif

#if defined(_MSC_VER) && _MSC_VER == 1200

namespace std {

    std::ostream &operator<<(std::ostream &os, __int64)
    {
        // TODO
        RCF_ASSERT(0);
        return os;
    }

    std::ostream &operator<<(std::ostream &os, unsigned __int64)
    {
        // TODO
        RCF_ASSERT(0);
        return os;
    }

    std::istream &operator>>(std::istream &os, __int64 &)
    {
        // TODO
        RCF_ASSERT(0);
        return os;
    }

    std::istream &operator>>(std::istream &os, unsigned __int64 &)
    {
        // TODO
        RCF_ASSERT(0);
        return os;
    }

}

#endif


namespace RCF {

    std::string toString(const std::exception &e)
    {
        std::ostringstream os;

        const RCF::Exception *pE = dynamic_cast<const RCF::Exception *>(&e);
        if (pE)
        {
            int err = pE->getErrorId();
            std::string errMsg = pE->getErrorString();
            os << "[RCF: " << err << ": " << errMsg << "]";
        }
        else
        {
            os << "[What: " << e.what() << "]" ;
        }

        return os.str();
    }

    // 32 bit millisecond counter. Turns over every 49 days or so.
    unsigned int getCurrentTimeMs()
    {
        return Platform::OS::getCurrentTimeMs();
    }

    // Generate a timeout value for the given ending time.
    // Returns zero if endTime <= current time <= endTime+10% of timer resolution,
    // otherwise returns a nonzero duration in ms.
    // Timer resolution as above (49 days).
    unsigned int generateTimeoutMs(unsigned int endTimeMs)
    {
        // 90% of the timer interval
        static const unsigned int maxTimeoutMs = (((unsigned int)-1)/10)*9;
        unsigned int currentTimeMs = getCurrentTimeMs();
        unsigned int timeoutMs = endTimeMs - currentTimeMs;
        return (timeoutMs < maxTimeoutMs) ? timeoutMs : 0;
    }

    boost::uint64_t fileSize(const std::string & path)
    {
        // TODO: this may not work for files larger than 32 bits, on some 32 bit
        // STL implementations. msvc for instance.

        std::ifstream fin ( path.c_str() );
        RCF_VERIFY(fin, Exception(_RcfError_FileOpen(path)));
        std::size_t begin = static_cast<std::size_t>(fin.tellg());
        fin.seekg (0, std::ios::end);
        std::size_t end = static_cast<std::size_t>(fin.tellg());
        fin.close();
        return end - begin;
    }

} // namespace RCF
