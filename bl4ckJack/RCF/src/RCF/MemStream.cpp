
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

#include <RCF/MemStream.hpp>

#include <RCF/Tools.hpp>

namespace RCF {

    // mem_streambuf implementation

    mem_streambuf::mem_streambuf(   
        char * buffer, 
        std::size_t bufferLen) : 
            mBuffer(buffer),
            mBufferLen(bufferLen)
    {   
        setg(mBuffer, mBuffer, mBuffer + mBufferLen);
    }   

    void mem_streambuf::reset(char * buffer, std::size_t bufferLen)
    {
        mBuffer = buffer;
        mBufferLen = bufferLen;
        setg(mBuffer, mBuffer, mBuffer + mBufferLen);
    }

    std::streambuf::int_type mem_streambuf::underflow()   
    {   
        if (gptr() < egptr())
        {
            return traits_type::to_int_type(*gptr());
        }

        return traits_type::eof();   
    }

    mem_streambuf::pos_type mem_streambuf::seekoff(
        mem_streambuf::off_type offset, 
        std::ios_base::seekdir dir,
        std::ios_base::openmode mode)
    {
        RCF_UNUSED_VARIABLE(mode);

        char * pBegin = mBuffer;
        char * pEnd = mBuffer + mBufferLen;
        
        char * pBase = NULL;
        switch(dir)
        {
            case std::ios::cur: pBase = gptr(); break;
            case std::ios::beg: pBase = pBegin; break;
            case std::ios::end: pBase = pEnd; break;
            default: assert(0); break; 
        }

        char * pNewPos = pBase + offset;
        if (pBegin <= pNewPos && pNewPos <= pEnd)
        {
            setg(pBegin, pNewPos, pEnd);
            return pNewPos - pBegin;
        }
        else
        {
            return pos_type(-1);
        }
    }

    // mem_istream implementation

    mem_istream::mem_istream(const char * buffer, std::size_t bufferLen) :
        std::basic_istream<char>(&mBuf),
        mBuf(const_cast<char *>(buffer), bufferLen)
    {   
    }

    void mem_istream::reset(const char * buffer, std::size_t bufferLen)
    {
        clear();
        mBuf.reset(const_cast<char *>(buffer), bufferLen);
    }

} // namespace RCF