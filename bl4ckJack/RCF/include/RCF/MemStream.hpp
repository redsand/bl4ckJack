
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

#ifndef INCLUDE_RCF_MEMSTREAM_HPP
#define INCLUDE_RCF_MEMSTREAM_HPP

#include <istream>
#include <streambuf>

// std::size_t for vc6
#include <boost/cstdint.hpp> 

#include <boost/noncopyable.hpp>

namespace RCF {

    // mem_streambuf

    class mem_streambuf :
        public std::streambuf, 
        boost::noncopyable   
    {   
      public:   
        mem_streambuf(char * buffer = NULL, std::size_t bufferLen = 0);
        void reset(char * buffer, std::size_t bufferLen);
             
      private:   
        std::streambuf::int_type underflow();   

        pos_type seekoff(
            off_type off, 
            std::ios_base::seekdir dir,
            std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out);
           
        char * mBuffer;
        std::size_t mBufferLen; 
    };   

    // mem_istream - essentially a replacement for std::istrstream.

    class mem_istream : 
        public std::basic_istream<char>
    {   
      public:   
        mem_istream(const char * buffer = NULL, std::size_t bufferLen = 0);   
        void reset(const char * buffer, std::size_t bufferLen);
           
      private:   
        mem_streambuf mBuf;
    };   


} // namespace RCF

#endif
