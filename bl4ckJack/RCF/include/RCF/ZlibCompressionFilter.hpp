
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

#ifndef INCLUDE_RCF_ZLIBCOMPRESSIONFILTER_HPP
#define INCLUDE_RCF_ZLIBCOMPRESSIONFILTER_HPP

#include <memory>
#include <vector>

#include <boost/noncopyable.hpp>

#include <RCF/AsyncFilter.hpp>
#include <RCF/Export.hpp>

namespace RCF {

    static const int ZlibDefaultBufferSize = 4096;

    class ZlibCompressionReadFilter;
    class ZlibCompressionWriteFilter;

    class RCF_EXPORT ZlibCompressionFilterBase : 
        public Filter, 
        boost::noncopyable
    {
    public:
        ZlibCompressionFilterBase(bool stateful, bool serverSide);
       
    private:
        void reset();

        void read(const ByteBuffer &byteBuffer, std::size_t bytesRequested);
        void write(const std::vector<ByteBuffer> &byteBuffers);
        void onReadCompleted(const ByteBuffer &byteBuffer);
        void onWriteCompleted(std::size_t bytesTransferred);

        enum IoState
        {
            Ready,
            Reading,
            Writing
        };

        // input state
        IoState mPreState;

        friend class ZlibCompressionReadFilter;
        friend class ZlibCompressionWriteFilter;

        boost::shared_ptr<ZlibCompressionReadFilter> mReadFilter;
        boost::shared_ptr<ZlibCompressionWriteFilter> mWriteFilter;
    };

    class ServerSide {};

    /// Filter implementing a stateless compression protocol, through the Zlib library.
    class RCF_EXPORT ZlibStatelessCompressionFilter : 
        public ZlibCompressionFilterBase
    {
    private:
        friend class ZlibStatelessCompressionFilterFactory;

        ZlibStatelessCompressionFilter(
            ServerSide *) :
                ZlibCompressionFilterBase(false, true)
        {}

    public:
        /// Constructor.
        /// \param bufferSize Internal buffer size, limiting how much data can be compressed/decompressed in a single operation.
        ZlibStatelessCompressionFilter() :
                ZlibCompressionFilterBase(false, false)
        {}        

        static const FilterDescription & sGetFilterDescription();
        const FilterDescription & getFilterDescription() const;

        // TODO: should be private
        static const FilterDescription *spFilterDescription;
    };

    /// Filter implementing a stateful compression protocol, through the Zlib library.
    class RCF_EXPORT ZlibStatefulCompressionFilter : 
        public ZlibCompressionFilterBase
    {
    private:
        friend class ZlibStatefulCompressionFilterFactory;

        ZlibStatefulCompressionFilter(
            ServerSide *) :
                ZlibCompressionFilterBase(true, true)
        {}

    public:
        /// Constructor.
        /// \param bufferSize Internal buffer size, limiting how much data can be compressed/decompressed in a single operation.
        ZlibStatefulCompressionFilter() :
                ZlibCompressionFilterBase(true, false)
        {}

        static const FilterDescription & sGetFilterDescription();
        const FilterDescription & getFilterDescription() const;

        // TODO: should be private
        static const FilterDescription *spFilterDescription;
    };
   
    /// Filter factory for ZlibStatelessCompressionFilter.
    class RCF_EXPORT ZlibStatelessCompressionFilterFactory : 
        public FilterFactory
    {
    public:
        ZlibStatelessCompressionFilterFactory();

        FilterPtr createFilter();
        const FilterDescription & getFilterDescription();
    };

    /// Filter factory for ZlibStatefulCompressionFilter.
    class RCF_EXPORT ZlibStatefulCompressionFilterFactory : 
        public FilterFactory
    {
    public:
        ZlibStatefulCompressionFilterFactory();

        FilterPtr createFilter();
        const FilterDescription & getFilterDescription();
    };

    typedef ZlibStatefulCompressionFilter               ZlibCompressionFilter;
    typedef boost::shared_ptr<ZlibCompressionFilter>    ZlibCompressionFilterPtr;

    typedef ZlibStatefulCompressionFilterFactory                ZlibCompressionFilterFactory;
    typedef boost::shared_ptr<ZlibCompressionFilterFactory>     ZlibCompressionFilterFactoryPtr;

} // namespace RCF

#endif // ! INCLUDE_RCF_ZLIBCOMPRESSIONFILTER_HPP
