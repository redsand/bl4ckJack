
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

#ifndef INCLUDE_SF_VECTOR_HPP
#define INCLUDE_SF_VECTOR_HPP

#include <vector>

#include <SF/Serializer.hpp>
#include <SF/SerializeStl.hpp>

namespace SF {

    // std::vector

    template<typename T, typename A>
    inline void serializeVector(
        SF::Archive &           ar,
        std::vector<T,A> &      vec,
        boost::mpl::false_ *)
    {
        serializeStlContainer<PushBackSemantics, ReserveSemantics>(ar, vec);
    }

    template<typename T, typename A>
    inline void serializeVector(
        SF::Archive &           ar,
        std::vector<T,A> &      vec,
        boost::mpl::true_ *)
    {
        serializeVectorFast(ar, vec);
    }

    template<typename T, typename A>
    inline void serialize_vc6(
        SF::Archive &           ar,
        std::vector<T,A> &      vec, 
        const unsigned int)
    {
        typedef typename RCF::IsFundamental<T>::type type;
        serializeVector(ar, vec, (type *) 0);
    }



    class I_VecWrapper
    {
    public:
        virtual void            resize(std::size_t newSize) = 0;
        virtual boost::uint32_t size() = 0;
        virtual char *          addressOfElement(std::size_t idx) = 0;
        virtual boost::uint32_t sizeofElement() = 0;
    };

    template<typename Vec>
    class VecWrapper : public I_VecWrapper
    {
    public:
        VecWrapper(Vec & vec) : mVec(vec)
        {
        }

        void resize(std::size_t newSize)
        {
            mVec.resize(newSize);
        }

        boost::uint32_t size()
        {
            return static_cast<boost::uint32_t>(mVec.size());
        }

        char * addressOfElement(std::size_t idx)
        {
            return reinterpret_cast<char *>( &mVec[idx] );
        }

        boost::uint32_t sizeofElement()
        {
            typedef typename Vec::value_type ValueType;
            return sizeof(ValueType);
        }

    private:
        Vec & mVec;
    };

    RCF_EXPORT void serializeVectorFastImpl(
        SF::Archive &           ar,
        I_VecWrapper &          vec);

    template<typename T, typename A>
    inline void serializeVectorFast(
        SF::Archive &           ar,
        std::vector<T,A> &      vec)
    {
        VecWrapper< std::vector<T,A> > vecWrapper(vec);
        serializeVectorFastImpl(ar, vecWrapper);
    }
        
} // namespace SF

#endif // ! INCLUDE_SF_VECTOR_HPP
 INCLUDE_SF_VECTOR_HPP
