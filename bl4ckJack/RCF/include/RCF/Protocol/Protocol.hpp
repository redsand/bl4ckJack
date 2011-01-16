
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

#ifndef INCLUDE_RCF_PROTOCOL_PROTOCOL_HPP
#define INCLUDE_RCF_PROTOCOL_PROTOCOL_HPP

#include <iosfwd>

#include <boost/function.hpp>

#include <RCF/Config.hpp>
#include <RCF/Exception.hpp>
#include <RCF/Tools.hpp>

namespace RCF {

    enum SerializationProtocol 
    {

        Sp_SfBinary       = 1,
        Sp_SfText         = 2,
        Sp_BsBinary       = 3,
        Sp_BsText         = 4,
        Sp_BsXml          = 5,

        // Legacy identifiers (RCF 1.1).
        SfBinary       = 1,
        SfText         = 2,
        BsBinary       = 3,
        BsText         = 4,
        BsXml          = 5

    };

    extern const SerializationProtocol DefaultSerializationProtocol;

    enum MarshalingProtocol
    {
        Mp_Rcf = 1,
        Mp_Protobuf = 2
    };

    extern const MarshalingProtocol DefaultMarshalingProtocol;

    template<typename N>
    class Protocol
    {
    public:

        static std::string getName()
        {
            return "";
        }

        class In
        {
        public:
            void bind(
                std::istream &is, 
                std::size_t archiveSize, 
                int runtimeVersion, 
                int archiveVersion)
            {
                RCF_UNUSED_VARIABLE(is);
                RCF_UNUSED_VARIABLE(archiveSize);
                RCF_UNUSED_VARIABLE(runtimeVersion);
                RCF_UNUSED_VARIABLE(archiveVersion);
            }

            void unbind()
            {}
           
            template<typename T>
            In &operator>>(T &t)
            {
                RCF_UNUSED_VARIABLE(t);
                RCF_THROW(RCF::SerializationException(_RcfError_UnknownSerializationProtocol(N::value)))(N::value);
                return *this;
            }
        };

        class Out
        {
        public:
            void bind(
                std::ostream &os, 
                int runtimeVersion, 
                int archiveVersion)
            {
                RCF_UNUSED_VARIABLE(os);
                RCF_UNUSED_VARIABLE(runtimeVersion);
                RCF_UNUSED_VARIABLE(archiveVersion);
            }

            void unbind()
            {}

            template<typename T>
            Out &operator<<(const T &t)
            {
                RCF_UNUSED_VARIABLE(t);
                RCF_THROW(RCF::SerializationException(_RcfError_UnknownSerializationProtocol(N::value)))(N::value);
                return *this;
            }
        };
    };


    template<typename IS, typename OS>
    class ProtocolImpl1
    {
    public:
        class In
        {
        public:
            void bind(
                std::istream &is, 
                std::size_t archiveSize, 
                int runtimeVersion, 
                int archiveVersion)
            {
                mAr.clearState();
                mAr.setIs(is, archiveSize, runtimeVersion, archiveVersion);
                invokeCustomizationCallback();
            }

            void unbind()
            {}
           
            template<typename T> In &operator>>(T &t)
            {
                mAr >> t;
                return *this;
            }

            typedef boost::function1<void, IS &> CustomizationCallback;

            void setCustomizationCallback(CustomizationCallback customizationCallback)
            {
                mCustomizationCallback = customizationCallback;
                invokeCustomizationCallback();
            }           
       
        private:
            IS mAr;

            CustomizationCallback mCustomizationCallback;

            void invokeCustomizationCallback()
            {
                if (mCustomizationCallback)
                {
                    mCustomizationCallback(mAr);
                }
            }

        };

        class Out
        {
        public:
            void bind(
                std::ostream &os, 
                int runtimeVersion, 
                int archiveVersion)                           
            {
                mAr.clearState();
                mAr.setOs(os, runtimeVersion, archiveVersion);
                mAr.suppressVersionStamp();

                invokeCustomizationCallback();
            }

            void unbind()
            {}
           
            template<typename T> Out &operator<<(const T &t)
            {
                mAr << t;
                return *this;
            }

            typedef boost::function1<void, OS &> CustomizationCallback;

            void setCustomizationCallback(CustomizationCallback customizationCallback)
            {
                mCustomizationCallback = customizationCallback;
                invokeCustomizationCallback();
            }

        private:
            OS mAr;

            CustomizationCallback mCustomizationCallback;

            void invokeCustomizationCallback()
            {
                if (mCustomizationCallback)
                {
                    mCustomizationCallback(mAr);
                }
            }

        };
    };
   
    template<typename IS, typename OS>
    class ProtocolImpl2
    {
    public:
        class In
        {
        public:
            void bind(
                std::istream &is, 
                std::size_t archiveSize, 
                int runtimeVersion, 
                int archiveVersion)
            {
                RCF_UNUSED_VARIABLE(archiveSize);
                RCF_UNUSED_VARIABLE(runtimeVersion);
                RCF_UNUSED_VARIABLE(archiveVersion);
                mAr.reset( new IS(is) );
                invokeCustomizationCallback();
            }

            void unbind()
            {
                mAr.reset();
            }

            template<typename T> In &operator>>(T &t)
            {
                *mAr >> t;
                return *this;
            }

            typedef boost::function1<void, IS &> CustomizationCallback;

            void setCustomizationCallback(CustomizationCallback customizationCallback)
            {
                mCustomizationCallback = customizationCallback;
                invokeCustomizationCallback();
            }           

        private:
            std::auto_ptr<IS> mAr;

            CustomizationCallback mCustomizationCallback;

            void invokeCustomizationCallback()
            {
                if (mCustomizationCallback)
                {
                    mCustomizationCallback(*mAr);
                }
            }

        };

        class Out
        {
        public:
            void bind(
                std::ostream &os, 
                int runtimeVersion, 
                int archiveVersion)                           
            {
                RCF_UNUSED_VARIABLE(runtimeVersion);
                RCF_UNUSED_VARIABLE(archiveVersion);
                mAr.reset( new OS(os) );
                invokeCustomizationCallback();
            }

            void unbind()
            {
                mAr.reset();
            }

            template<typename T> Out &operator<<(const T &t)
            {
                *mAr << t;
                return *this;
            }

            typedef boost::function1<void, OS &> CustomizationCallback;

            void setCustomizationCallback(CustomizationCallback customizationCallback)
            {
                mCustomizationCallback = customizationCallback;
                invokeCustomizationCallback();
            }

        private:
            std::auto_ptr<OS> mAr;

            CustomizationCallback mCustomizationCallback;

            void invokeCustomizationCallback()
            {
                if (mCustomizationCallback)
                {
                    mCustomizationCallback(*mAr);
                }
            }

        };
    };

} // namespace RCF

#endif //! INCLUDE_RCF_PROTOCOL_PROTOCOL_HPP
