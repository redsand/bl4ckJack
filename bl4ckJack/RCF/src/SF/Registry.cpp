
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

#include <SF/Registry.hpp>

#include <RCF/InitDeinit.hpp>

#include <SF/SerializeAny.hpp>
#include <SF/string.hpp>

namespace SF {

    // serialization for boost::any
    void serialize(SF::Archive &ar, boost::any &a)
    {
        if (ar.isWrite())
        {
            std::string which = 
                SF::Registry::getSingleton().getTypeName(a.type());

            ar & which;
            SF::Registry::getSingleton().getAnySerializer(which)
                .serialize(ar, a);
        }
        else
        {
            std::string which;
            ar & which;
            SF::Registry::getSingleton().getAnySerializer(which)
                .serialize(ar, a);
        }
    }

    void initRegistrySingleton();

    static Registry *pRegistry;

    Registry::Registry() :
        mReadWriteMutex(Platform::Threads::writer_priority)
    {}

    Registry &Registry::getSingleton()
    {
        if (!pRegistry)
        {
            initRegistrySingleton();
        }
        return *pRegistry;
    }

    Registry *Registry::getSingletonPtr()
    {
        return &getSingleton();
    }

    bool Registry::isTypeRegistered(const std::string &typeName)
    {
        ReadLock lock(mReadWriteMutex); 
        RCF_UNUSED_VARIABLE(lock);
        return mTypenameToRtti.find(typeName) != mTypenameToRtti.end();
    }

    bool Registry::isTypeRegistered(const std::type_info &ti)
    {
        ReadLock lock(mReadWriteMutex); 
        RCF_UNUSED_VARIABLE(lock);
        Rtti typeRtti = ti.name();
        return mRttiToTypename.find(typeRtti) != mRttiToTypename.end();
    }

    I_SerializerAny &Registry::getAnySerializer(const std::string &which)
    {
        ReadLock lock(mReadWriteMutex); 
        RCF_UNUSED_VARIABLE(lock);
        if (mTypenameToRtti.find(which) != mTypenameToRtti.end())
        {
            Rtti rtti = mTypenameToRtti[which];

            RCF_VERIFY(
                mRttiToSerializerAny.find(rtti) != mRttiToSerializerAny.end(),
                RCF::Exception(RCF::_RcfError_AnySerializerNotFound(which)));

            return *mRttiToSerializerAny[rtti];
        }
        
        RCF::Exception e(RCF::_RcfError_AnySerializerNotFound(which));
        RCF_THROW(e);
    }

    std::string Registry::getTypeName(const std::type_info &ti)
    {
        ReadLock lock(mReadWriteMutex); 
        RCF_UNUSED_VARIABLE(lock);
        Rtti typeRtti = ti.name();
        if (mRttiToTypename.find(typeRtti) == mRttiToTypename.end())
        {
            return "";
        }
        else
        {
            return mRttiToTypename[typeRtti];
        }
    }

    void Registry::clear()
    {
        mTypenameToRtti.clear();
        mRttiToTypename.clear();
        mRttiToSerializerPolymorphic.clear();
        mRttiToSerializerAny.clear();
    }

    void initRegistrySingleton()
    {
        if (!pRegistry)
        {
            pRegistry = new Registry();
        }
    }

    void deinitRegistrySingleton()
    {
        delete pRegistry;
        pRegistry = NULL;
    }

    RCF_ON_INIT_DEINIT(
        initRegistrySingleton(),
        deinitRegistrySingleton())

}
