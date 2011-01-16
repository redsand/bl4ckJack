
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

#include <RCF/FilterService.hpp>

#include <boost/bind.hpp>

#include <RCF/CurrentSession.hpp>
#include <RCF/RcfServer.hpp>
#include <RCF/RcfSession.hpp>
#include <RCF/ServerInterfaces.hpp>

#ifdef RCF_USE_PROTOBUF
#include <RCF/protobuf/RcfMessages.pb.h>
#endif

namespace RCF {

    FilterService::FilterService() :
        mFilterFactoryMapMutex(WriterPriority)
    {
    }

#ifdef RCF_USE_PROTOBUF

    class FilterServicePb
    {
    public:

        FilterServicePb(FilterService & fs) : mFs(fs)
        {
        }

        void RequestTransportFilters(const PbRequestTransportFilters & request)
        {
            std::vector<boost::int32_t> filterIds;
            int filterCount = request.filterids_size();
            for (int i=0; i<filterCount; ++i)
            {
                filterIds.push_back(request.filterids(i));
            }

            int error = mFs.RequestTransportFilters(filterIds);

            if (error != RCF::RcfError_Ok)
            {
                Error err(error);
                RemoteException e(err);
                RCF_THROW(e);
            }
        }

    private:
        FilterService & mFs;
    };

    void onServiceAddedProto(FilterService & fs, RcfServer & server)
    {
        boost::shared_ptr<FilterServicePb> fsPbPtr(
            new FilterServicePb(fs));

        server.bind((I_RequestTransportFiltersPb *) NULL, fsPbPtr);
    }

    void onServiceRemovedProto(FilterService & fs, RcfServer & server)
    {
        server.unbind( (I_RequestTransportFiltersPb *) NULL);
    }

#else

    void onServiceAddedProto(FilterService &, RcfServer &)
    {
    }

    void onServiceRemovedProto(FilterService &, RcfServer &)
    {
    }

#endif // RCF_USE_PROTOBUF


    void FilterService::onServiceAdded(RcfServer & server)
    {
        server.bind( (I_RequestTransportFilters *) NULL, *this);

        onServiceAddedProto(*this, server);
    }

    void FilterService::onServiceRemoved(RcfServer & server)
    {
        server.unbind( (I_RequestTransportFilters *) NULL);

        onServiceRemovedProto(*this, server);
    }

    void FilterService::addFilterFactory(FilterFactoryPtr filterFactoryPtr)
    {
        FilterDescription filterDescription = filterFactoryPtr->getFilterDescription();
        WriteLock writeLock(mFilterFactoryMapMutex);
        mFilterFactoryMap[ filterDescription.getId() ] = filterFactoryPtr;
    }

    void FilterService::addFilterFactory(
        FilterFactoryPtr filterFactoryPtr,
        const std::vector<int> &filterIds)
    {
        FilterDescription filterDescription = filterFactoryPtr->getFilterDescription();
        WriteLock writeLock(mFilterFactoryMapMutex);
        for (std::size_t i=0; i<filterIds.size(); ++i)
        {
            mFilterFactoryMap[ filterIds[i] ] = filterFactoryPtr;
        }
    }

#if defined(_MSC_VER) && _MSC_VER <= 1200
#define for if (0) {} else for
#endif

    // remotely accessible
    boost::int32_t FilterService::RequestTransportFilters(const std::vector<boost::int32_t> &filterIds)
    {
        RCF_LOG_3()(filterIds) << "FilterService::RequestTransportFilters() - entry";

        ReadLock readLock(mFilterFactoryMapMutex);
        for (unsigned int i=0; i<filterIds.size(); ++i)
        {
            int filterId = filterIds[i];
            if (mFilterFactoryMap.find(filterId) == mFilterFactoryMap.end())
            {
                RCF_LOG_3()(filterId) << "FilterService::RequestTransportFilters() - unknown filter.";
                return RcfError_UnknownFilter;
            }
        }
        boost::shared_ptr< std::vector<FilterPtr> > filters(
            new std::vector<FilterPtr>());

        for (unsigned int i=0; i<filterIds.size(); ++i)
        {
            int filterId = filterIds[i];
            FilterFactoryPtr filterFactoryPtr = mFilterFactoryMap[filterId];
            FilterPtr filterPtr( filterFactoryPtr->createFilter() );
            filters->push_back(filterPtr);
        }
        RcfSession & session = getCurrentRcfSession();
        if (session.transportFiltersLocked())
        {
            RCF_LOG_3() << "FilterService::RequestTransportFilters() - filter sequence already locked.";
            return RcfError_FiltersLocked;
        }
        else
        {
            session.addOnWriteCompletedCallback( boost::bind(
                &FilterService::setTransportFilters, 
                this, 
                _1, 
                filters) );
        }        

        RCF_LOG_3() << "FilterService::RequestTransportFilters() - exit";
        return RcfError_Ok;
    }

#if defined(_MSC_VER) && _MSC_VER <= 1200
#undef for
#endif

    // remotely accessible
    boost::int32_t FilterService::QueryForTransportFilters(const std::vector<boost::int32_t> &filterIds)
    {
        ReadLock readLock(mFilterFactoryMapMutex);
        for (unsigned int i=0; i<filterIds.size(); ++i)
        {
            int filterId = filterIds[i];
            if (mFilterFactoryMap.find(filterId) == mFilterFactoryMap.end())
            {
                return RcfError_UnknownFilter;
            }
        }

        return getCurrentRcfSession().transportFiltersLocked() ?
            RcfError_FiltersLocked :
            RcfError_Ok;
    }

    FilterFactoryPtr FilterService::getFilterFactoryPtr(int filterId)
    {
        ReadLock readLock(mFilterFactoryMapMutex);
        return mFilterFactoryMap.find(filterId) == mFilterFactoryMap.end() ?
            FilterFactoryPtr() :
            mFilterFactoryMap[filterId];
    }

    void FilterService::setTransportFilters(
        RcfSession &session,
        boost::shared_ptr<std::vector<FilterPtr> > filters)
    {
        session.getSessionState().setTransportFilters(*filters);
    }

} // namespace RCF
