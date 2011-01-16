
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

#ifndef INCLUDE_RCF_IOCP_HPP
#define INCLUDE_RCF_IOCP_HPP

#include <windows.h>
#include <WinSock2.h>

// ULONG_PTR definition for vc6
#include <RCF/Tools.hpp>

namespace RCF {

    // Iocp

    class Iocp
    {
    public:
        Iocp(int nMaxConcurrency = -1);
        ~Iocp();

        void Create(
            int nMaxConcurrency = 0);

        void AssociateDevice(
            HANDLE hDevice,
            ULONG_PTR CompKey);

        void AssociateSocket(
            SOCKET hSocket,
            ULONG_PTR CompKey);

        void PostStatus(
            ULONG_PTR CompKey,
            DWORD dwNumBytes = 0,
            OVERLAPPED* po = NULL) ;

        BOOL GetStatus(
            ULONG_PTR* pCompKey,
            PDWORD pdwNumBytes,
            OVERLAPPED** ppo,
            DWORD dwMilliseconds = INFINITE);


        void cycle(int timeoutMs);

    private:
        HANDLE m_hIOCP;
    };

    class IocpOverlapped : public OVERLAPPED
    {
    public:

        IocpOverlapped()
        {
            clearOverlapped();
        }

        virtual ~IocpOverlapped()
        {}

        void clearOverlapped()
        {
            memset(static_cast<OVERLAPPED *>(this), 0, sizeof(OVERLAPPED));
        }

        virtual void onCompletion(
            BOOL ret, 
            DWORD dwErr,
            ULONG_PTR completionKey, 
            DWORD dwNmBytes) = 0;
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_IOCP_HPP
