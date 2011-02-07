
/* 
    Copyright (C) 2009  Benjamin Vernoux, titanmkd@gmail.com

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA */

/* GPU device manager class for Win32 or Linux
This class is a singleton and manage:
Multi GPU ...
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuda_gpu.h"

#if (defined(WIN32) || defined(__WIN32__) || defined(__WIN32))
	#include <windows.h> /* For Win32 QueryPerformanceCounter() */
#endif
#include "bl4ckJack_timer.h"

#ifndef snprintf
#define snprintf sprintf_s
#endif

#include <cuda_runtime.h>

/* Internal/Private */
inline bool GPU_Dev::isCUDAError(void)
{
	if(this->devInf.deviceCount < 1)
	{
		return true;
	}
	return false;
}

/* Internal/Private */
inline bool GPU_Dev::isCUDAError(int CUDA_device_ID)
{
	if(CUDA_device_ID < 0 || CUDA_device_ID > MAX_DEVICES || isCUDAError())
	{
		return true;
	}

	return false;
}

/*
Internal/Private
Return CUDA device index
Return <0 if error
*/
inline int GPU_Dev::findCUDA_DevsIndex(int CUDA_device_ID)
{
	int dev = -1;

	for(int i=0; i<MAX_DEVICES; i++)
	{
		if(CUDA_device_ID == this->devInf.Devs[i].CUDA_device_ID)
		{
			dev = i;
			break;
		}
	}
	return dev;
}

int GPU_Dev::getDevCount() {
	int ret = 0;
	if(this->devInf.deviceCount == -1) {
		cudaError_t err = cudaGetDeviceCount(&ret);
		if (err != cudaSuccess || ret == 0)
		{
			/* There is no device supporting CUDA */
			ret = -1;
			return ret;
		}

		return ret;
	}

	return this->devInf.deviceCount;
}

GPU_Dev::GPU_Dev(void)
{
	cudaError_t err;
	int CUDAdev;
	bool isCUDA = false;

	this->devInf.deviceCount = -1;
	
	this->devInf.deviceCount = getDevCount();

	if (this->devInf.deviceCount <= 0)
	{
		/* There is no device supporting CUDA */
		this->devInf.deviceCount = -1;
		return;
	}

	/* Set all devices to invalid */
	for(int i=0; i<MAX_DEVICES; i++)
		this->devInf.Devs[i].CUDA_device_ID = -1;

	/* Retrieve information for all devices */
	CUDAdev = 0;
	for (int dev = 0; dev < this->devInf.deviceCount; ++dev) 
	{
		struct cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, dev);
		if (devProp.major >= 1)
		{
			isCUDA = true;
			this->devInf.Devs[CUDAdev].CUDA_device_ID = dev;
			memcpy(&this->devInf.Devs[CUDAdev].deviceProp, &devProp, sizeof(cudaDeviceProp));
			++CUDAdev;
		}
	}

	this->devInf.deviceCount = CUDAdev;

	if(!isCUDA)
		this->devInf.deviceCount = -1;
}

GPU_Dev::~GPU_Dev(void)
{

}

/*
Get Information for device.
Return <0 if error
-1 : There is no device supporting CUDA
*/
int GPU_Dev::getDevInfoStr(int CUDA_device_ID, char *str, int len) 
{
	int dev;

	str[0] = 0;

	if(isCUDAError(CUDA_device_ID))
		return -1;

	dev = findCUDA_DevsIndex(CUDA_device_ID);
	if(dev<0)
	{
		return -1;
	}

	/* There are X devices supporting CUDA */
	snprintf(str, len,"%s (v%d.%d)\n\tProcessors: %d, ClockRate: %01.02f MHz, TotalMem:%01.02f MB\n",
		this->devInf.Devs[dev].deviceProp.name, 
		this->devInf.Devs[dev].deviceProp.major,
		this->devInf.Devs[dev].deviceProp.minor,
		this->devInf.Devs[dev].deviceProp.multiProcessorCount,
		((double)(this->devInf.Devs[dev].deviceProp.clockRate)/1000.0),
		((double)(this->devInf.Devs[dev].deviceProp.totalGlobalMem)/(double)(1024*1024)));
	str[len] = 0;

	return 0;
}

/*
Get Information for devices.
Return <0 if error
-1 : There is no device supporting CUDA
*/
int GPU_Dev::getDevs(struct devInfo *devInf) 
{
	if(isCUDAError())
		return -1;

	memcpy(devInf, &this->devInf, sizeof(devInfo));
	return 0;
}

/*
Set the device used for CUDA computation.
Return <0 if error
-1 : There is no device supporting CUDA or invalid device
*/
int GPU_Dev::setDevice(int CUDA_device_ID) 
{
	cudaError_t err;
	
	if(isCUDAError(CUDA_device_ID))
		return -1;

	err = cudaSetDevice(CUDA_device_ID);
	if( err != cudaSuccess)
		return -1;
	else
		return 0;
}

/*
Allocate memory.
Return <0 if error
-1 : There is no device supporting CUDA or invalid device
*/
int GPU_Dev::mallocHost(void **ptr, size_t size) 
{
	cudaError_t err;

	err = cudaMallocHost(ptr, size);
	if( err != cudaSuccess)
		return -1;
	else
		return 0;
}

/*
Allocate GPU memory used only by GPU.
Return <0 if error
-1 : There is no device supporting CUDA or invalid device
*/
int GPU_Dev::mallocGPU(void **ptr, size_t size) 
{
	cudaError_t err;

	err = cudaMalloc(ptr, size);
	if( err != cudaSuccess)
		return -1;
	else
		return 0;
}


/*
Free GPU memory used only by GPU.
Return <0 if error
-1 : There is no device supporting CUDA or invalid device
*/
int GPU_Dev::freeGPU(void *ptr) 
{
	cudaError_t err;

	err = cudaFree(ptr);
	if( err != cudaSuccess)
		return -1;
	else
		return 0;
}

/*
Free GPU memory used only by GPU.
Return <0 if error
-1 : There is no device supporting CUDA or invalid device
*/
int GPU_Dev::freeHost(void *ptr) 
{
	cudaError_t err;

	err = cudaFreeHost(ptr);
	if( err != cudaSuccess)
		return -1;
	else
		return 0;
}

/*
Return 0 if no error
Return 1 if error
*/
int GPU_Dev::last_error_string(char *str, uint str_len)
{
	const char* Err_str;
	cudaError_t last_err;
	last_err = cudaGetLastError();
	if(last_err == cudaSuccess)
	{
		/* No error */
		str[0] = 0;
		return 0;
	}else
	{
		Err_str = cudaGetErrorString(last_err);
		snprintf(str, str_len, "%s", Err_str);
		return 1;
	}
}
