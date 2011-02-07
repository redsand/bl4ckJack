
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

#ifndef GPU_DEVICE_H
#define GPU_DEVICE_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <class T>
class Singleton
{
public:
	static T& getInstance() {
		static T _instance;
		return _instance;
	}
private:
	Singleton();          // ctor hidden
	~Singleton();          // dtor hidden
	Singleton(Singleton const&);    // copy ctor hidden
	Singleton& operator=(Singleton const&);  // assign op hidden
};

#ifdef  __cplusplus
extern "C" {
#endif

	#define MAX_DEVICES 512
	/* CUDA Device Information */
	struct devInfo
	{
		/* Number of CUDA capable device detected */
		int deviceCount;
		/* CUDA specific */
		struct Devices
		{
			int CUDA_device_ID;
			struct cudaDeviceProp deviceProp;
		}Devs[MAX_DEVICES];
	};

	class GPU_Dev {

	public:
		GPU_Dev(void);
		~GPU_Dev(void);
		int getDevs(struct devInfo *devInf);
		int getDevCount();
		int getDevInfoStr(int CUDA_device_ID, char *str, int strlen);
		int setDevice(int CUDA_device);
		int mallocHost(void **ptr, size_t size);
		int freeHost(void *ptr);
		int mallocGPU(void **ptr, size_t size);
		int freeGPU(void *ptr); 
		int last_error_string(char *str, unsigned int str_len);

	private:
		bool isCUDAError(void);
		bool isCUDAError(int CUDA_device_ID);
		int findCUDA_DevsIndex(int CUDA_device_ID);
		struct devInfo devInf;
	};

	typedef Singleton<GPU_Dev> GPU;

#ifdef  __cplusplus
}
#endif

#endif /* GPU_DEVICE_H */