
#include "stdafx.h"
#include <bl4ckJack_module.h>
#include "md5_global.h"


#define MAX_CHARSET		255		// bytes (non-unicode support atm)
__device__ __constant__ char __align__(16) gpu_charset[MAX_CHARSET];
__device__ __constant__ unsigned int gpu_charset_len;

// init our charset
// hashes into dev mem btree
// 
int bl4ckJackGPUKernelInit(char *charset, unsigned int charset_len, void **hashes, unsigned long long hashCount) {

	if(cudaMemcpyToSymbol(gpu_charset, charset, charset_len) != cudaSuccess) {
			return 0;
	}

	if(cudaMemcpyToSymbol(&gpu_charset_len, &charset_len, sizeof(charset_len)) != cudaSuccess) {
			return 0;
	}
	
	// hashes & hashCount
	// create our btree in device memory for matching?
	// or create a large array in device memory and do comparisons instead
	//			sort large array for easier/faster matching?
	
}

// __inline__ ToBase

// need to allocate max charset len +1 * index so our kernel can calculate our string

// kernel will take base value (can calc or assign value to array)
// each thread = compute hash and check btree for result
// each thread will += until its id > stopping key
__global__ void bl4ckJackGPUKernelExecute(long double start, long double stop) {
    
    unsigned long long index = threadIdx.x + blockIdx * blockDim.x;
    int i=0;
    
    // prime into shared regional memory
    // because i was told this is faster than device memory
    __shared__ char charset[256];
    __shared__ int charsetLen;
	if(threadIdx.x == 0)
	{
		// load charset/len from gpu mem
		while(gpu_charset[i]) {
			charset[i] = gpu_charset[i];
			i++;
		}
		charset[i] = '\0';
		charsetLen = i;
	}
	
	//Wait for all cache filling to finish
	__syncthreads();
    
    long double start_token = start + index; // base token
    long double stop_token = stop + index;
    long double iter = 0;
    if(stop_token > stop)
		stop_token -= index;
	
	for(iter = start_token; iter <= stop_token; iter += index) {
		/*
		ToBase(iter, (char *)bruteStr, MAX_BRUTE_CHARS);
		pfbl4ckJackGenerate((unsigned char *)results, &retLen, (unsigned char *)bruteStr, strlen((const char *)bruteStr));
		// gen our hash and check for existance in hash list
		if(this->btree.find(results, retLen)) {
			// if match, somehow notify on backend of success
		}
		*/
	}
	
    return;
}

}


// end our init and free all our memory, including btree, etc.