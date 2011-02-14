
#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "md5_gpu.h"

#define MAX_CHARSET		255		// bytes (non-unicode support atm)
#define MAX_PASSLENGTH	255
#define MAX_PASSCOUNT	1024
__device__ __constant__ char __align__(16) gpu_charset[MAX_CHARSET];
__device__ __constant__ unsigned int gpu_charset_len;

__device__ __constant__ char __align__(16) **hashList=NULL;
__device__ unsigned long hashListCount=0;

__device__ unsigned long matchCount;
__device__ char __align__(16) matchHashList[MAX_PASSCOUNT][MAX_PASSLENGTH + 1];
__device__ char __align__(16) matchPassList[MAX_PASSCOUNT][MAX_PASSLENGTH + 1];

// init our charset
// init our hashes for matching as well
//! Initialize GPU for Bruteforcing
extern "C" __declspec(dllexport) void bl4ckJackInitGPU(char *charset, int charsetLen) {
	
	if(cudaMemcpyToSymbol("gpu_charset", charset, charsetLen+1, 0, cudaMemcpyHostToDevice) != cudaSuccess) {
		return;
	}

	if(cudaMemcpyToSymbol("gpu_charset_len", &charsetLen, sizeof(charsetLen)) != cudaSuccess) {
		return;
	}

	if(cudaMemset(matchHashList, 0, MAX_PASSCOUNT * MAX_PASSLENGTH) != cudaSuccess) {
		return;
	}

	if(cudaMemset(matchPassList, 0, MAX_PASSCOUNT * MAX_PASSLENGTH) != cudaSuccess) {
		return;
	}

	return;
}

//! Free initialized memory
extern "C" __declspec(dllexport) void bl4ckJackFreeGPU(void) {
	/*
	unsigned long i;
	for(i=0; i < hashListCount; i++)
		cudaFree(hashList[i]);
	cudaFree(hashList);
	*/
}

// need to allocate max charset len +1 * index so our kernel can calculate our string
__device__ size_t my_strlen(const char *c) {
	if(!c) return 0;
	register unsigned int i=0;
	while(c[i++]);
	return i;
}

__device__ int my_memcmp ( const void *s1V, const void *s2V, int n )
{
   int res;
   unsigned char a0;
   unsigned char b0;
   unsigned char* s1 = (unsigned char*)s1V;
   unsigned char* s2 = (unsigned char*)s2V;

   while (n != 0) {
      a0 = s1[0];
      b0 = s2[0];
      s1 += 1;
      s2 += 1;
      res = ((int)a0) - ((int)b0);
      if (res != 0)
         return res;
      n -= 1;
   }
   return 0;
}

// kernel will take base value (can calc or assign value to array)
// each thread = compute hash and check btree for result
// each thread will += until its id > stopping key
extern "C" __global__ __declspec(dllexport) void bl4ckJackGenerateGPUInternal(double *start, double *stop, int maxIterations,  char **gpuHashList, int *gpuHashListCount, int *maxSuccess) {
    
    //int index = (blockDim.x * blockIdx.x) + threadIdx.x; //threadIdx.x + blockIdx * blockDim.x;
	int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x + threadIdx.x; // assuming blockDim.y = 1 and threadIdx.y = 0, always

    char input[256]; // max len of our passwd
    unsigned char retBuf[64];
    int i=0;
    // prime into shared regional memory
    // because i was told this is faster than device memory
    __shared__ char localcharset[256];
    __shared__ int localcharsetLen;

	if(threadIdx.x == 0)
	{
		// load charset/len from gpu mem
		i=0;
		while(gpu_charset[i]) {
			localcharset[i] = gpu_charset[i];
			i++;
		}
		localcharset[gpu_charset_len] = '\0';

	}

	//Wait for all cache filling to finish
	__syncthreads();
    
	
    double start_token = *start + index; // base token
    double stop_token = *stop + index;
    double iter = 0;
	int count=0;

    if(stop_token > *stop)
		stop_token -= index;
	
	
	for(iter = start_token; iter <= stop_token; iter += index)
	{
	
		//ToBase(iter, input, sizeof(input)-1, localcharset, localcharsetLen);
		int r = 0, biter = 0;
		float number = iter - 1;
		if(number < 0) {
			input[0] = '\0';
		}
		do {
			if(biter > (sizeof(input)-1)) break;
			r =  floor(fmod(number, localcharsetLen)); //number % (this->charsetLen)
			if(r < localcharsetLen)
				input[biter++] = localcharset[r];
			else
				input[biter++] = '=';
			if(biter >= localcharsetLen) break;
			number = (number / localcharsetLen) - 1;
		} while(number >= 0);

		input[biter] = '\0';
		
		i=0;
		
		size_t inputLen = my_strlen(input);

		if (i<(inputLen/2)) {
			char c;
			c=input[i];
			input[i]=input[inputLen-i-1];
			input[inputLen-i-1]=c;
		}

		// this breaks if > than 16 chars
		// #include "md5 codes"
		uint4 i0;
		uint a, b, c, d;
		/*
		input[0]='a';
		input[1]='d';
		input[2]='m';
		input[3]='i';
		input[4]='n';
		input[5]='\0';
		*/

		memcpy(&i0.x, input + 0, sizeof(i0.x));
		memcpy(&i0.y, input + 0 + sizeof(i0.x), sizeof(i0.y));
		memcpy(&i0.z, input + 0 + sizeof(i0.x) + sizeof(i0.z), sizeof(i0.z));

		md5_transform(i0, a, b, c , d);
		memcpy(retBuf, &a, sizeof(a));
		memcpy(retBuf + sizeof(a), &b, sizeof(b));
		memcpy(retBuf + sizeof(a) + sizeof(b), &c, sizeof(c));
		memcpy(retBuf + sizeof(a) + sizeof(b) + sizeof(c), &d, sizeof(d));

		unsigned long ihash=0;
		int match=0;
		
		for(ihash=0; ihash < *gpuHashListCount; ihash++) {
			
			if(!my_memcmp(retBuf, gpuHashList[ihash], 16)) {
				match=1;
				break;
			}
			
		}

		if(match==1)
		{	

			for(i=0; i < 16; i++) {
				matchHashList[matchCount][i] = retBuf[i];
			}
			

			for(i=0; i < inputLen; i++) {
				matchPassList[matchCount][i] = input[i];
			}
			matchHashList[matchCount][i] = '\0';
			
			matchCount++;
	
			if(matchCount + 1 > *maxSuccess)
				break;			
		}

		//if(threadIdx.x == 0)
		if(*start < iter)
			*start = iter;

		if(++count > maxIterations)
			break;
	}

    return;
}

// end our init and free all our memory, including btree, etc.

extern "C" __declspec(dllexport) void bl4ckJackGenerateGPU(int block, int thread, int shmem, double *start, double *stop, int maxIterations, char **gpuHashList, int *gpuHashListCount, int *matchCount) {

	//bl4ckJackGenerateGPUInternal<<<block,thread,shmem>>>(start, stop, maxIterations, matchCount);
	bl4ckJackGenerateGPUInternal<<<block,thread>>>(start, stop, maxIterations, gpuHashList, gpuHashListCount, matchCount);
	
	cudaThreadSynchronize();

	if(cudaGetLastError() != cudaSuccess) {
		OutputDebugString("CUDA Error: ");
		OutputDebugString(cudaGetErrorString(cudaGetLastError()));
		OutputDebugString("\n");
	}

	// copy success to and from and update passworsd per second
	return;
}


// btree functions
/*
struct node* newNode(struct node* parent, void *data, int dataLen) {
  struct node *node=NULL;
  cudaError_t err;
  
  err = cudaMalloc((void**)&node, sizeof(struct node));
  if( err != cudaSuccess)
	  return NULL;
  
  void *ptr = NULL;

  err = cudaMalloc((void **)&ptr, dataLen);
  if( err != cudaSuccess)
	  return NULL;

  if(ptr) {
	err = cudaMemcpy(ptr, &data, dataLen , cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;
  }
  else {
	cudaFree(ptr);
	cudaFree(node);
	return NULL;
  }
  
  //node->data = ptr;
  err = cudaMemcpy(node->data, &ptr, sizeof(ptr) , cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;

  //node->dataLen = dataLen;
  err = cudaMemcpy(&node->dataLen, &dataLen, sizeof(dataLen) , cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;

  //node->left = NULL;
  struct node *n=NULL;
  err = cudaMemcpy(&node->left, &n, sizeof(n), cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;

  //node->right = NULL;
  err = cudaMemcpy(&node->right, &n, sizeof(n) , cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;

  //node->parent = parent;
  err = cudaMemcpy(&node->parent, &parent, sizeof(parent), cudaMemcpyHostToDevice);
	if( err != cudaSuccess)
	  return NULL;

  return(node);
}

int lessThan(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {


	register unsigned int i=0;
	if(!base) return 0;
	if(!compare) return 0;

	while( i < baseLen && i < compareLen) {
		if(compare[i] > base[i])
			return 0;
		else if(compare[i] == base[i]) {
			i++;
			continue;
		} else
			return 1;
	}

	return 1;
}

__device__ int devicelessThan(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {


	register unsigned int i=0;
	if(!base) return 0;
	if(!compare) return 0;

	while( i < baseLen && i < compareLen) {
		if(compare[i] > base[i])
			return 0;
		else if(compare[i] == base[i]) {
			i++;
			continue;
		} else
			return 1;
	}

	return 1;
}
int greaterThan(unsigned char *base, size_t baseLen, unsigned char *compare, size_t compareLen) {
	register unsigned int i=0;
	if(!base) return 0;
	if(!compare) return 0;

	while(i < baseLen && i < compareLen) {
		if(compare[i] < base[i])
			return 0;
		else if(compare[i] == base[i]) {
			i++;
			continue;
		} else
			return 1;
	}
	return 1;
}

int Equals(unsigned char *base, int baseLen, unsigned char *compare, int compareLen) {

	register int i=0;
	if(!base) return 0;
	if(!compare) return 0;

	while(i < baseLen && i < compareLen) {
		if(base[i] != compare[i]) 
			return 0;
		i++;
	}
	return 1;
}

__device__ int deviceEquals(unsigned char *base, int baseLen, unsigned char *compare, int compareLen) {

	register int i=0;
	if(!base) return 0;
	if(!compare) return 0;

	while(i < baseLen && i < compareLen) {
		if(base[i] != compare[i]) 
			return 0;
		i++;
	}
	return 1;
}

__device__ struct node *first_node(struct node *tree) {
    struct node *tmp;
    while(tree){
        tmp = tree;
        tree = tree->left;
    }
    return tmp;
}

__device__ struct node *next_node(struct node *n){

    if(!n) return NULL;

    if(n->right)
        return first_node(n->right);

    while(n->parent && n->parent->right == n)
        n = n->parent;

    if(!n->parent)
        return NULL;

    return n;
}

__device__ int lookup(struct node *parentTree, void *target, int targetLen) {
	struct node *node = first_node(parentTree);
	while(node) {
		if (deviceEquals((unsigned char *)node->data, node->dataLen, (unsigned char *)target, targetLen))
			return 1;
			
		node = next_node(node);
	}
	return 0;
}
*/
/*
__device__ int lookup(struct node* node, void *target, int targetLen) {
  // 1. Base case == empty tree
  // in that case, the target is not found so return false
  if (node == NULL) {
		return 0;
  }
  else {
    // 2. see if found here
    if (deviceEquals((unsigned char *)node->data, node->dataLen, (unsigned char *)target, targetLen))
		return 1;
    else {
      // 3. otherwise recur down the correct subtree
      if (devicelessThan((unsigned char *)target, targetLen, (unsigned char *)node->data, node->dataLen) == 1)
		return(lookup2(node->left, (unsigned char *)target, targetLen));
      else 
		return(lookup2(node->right, (unsigned char *)target, targetLen));
    }
  }
}
*/
/*
struct node* insert(struct node* node, struct node *parentNode, void *data, int dataLen) {
  // 1. If the tree is empty, return a new, single node
  if (node == NULL) {
    return(newNode(parentNode, data, dataLen));
  }
  else {
    // 2. Otherwise, recur down the tree
    if (lessThan((unsigned char *)data, dataLen, (unsigned char *)node->data, node->dataLen) == 1
		|| Equals((unsigned char *)data, dataLen, (unsigned char *)node->data, node->dataLen == 1) )
		node->left = insert(node->left, node, data, dataLen);
    else node->right = insert(node->right, node, data, dataLen);
    return(node); // return the (unchanged) node pointer
  }
}

void destroy(struct node *p)
{
  cudaError_t err;
  if (p != 0)
    {
      destroy(p->left);
      destroy(p->right);
	  err = cudaFree(p->data);
	  if(err != cudaSuccess)
		  return;
      err = cudaFree(p);
	  if(err != cudaSuccess)
		  return;
    }
}
*/
