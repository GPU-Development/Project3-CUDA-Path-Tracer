#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
#include "efficient.h"

#define SHARED_MEMORY 0

#define MAX_ARRAY_SIZE 1024
#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace StreamCompaction {
namespace Efficient {


__global__ void scanBlock(int n, int *odata, const int *idata) {

	extern __shared__ int temp[];  // allocated on invocation  

	int blid = blockIdx.x * blockDim.x;
	int thid = threadIdx.x;

	if (blid + thid >= n/2) {
		return;
	}

	int offset = 1;

	int ai = thid;
	int bi = thid + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = idata[ai]; // load input into shared memory  
	temp[bi + bankOffsetB] = idata[bi];

	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			ai = offset*(2 * thid + 1) - 1;
			bi = offset*(2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}


	if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }   // clear the last element 

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			ai = offset*(2 * thid + 1) - 1;
			bi = offset*(2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	// write results to device memory  

	odata[ai + blid] = temp[ai + bankOffsetA];
	odata[bi + blid] = temp[bi + bankOffsetB];
}

__global__ void scanMultipleBlocks(int n, int *odata, const int *idata) {
}

__global__ void kernUpStep(int n, int d, int *data) {

	int s = 1 << (d + 1);
	int index = (threadIdx.x + (blockIdx.x * blockDim.x)) *s;

	if (index >= n) {
		return;
	}

	data[index + s - 1] += data[index + s / 2 - 1];
}

__global__ void kernDownStep(int n, int d, int *data) {
	
	int s = 1 << (d + 1);
	int index = (threadIdx.x + (blockIdx.x * blockDim.x)) * s;
	
	if (index >= n) {
		return;
	}


	int t = data[index + s / 2 - 1];
	data[index + s / 2 - 1] = data[index + s - 1];
	data[index + s - 1] += t;
}


/**
* Performs prefix-sum (aka scan) on idata, storing the result into odata.
* For use with arrays intiialized on GPU already.
*/
void scan_dev(int n, int *dev_in) {



#if SHARED_MEMORY
	// create device arrays to pad to power of 2 size array
	//int pot = pow(2, ilog2ceil(n));
	int pot = 1 << ilog2ceil(n);
	int *dev_data;
	int host[512] = {0};

	cudaMalloc((void**)&dev_data, pot*sizeof(int));
	cudaMemset(dev_data, 0, pot*sizeof(int));
	cudaMemcpy(dev_data, dev_in, n*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(host, dev_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	dim3 fullBlocksPerGrid((n + MAX_ARRAY_SIZE - 1) / MAX_ARRAY_SIZE);
	scanBlock << < fullBlocksPerGrid, MAX_ARRAY_SIZE, MAX_ARRAY_SIZE >> >(pot, dev_data, dev_in);
	cudaMemcpy(host, dev_data, 512 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(dev_in, dev_data, n*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaFree(dev_data);

#else
	dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// create device arrays to pad to power of 2 size array
	//int pot = pow(2, ilog2ceil(n));
	int pot = 1 << ilog2ceil(n);
	int *dev_data;
	cudaMalloc((void**)&dev_data, pot*sizeof(int));
	cudaMemset(dev_data, 0, pot*sizeof(int));
	cudaMemcpy(dev_data, dev_in, n*sizeof(int), cudaMemcpyDeviceToDevice);

	float d = 0;
	for (d; d < ilog2ceil(pot); d++) {
		int div = 1 << (int) d + 1;
		fullBlocksPerGrid.x = ((pot / div + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernUpStep << < fullBlocksPerGrid, BLOCK_SIZE >> >(pot, d, dev_data);
	}

	cudaMemset(&dev_data[pot - 1], 0, sizeof(int));
	for (d = ilog2ceil(pot); d >= 0; d--) {
		int div = 1 << (int) d + 1;
		fullBlocksPerGrid.x = ((pot / div + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernDownStep << < fullBlocksPerGrid, BLOCK_SIZE >> >(pot, d, dev_data);
	}

	cudaMemcpy(dev_in, dev_data, n*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaFree(dev_data);
#endif

}


int compact_dev(int n, int *dev_out, const int *dev_in) {

	// create device arrays
	int *dev_indices;
	int *dev_bools;
	int rtn = -1;

	//int n_iter = 1 << 24;

	cudaMalloc((void**)&dev_indices, n*sizeof(int));
	cudaMalloc((void**)&dev_bools, n*sizeof(int));
	
	dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	cudaMemset(dev_indices, 0, n);
	cudaMemset(dev_bools, 0, n);

	StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, BLOCK_SIZE >> >(n, dev_bools, dev_in);

	// scan without wasteful device-host-device write
	cudaMemcpy(dev_indices, dev_bools, n*sizeof(int), cudaMemcpyDeviceToDevice);
	scan_dev(n, dev_indices);

	// scatter
	StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, BLOCK_SIZE >> >(n, dev_out, dev_in, dev_bools, dev_indices);
	cudaMemcpy(&rtn, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
	
	// Synch and throw away run time error that thinks we can't handle arrays over 2^16
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	
	cudaFree(dev_bools);
	cudaFree(dev_indices);

	return rtn + 1;
}


}
}
