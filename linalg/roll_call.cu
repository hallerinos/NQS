#include <stdio.h>

__global__ void roll_call() {
	const int threadIndex = threadIdx.x;
	printf("Thread %d here!\n", threadIndex);
}

__global__ void sm_roll_call() {
	const int threadIndex = threadIdx.x;
	
	uint streamingMultiprocessorId;
	asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId) );
	
	printf("Thread %d running on SM %d!\n", threadIndex, streamingMultiprocessorId);
}

__global__ void warp_roll_call() {

	const int threadIndex = threadIdx.x;
	
	uint streamingMultiprocessorId;
	asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId));
	
	uint warpId;
	asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpId));
	
	uint laneId;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneId));
	
	printf("SM: %d | Warp: %d | Lane: %d | Thread %d - Here!\n", streamingMultiprocessorId, warpId, laneId, threadIndex);
}

__global__ void array_increment(int* in) {
	const int threadIndex = threadIdx.x;
	in[threadIndex] = in[threadIndex] + 1;
}

void printArray(int* array, int arraySize) {
	printf("[");
	for (int i = 0; i < arraySize; i++) {
		printf("%d", array[i]);
		if (i < arraySize - 1) {
			printf(", ");
		}
	}
	printf("]\n");
}

int mlauncher() {
	const int arraySize = 1024;

	warp_roll_call<<<1, arraySize>>>();	// if arraySize > 2**10, roll_call doesn't output.
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

	// Allocate host memory for the input array.
	//
	// The amount of memory allocated is equal to 
	// the length of the array times the size of 
	// an integer
	int* array = (int*)malloc(arraySize * sizeof(int));

	// Initialize the input array with values 0, 10, 20, 30, ...
	for (int i = 0; i < arraySize; i++) {
		array[i] = i;
	}

	// Allocate GPU memory for the input array
	int* d_array;
	cudaMalloc(&d_array, arraySize * sizeof(int));

	// Copy the input array from host memory to GPU memory
	cudaMemcpy(d_array, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	printf("Before: ");
	printArray(array, arraySize);

	array_increment<<<1, arraySize>>>(d_array);

	// Copy the result array from GPU memory back to host memory
	cudaMemcpy(array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

	printf("After: ");
	printArray(array, arraySize);
	
	// Free the host and GPU memory
	free(array);
	cudaFree(d_array);

	return 0;
}


int main() {
	mlauncher();
	return 0;
}