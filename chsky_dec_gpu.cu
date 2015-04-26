#include "chsky_dec_gpu.h"
#include "gpu_help.h"

#define matrix(x,y) matrix[x * len + y]

__global__ void kernel_chsky_dec (data_t * matrix, int len) {
	const int x = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const int y = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;


}

void chsky_dec_gpu(data_t * matrix, const int len){

	const int NUM_THREADS_PER_BLOCK_PER_DIM = 16;
	const int NUM_BLOCKS_PER_DIM  = (NUM_THREADS_PER_BLOCK_PER_DIM + len - 1) /NUM_THREADS_PER_BLOCK_PER_DIM;
	const int CELL_PER_THREAD = len/NUM_THREADS_PER_BLOCK_PER_DIM;

	dim3 dimGrid(NUM_BLOCKS_PER_DIM, NUM_BLOCKS_PER_DIM);
	dim3 dimBlock(NUM_THREADS_PER_BLOCK_PER_DIM, NUM_THREADS_PER_BLOCK_PER_DIM, 1);

	kernel_chsky_dec<<<dimGrid, dimBlock>>>(matrix, len);
}
