#include "chsky_dec_gpu.h"
#include "gpu_help.h"

#define matrix(x,y) matrix[x * len + y]


__global__ void kernel_sqrt(data_t * matrix, int k, int len){
	matrix(k,k) = sqrt(matrix(k,k));
}

__global__ void kernel_norm(data_t * matrix, int k, int len){
	const int col = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	if (col > k && col < len) {
		matrix(k, col) = matrix(k, col) / matrix(k, k);
	}
}

__global__ void kernel_chsky_outerprod (data_t * matrix, int k, int len) {

	const int col = k + IMUL(blockDim.x, blockIdx.x) + threadIdx.x + 1;
	const int row = k + IMUL(blockDim.y, blockIdx.y) + threadIdx.y + 1;

	if (col >= row && col < len && row < len) {
		matrix(row, col) = matrix(row, col) - matrix(k, row) * matrix(k, col);
	}
}

void chsky_dec_gpu(data_t * matrix, const int len){

	const int NUM_THREADS_PER_BLOCK_PER_DIM = 16;
	const int NUM_BLOCKS_PER_DIM  = (NUM_THREADS_PER_BLOCK_PER_DIM + len - 1) /NUM_THREADS_PER_BLOCK_PER_DIM;

	dim3 dimGrid(NUM_BLOCKS_PER_DIM, NUM_BLOCKS_PER_DIM);
	dim3 dimBlock(NUM_THREADS_PER_BLOCK_PER_DIM, NUM_THREADS_PER_BLOCK_PER_DIM, 1);

	const dim3 dimGrid_norm(NUM_BLOCKS_PER_DIM, 1);
	const dim3 dimBlock_norm(NUM_THREADS_PER_BLOCK_PER_DIM, 1, 1);

	//printf("%d, %d\n", NUM_BLOCKS_PER_DIM, NUM_THREADS_PER_BLOCK_PER_DIM);

	for(int k = 0; k < len; k++) {
		kernel_sqrt<<<1, 1>>>(matrix, k, len);
		kernel_norm<<<dimGrid_norm, dimBlock_norm>>>(matrix, k, len);
		const int grid_length = (NUM_THREADS_PER_BLOCK_PER_DIM + (len - k - 1)- 1) /NUM_THREADS_PER_BLOCK_PER_DIM;
		const dim3 innerGrid(grid_length, grid_length);

		kernel_chsky_outerprod<<<innerGrid, dimBlock>>>(matrix, k, len);
	}
}



