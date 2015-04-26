#include "chsky_dec.h"
#include "chsky_dec_gpu.h"
#include "gpu_help.h"
#include "main.h"
#include <cstdio>

int main() {

	const int len = 3;
	//data_t input[] = {4,12,-16,12,47,-43,-16, -43, 98};
	data_t input[] = {25, 15, -5, 0, 18,  0, 0,  0, 11};
	data_t input2[len * len];
	copy_matrix(input, input2, len);
	//data_t input[] = {18, 22,  54,  42, 22, 70,  86,  62, 54, 86, 174, 134, 42, 62, 134, 106};
	print_matrix(input, len);
	//chsky_dec_baseline(input, len);
	printf("\n");
	//print_matrix(input, len);

	//Allocating memory on 
	size_t allocSize = len * len * sizeof(data_t);
	
	// Allocate GPU memory
	data_t * d_matrix;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_matrix, allocSize));

	// Allocate arrays on host memory
	data_t *h_result = (float *) malloc(allocSize);

	CUDA_SAFE_CALL(cudaMemcpy(d_matrix, input2, allocSize, cudaMemcpyHostToDevice));

	chsky_dec_gpu(d_matrix, len);

	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_matrix, allocSize, cudaMemcpyDeviceToHost));

	print_matrix(h_result, len);

	CUDA_SAFE_CALL(cudaFree(d_matrix));
	free(h_result);
}



int init_matrix_zero(data_t * matrix, long int len)
{
  long int i;
  if (len > 0) {
    for (i = 0; i < len*len; i++)
      matrix[i] = 0;
    return 1;
  }
  else return 0;
}

void copy_matrix(data_t * src, data_t * dest, int len) {
	int i;
	for(i = 0 ; i < len * len; i ++ ) {
		*(dest + i) = *(src + i);
	}
}

void print_matrix(data_t * mat, long int len) {
	printf("\n");
	int i;
	for (i = 0; i < len * len; i++) {
		if(i % len == 0)
			printf("\n"); 
		//printf("%.2f, ", mat[i]);
		printf("%f, ", mat[i]);
	}
	printf("\n");
}
