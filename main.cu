#include "chsky_dec.h"
#include "chsky_dec_gpu.h"
#include "gpu_help.h"
#include "main.h"
#include <cstdio>

#define TOL 1e-4

int main() {
	const char filename[] = "matrix/500.txt";

	//READING FROM FILE
	data_t * input;
	data_t * input_for_pthread;
	int len;
	read_matrix_from_file(&input, filename, & len);
	//END READING FROM FILE

	printf("Matrix dimension: %d X %d\n", len, len);
	
	// Allocate GPU memory
	size_t allocSize = len * len * sizeof(data_t);
	data_t * d_matrix;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_matrix, allocSize));

	// Allocate arrays on host memory
	data_t *h_result = (float *) malloc(allocSize);

	// Copying input matrix for pthread version
	input_for_pthread = (data_t*) malloc(allocSize);
	copy_matrix(input, input_for_pthread, len);

	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;

	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);

	CUDA_SAFE_CALL(cudaMemcpy(d_matrix, input, allocSize, cudaMemcpyHostToDevice));

	chsky_dec_gpu(d_matrix, len);

	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_matrix, allocSize, cudaMemcpyDeviceToHost));

	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);

	printf("\nGPU time: %f (msec)\n", elapsed_gpu);

	cudaEventRecord(start, 0);

	chsky_dec_baseline(input, len);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nCPU time: %f (msec)\n", elapsed_gpu);

	cudaEventRecord(start, 0);
	
	chsky_dec_strip(input_for_pthread, len);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\npthread time (%d threads): %f (msec)\n", NUM_THREADS, elapsed_gpu);

	int errCount = 0, zeroCount = 0;
	float max_diff = 0;

	for(int i = 0; i < len * len; i++) {
		double diff = abs(h_result[i] - input[i]);
		if (diff > TOL) {
			if(diff > max_diff)
					max_diff = diff;
			errCount++;
		}
		if (h_result[i] == 0) {
			zeroCount++;
		}
	}
	printf("\nMaximum difference: %f", max_diff);
	
	if (errCount > 0) {
		printf("\n@ERROR: GPU TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: GPU TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nGPU TEST PASSED: All results matched\n");
	}


	for(int i = 0; i < len * len; i++) {
		double diff = abs(input_for_pthread[i] - input[i]);
		if (diff > TOL) {
			if(diff > max_diff)
					max_diff = diff;
			errCount++;
		}
		if (input_for_pthread[i] == 0) {
			zeroCount++;
		}
	}
	printf("\nMaximum difference: %f", max_diff);
	
	if (errCount > 0) {
		printf("\n@ERROR: PTHREAD TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: PTHREAD TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nPTHREAD TEST PASSED: All results matched\n");
	}

	CUDA_SAFE_CALL(cudaFree(d_matrix));
	free(h_result);
	free(input);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


void read_matrix_from_file(data_t ** matrix, const char filename[], int * len) {

	FILE * file = fopen(filename, "r");
	const int BUFFER_SIZE = 1e5;
	char buffer[BUFFER_SIZE];
	FILE * line = fmemopen(buffer, BUFFER_SIZE, "r");

	// GETING LENGTH
	*len = 0;
	if(fgets(buffer, BUFFER_SIZE, file) != NULL){
		data_t dummy;
		while(fscanf(line, "%f", &dummy)) {
			(*len)++;
		}
	}
	fclose(line);
	fclose(file);

	*matrix = (data_t *) malloc(sizeof(data_t) * (*len) * (*len));
	//IMPORTING matrix
	file = fopen(filename, "r");
	int i = 0;
	int code = fscanf(file, "%f", (*matrix + i));
	while(code && code != EOF){
		i++;
		code = fscanf(file, "%f", (*matrix + i));
	}
	fclose(file);
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
