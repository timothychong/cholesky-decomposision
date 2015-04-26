#include "chsky_dec.h"
#include "chsky_dec_gpu.h"
#include "gpu_help.h"
#include "main.h"
#include <cstdio>



int main() {

	const char filename[] = "matrix.txt";

	data_t * input;
	int len;

	read_matrix_from_file(&input, filename, & len);
	print_matrix(input, len);

	data_t input2[len * len];
	copy_matrix(input, input2, len);
	print_matrix(input, len);
	chsky_dec_baseline(input, len);
	printf("\n");
	print_matrix(input, len);

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
	free(input);
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
