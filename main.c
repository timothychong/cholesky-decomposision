#include "chsky_dec.h"
#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	//data_t input[] = {4,12,-16,12,47,-43,-16, -43, 98};
	//data_t input[] = {25, 15, -5, 15, 18,  0, -5,  0, 11};
	data_t input[] = {18, 22,  54,  42, 22, 70,  86,  62, 54, 86, 174, 134, 42, 62, 134, 106};
	print_matrix(input, 4);

	chsky_dec_baseline(input, 4);

	printf("\n");
	print_matrix(input, 4);

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
	for (int i = 0; i < len * len; i++) {
		if(i % len == 0)
			printf("\n"); 
		//printf("%.2f, ", mat[i]);
		printf("%f, ", mat[i]);
	}
	printf("\n");
}
