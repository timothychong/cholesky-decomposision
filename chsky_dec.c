#include "chsky_dec.h"
#include "math.h"

#define matrix(x,y) matrix[x * len + y]

void chsky_dec_baseline(data_t * matrix, const int len){
	for(int k=0; k < len; k++) {
		matrix(k,k) = sqrt(matrix(k, k));
		for(int j = k + 1; j < len; j++) {
			matrix(k,j) = matrix(k,j) / matrix[k * len + k];
		}
		for(int i = k + 1; i < len; i++) {
			for(int j = i; j <len; j++) {
				matrix(i, j) = matrix(i,j) - matrix(k,i)  * matrix(k,j);
			}
		}
	}
}
