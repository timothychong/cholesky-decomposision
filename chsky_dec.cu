#include "chsky_dec.h"
#include "math.h"

#define matrix(x,y) matrix[x * len + y]

void chsky_dec_baseline(data_t * matrix, const int len){
	int k, j, i;
	for(k=0; k < len; k++) {
		matrix(k,k) = sqrt(matrix(k, k));
		for(j = k + 1; j < len; j++) {
			matrix(k,j) = matrix(k,j) / matrix(k,k);
		}
		for(i = k + 1; i < len; i++) {
			for(j = i; j <len; j++) {
				matrix(i, j) = matrix(i,j) - matrix(k,i)  * matrix(k,j);
			}
		}
	}
}

void chsky_dec_strip(data_t * matrix, const int len){

	//TOOD JOHN

}

void chsky_dec_block(data_t * matrix, const int len){

	//TODO JOHN

}
