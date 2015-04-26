#include "chsky_dec.h"
#include "math.h"

void chsky_dec_baseline(data_t * input, data_t * output, const int len){
	for(int i = 0; i < len; i++) {
		for(int j = 0; j < i; j++) {
				data_t result = input[i * len + j];
				for(int k = 0; k < j; k++){
						data_t temp_1 = output[i * len + k];
						data_t temp_2 = output[j * len + k];
						result = result - temp_1 * temp_2;
				}
				result = result / output[j * len + j];
				output[i * len + j] = result;
		}
		data_t result_diag = input[i * len + i];
		for(int k = 0; k < i; k++){
			 data_t temp = output[i * len + k];
			 result_diag = result_diag - temp * temp;
		}
		result_diag = sqrt(result_diag);
		output[i * len + i] = result_diag;
	}
}
