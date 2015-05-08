#ifndef __CHSKY_DEC_H__
#define __CHSKY_DEC_H__
#include "global.h"

void chsky_dec_gpu(data_t * matrix, const int len);
void chsky_dec_gpu_shared(data_t * matrix, const int len);

#endif

