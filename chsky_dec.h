#ifndef __CHSKY_DEC_H___
#define __CHSKY_DEC_H___
#include "global.h"

void chsky_dec_baseline(data_t * matrix, const int len);

void chsky_dec_strip(data_t * matrix, const int len);

void chsky_dec_block(data_t * matrix, const int len);

void chsky_dec_gpu(data_t * matrix, const int len);



#endif
