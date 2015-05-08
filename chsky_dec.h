#ifndef __CHSKY_DEC_H___
#define __CHSKY_DEC_H___
#include "global.h"
#define NUM_THREADS 4

void chsky_dec_baseline(data_t * matrix, const int len);

void chsky_dec_strip(data_t * matrix, const int len);

void chsky_dec_block(data_t * matrix, const int len);


#endif
