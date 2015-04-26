#ifndef __GPU_HELP_H__
#define __GPU_HELP_H__

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define IMUL(a, b) __mul24(a, b)

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif

