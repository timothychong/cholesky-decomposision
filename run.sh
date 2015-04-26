#gcc -O1 -std=c99 -o chsky_dec chsky_dec.c main.c -lrt -lm; ./chsky_dec
nvcc -o chsky_dec chsky_dec_gpu.cu chsky_dec.cu main.cu gpu_help.cu; ./chsky_dec
