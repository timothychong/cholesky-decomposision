#ifndef __MAIN_H__
#define __MAIN_H__

int init_matrix_zero(float * matrix, long int len);
void copy_matrix(data_t * src, data_t * dest, int len);
void print_matrix(data_t * mat, long int len);
void read_matrix_from_file(data_t ** matrix, const char filename[], int * len);

#endif
