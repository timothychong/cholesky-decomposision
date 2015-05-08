#include "chsky_dec.h"
#include "math.h"
#include <pthread.h>
#include <stdio.h>

#define LOOP_UNROLL 4
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

// Pthread section //

// parameter to pass to the thread function
struct argument{
	data_t *A;
	 int len;
	 int thread_ID;
} ;

// Barrier variable
pthread_barrier_t barrier;

/*
 * thread worker function for Cholesky decomposition
 * @param thread_argument the struct that each thread receives as input
 */
void *worker_thread_optimized(void *thread_argument){

	// cast argument to struct pointer
	struct argument *my_argument = (struct argument*) thread_argument;
	
	// extract data from argument
	data_t* matrix 		= my_argument->A;
	const int len 		= my_argument->len;
	const int my_ID 	= my_argument->thread_ID;

	int i, j, k;
	int rc;
	int state = 0;
	float diagonal_element_reciprocal;
	float holder;

	// Go diagonally across the matrix
	for (k = 0; k < len; k++){
			

		// one thread is responsible for updating the elements along the diagonal
		if (my_ID == 0){ matrix(k, k) = sqrt(matrix(k, k));
			//printf("updating %d\n", count++);
		}

		// Make sure the diagonal element is updated before proceeding
  		rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    			printf("Could not wait on barrier\n");
    			exit(-1);
  		}
		
		// set diagonal_element_reciprocal once within this iteration and avoid divisions in the for loop
		diagonal_element_reciprocal = 1/matrix(k, k);

  		// Normalize the row to the left of the current diagonal element. 
  		state = 0;
  		for (i = k+1; i < len; i++){
  			// Once I have found a column to normalize, jump to next column in steps of NUM_THREADS 
  			if (i % NUM_THREADS == my_ID){
  				state = 1;
  				matrix(k, i) = matrix(k, i) * diagonal_element_reciprocal;
  			}
  			if(state == 1){
  				i += (NUM_THREADS-1);
  			}
  		}

  		// Make sure the whole row has been normalized before proceeding
  	  	rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    			printf("Could not wait on barrier\n");
    			exit(-1);
  		}

  		// Submatrix update: once I have found a row I am responsible for, jump to next row in steps of NUM_THREADS
  		state = 0;
  		for (i = k+1; i < len; i++){
  			if (i % NUM_THREADS == my_ID){
  				state = 1;
  				// update all the elements to the right in the row I'm responsible for.
				// variable to store the element in the normalized row
				holder = matrix(k, i);	// memory access only once
				j = i;

				// iterate by steps of one until we can start unrolling
				while ( (len-j)%LOOP_UNROLL != 0){
					matrix(i, j) = matrix(i, j) - holder * matrix(k, j);
					j++;
				}
				// unroll the loop
  				for (; j < len; j+=LOOP_UNROLL){
  					matrix(i, j) = matrix(i, j) - (holder * matrix(k, j));
					matrix(i, j+1) = matrix(i, j+1) - (holder * matrix(k, j+1));
					matrix(i, j+2) = matrix(i, j+2) - (holder * matrix(k, j+2));
					matrix(i, j+3) = matrix(i, j+3) - (holder * matrix(k, j+3));
					//matrix(i, j+4) = matrix(i, j+4) - (holder * matrix(k, j+4));
					//matrix(i, j+5) = matrix(i, j+5) - (holder * matrix(k, j+5));
					//matrix(i, j+6) = matrix(i, j+6) - (holder * matrix(k, j+6));
					//matrix(i, j+7) = matrix(i, j+7) - (holder * matrix(k, j+7));
  				}
  			}
  			if (state == 1){
  				i += (NUM_THREADS-1);
  			}
  		}

  		// Make sure the submatrix has been completely updated before moving on to the next iteration of k
  		rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    		printf("Could not wait on barrier\n");
    		exit(-1);
  		}
	}

	// done; exit thread
	pthread_exit(NULL);
}

/*
 *	pthread Cholesky decompostion
 *	This function spawns the threads responsible for computing the upper triangular
 *	@param matrix the matrix to decompose
 *	@param len the length of the matrix
 */
void chsky_dec_strip_optimized(data_t * matrix, const int len){

	// Array of threads and arguments for each
	pthread_t threads[NUM_THREADS];
	struct argument my_thread_arguments[NUM_THREADS];

	int t, rc;

  	// Barrier initialization 
 	if(pthread_barrier_init(&barrier, NULL, NUM_THREADS)) {
    		printf("Could not create a barrier\n");
    		return;
 	} 

 	// create the worker threads with unique arguments
	for (t = 0; t < NUM_THREADS; t++){
		my_thread_arguments[t].A = matrix;
		my_thread_arguments[t].len = len;
		my_thread_arguments[t].thread_ID = t;

		rc = pthread_create(&threads[t], NULL, worker_thread_optimized, (void*) &my_thread_arguments[t]);
		if (rc) {
      			printf("ERROR; return code from pthread_create() is %d\n", rc);
      			return;
    		}
	}

	// Join threads to assure all threads complete
  	for (t = 0; t < NUM_THREADS; t++) {
    		if (pthread_join(threads[t],NULL)){
      			printf("\n ERROR on join\n");
      			return;
    		}
  	}

}


// UNOPTIMIZED pthreads

/*
 * thread worker function for Cholesky decomposition (unoptomized)
 * @param thread_argument the struct that each thread receives as input
 */
void *worker_thread(void *thread_argument){

	// cast argument to struct pointer
	struct argument *my_argument = (struct argument*) thread_argument;
	
	// extract data from argument
	data_t* matrix 		= my_argument->A;
	const int len 		= my_argument->len;
	const int my_ID 	= my_argument->thread_ID;

	int i, j, k;
	int rc;
	int state = 0;

	// Go diagonally across the matrix
	for (k = 0; k < len; k++){

		// one thread is responsible for updating the elements along the diagonal
		if (my_ID == 0){
			matrix(k, k) = sqrt(matrix(k, k));
			//printf("updating %d\n", count++);
		}

		// Make sure the diagonal element is updated before proceeding
  		rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    			printf("Could not wait on barrier\n");
    			exit(-1);
  		}

  		// Normalize the row to the left of the current diagonal element. 
  		state = 0;
  		for (i = k+1; i < len; i++){
  			// Once I have found a column to normalize, jump to next column in steps of NUM_THREADS 
  			if (i % NUM_THREADS == my_ID){
  				state = 1;
  				matrix(k, i) = matrix(k, i) / matrix(k, k);
  			}
  			if(state == 1){
  				i += (NUM_THREADS-1);
  			}
  		}

  		// Make sure the whole row has been normalized before proceeding
  	  	rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    			printf("Could not wait on barrier\n");
    			exit(-1);
  		}

  		// Submatrix update: once I have found a column I am responsible for, jump to next column in steps of NUM_THREADS
  		state = 0;
  		for (i = k+1; i < len; i++){
  			if (i % NUM_THREADS == my_ID){
  				state = 1;
  				// update all the elements below the normalized row in the column I'm responsible for
  				for (j = i; j < len; j++){
  					matrix(i, j) = matrix(i, j) - matrix(k, i) * matrix(k, j);
  				}
  			}
  			if (state == 1){
  				i += (NUM_THREADS-1);
  			}
  		}

  		// Make sure the submatrix has been completely updated before moving on to the next iteration of k
  		rc = pthread_barrier_wait(&barrier);
  		if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
    			printf("Could not wait on barrier\n");
    			exit(-1);
  		}
	}

	// done; exit thread
	pthread_exit(NULL);
}

/*
 *	pthread Cholesky decompostion unoptomized version
 *	This function spawns the threads responsible for computing the upper triangular
 *	@param matrix the matrix to decompose
 *	@param len the length of the matrix
 */
void chsky_dec_strip(data_t * matrix, const int len){

	// Array of threads and arguments for each
	pthread_t threads[NUM_THREADS];
	struct argument my_thread_arguments[NUM_THREADS];

	int t, rc;

  	// Barrier initialization 
 	if(pthread_barrier_init(&barrier, NULL, NUM_THREADS)) {
    		printf("Could not create a barrier\n");
    		return;
 	} 

 	// create the worker threads with unique arguments
	for (t = 0; t < NUM_THREADS; t++){
		my_thread_arguments[t].A = matrix;
		my_thread_arguments[t].len = len;
		my_thread_arguments[t].thread_ID = t;

		rc = pthread_create(&threads[t], NULL, worker_thread, (void*) &my_thread_arguments[t]);
		if (rc) {
      			printf("ERROR; return code from pthread_create() is %d\n", rc);
      			return;
    		}
	}

	// Join threads to assure all threads complete
  	for (t = 0; t < NUM_THREADS; t++) {
    	if (pthread_join(threads[t],NULL)){
      			printf("\n ERROR on join\n");
      			return;
    		}
  	}

}
