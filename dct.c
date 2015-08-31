#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// gpu function call wrapper 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// go through this - http://cuda-programming.blogspot.in/2013/01/handling-cuda-error-messages.html

__global__ void add(int *a,int *b, int *c)
{
        *c = *a + *b;
}

int main(int argc, char *argv[])
{
        int *d_a,*d_b,*d_c;
        int a = 10,b = 12,c = 0;
        int size = sizeof( int );

        gpuErrchk( cudaMalloc( (void**)&d_a, size ));
        gpuErrchk( cudaMalloc( (void**)&d_b, size ));
        gpuErrchk( cudaMalloc( (void**)&d_c, size ));


        cudaMemcpy( d_a, &a, size, cudaMemcpyHostToDevice );
        cudaMemcpy( d_b, &b, size, cudaMemcpyHostToDevice );

        add<<< 1, 1 >>>(d_a, d_b, d_c);

        cudaMemcpy( &c, d_c, size, cudaMemcpyDeviceToHost );
        printf("Result: %d\n",c);
        cudaFree( d_a );
        cudaFree( d_b );
        cudaFree( d_c );
        return 0;



}
