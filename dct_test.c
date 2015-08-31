#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmpReader.h"

#define numColor 3
#define N 8


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

__global__ void dct(unsigned char *red, float *dct, int numRowsint, int numCols, float *DCTv8matrix, float *DCTv8matrixT )
{
        const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

        const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

        __shared__ float s_data[N*N];
        int i = 0,j = 0;
        for (i = 0; i < N; ++i)
        {
                s_data[threadIdx.x * blockDim.x + threadIdx.y] = DCTv8matrixT[threadIdx * blockDim.x + i] * red[threadIdx + i]
        }
        __syncthreads(); // check if this for the threads in a block 

        for (i = 0; i < N; ++i)
        {
                s_data[threadIdx.x * blockDim.x + threadIdx.y] = DCTv8matrixT[threadIdx * blockDim.x + i] * red[threadIdx]
        }

}


int main(int argc, char *argv[])
{


        BMPData *image1 = NULL;
        int i = 0,j = 0;
        int size = 0;
        unsigned char *red = NULL,*green = NULL,*blue = NULL;
        unsigned char *d_red = NULL,*d_green = NULL,*d_blue = NULL;

        float *d_red_dct = NULL;
        float *red_dct = NULL;

        float *d_DCTv8matrix = NULL;
        float *d_DCTv8matrixT = NULL;

        const float DCTv8matrix[N*N] = {
        0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f, 
        0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f, 
        0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f, 
        0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f, 
        0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f, 
        0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f, 
        0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f, 
        0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
        };

        const float DCTv8matrixT[N*N] = {
        0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f,  0.3535533905932738f, 
        0.4903926402016152f,  0.4157348061512726f,  0.2777851165098011f,  0.0975451610080642f, -0.0975451610080641f, -0.2777851165098010f, -0.4157348061512727f, -0.4903926402016152f, 
        0.4619397662556434f,  0.1913417161825449f, -0.1913417161825449f, -0.4619397662556434f, -0.4619397662556434f, -0.1913417161825452f,  0.1913417161825450f,  0.4619397662556433f, 
        0.4157348061512726f, -0.0975451610080641f, -0.4903926402016152f, -0.2777851165098011f,  0.2777851165098009f,  0.4903926402016153f,  0.0975451610080640f, -0.4157348061512721f, 
        0.3535533905932738f, -0.3535533905932737f, -0.3535533905932738f,  0.3535533905932737f,  0.3535533905932738f, -0.3535533905932733f, -0.3535533905932736f,  0.3535533905932733f, 
        0.2777851165098011f, -0.4903926402016152f,  0.0975451610080642f,  0.4157348061512727f, -0.4157348061512726f, -0.0975451610080649f,  0.4903926402016152f, -0.2777851165098008f, 
        0.1913417161825449f, -0.4619397662556434f,  0.4619397662556433f, -0.1913417161825450f, -0.1913417161825453f,  0.4619397662556437f, -0.4619397662556435f,  0.1913417161825431f, 
        0.0975451610080642f, -0.2777851165098011f,  0.4157348061512727f, -0.4903926402016153f,  0.4903926402016152f, -0.4157348061512720f,  0.2777851165098022f, -0.0975451610080625f
        };

        //unsigned char  test[8][8] = {{48,39,40,68,60,38,50,121} , {149,82,79,101,113,106,27,62} , {58,63,77,69,124,107,74,125} , {80,97,74,54,59,71,91,66} , {18,34,33,46,64,61,32,37} , {149,108,80,106,116,61,73,92} , {211,233,159,88,107,158,161,109} , {212,104,40,44,71,136,113,66} };

        dim3 grid(64,64);            // defines a grid of 256 x 1 x 1 blocks
        dim3 block(8,8); 


        if((image1 = readBMPfile(argv[1])) == NULL)
        {
                printf("Error in File 1\n");
                return -1;
        }

        size = image1->infoHeader.height * image1->infoHeader.width;

        red = (unsigned char *) malloc(size * sizeof(char) );
        blue = (unsigned char *) malloc(size * sizeof(char) );
        green = (unsigned char *) malloc(size * sizeof(char) );


        red_dct = (float *) malloc(size * sizeof(float*) );

        for (i = 0; i < size * numColor; i+=3)
        {
                        blue[i] = image1->bitMapImage[i]; 
                        green[i] = image1->bitMapImage[i + 1];
                        red[i] = image1->bitMapImage[i + 2];
        }

        gpuErrchk( cudaMalloc( (void**)&d_red, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_green, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue, size * sizeof(char) ));

        gpuErrchk( cudaMalloc( (void**)&d_red_dct, size * sizeof(float) ));

        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrix, N * N * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrixT, N * N * sizeof(float) ));


        gpuErrchk(cudaMemcpy( d_red, &red, size * sizeof(char) , cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_green, &green, size * sizeof(char), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_blue, &blue, size * sizeof(char), cudaMemcpyHostToDevice ));

        gpuErrchk(cudaMemcpy( d_DCTv8matrix, &DCTv8matrix, N * N * sizeof(float), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_DCTv8matrixT, &DCTv8matrixT, N * N * sizeof(float), cudaMemcpyHostToDevice ));

        dct<<< grid, block >>>(d_red, d_red_dct, image1->infoHeader.height,
                                image1->infoHeader.width, d_DCTv8matrix, d_DCTv8matrixT);

        gpuErrchk(cudaMemcpy( &red_dct, d_red_dct, size * sizeof(char), cudaMemcpyDeviceToHost ));
        
        for (i = 0; i < image1->infoHeader.height ; ++i)
        {
                for (j = 0; j < image1->infoHeader.width ; ++j)
                        printf("%c ",red_dct[i*image1->infoHeader.height + j ]);
                printf("\n");
        }
        gpuErrchk(cudaFree( d_red ));
        gpuErrchk(cudaFree( d_green ));
        gpuErrchk(cudaFree( d_blue ));
        gpuErrchk(cudaFree( d_red_dct ));
        return 0;



}
