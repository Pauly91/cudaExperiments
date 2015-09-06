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

__global__ void dct(unsigned char *red, float *dct, float *idct, int numRowsint, int numCols, float *DCTv8matrix, float *DCTv8matrixT )
{
        const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                                blockIdx.y * blockDim.y + threadIdx.y);

        const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
        int i = 0;
        float temp = 0;
        __shared__ float ATX[N*N];
        for (i = 0; i < N*N; ++i)
        {
            ATX[i] = 0;
            dct[i] = 0;
        }

        __syncthreads();
        // y * column + x 
        // y down 
        // x right


        for (i = 0; i < N; ++i)
        {
                // printf("-->threadIdx.x:%d threadIdx.y:%d index1: %d index2: %d  index3: %d i:%d\n",
                //     threadIdx.x,threadIdx.y,threadIdx.y * blockDim.x + threadIdx.x,threadIdx.y * blockDim.x + i,i * blockDim.x + threadIdx.x,i);
                //ATX[threadIdx.y * blockDim.x + threadIdx.x] += DCTv8matrixT[threadIdx.y * blockDim.x + i] * red[blockIdx.x * blockDim.x + i * gridDim.x + threadIdx.x ];
                ATX[threadIdx.y * blockDim.x + threadIdx.x] += DCTv8matrixT[threadIdx.y * blockDim.x + i] * red[i * blockDim.x + threadIdx.x];
        }       // account for the 8 rows in each block in red[]
        __syncthreads(); // check if this for the threads in a block 

        // check if this multiplcation is correct   



        for (i = 0; i < N; ++i)
        {
                 printf("ATX[%d]:%f DCTv8matrix [%d]: %f\n",threadIdx.y * blockDim.x + i,ATX[threadIdx.y * blockDim.x + i], i * blockDim.x + threadIdx.x,DCTv8matrix [i * blockDim.x + threadIdx.x]);
                 dct[threadIdx.y * blockDim.x + threadIdx.x] += ATX[threadIdx.y * blockDim.x + i] * DCTv8matrix [i * blockDim.x + threadIdx.x];
        }

// __syncthreads();
//         for (i = 0; i < N*N; ++i)
//         {
//             ATX[i] = 0;
//         }

//         __syncthreads();
//         // y * column + x 
//         // y down 
//         // x right

//         for (i = 0; i < N; ++i)
//         {
//                 // printf("-->threadIdx.x:%d threadIdx.y:%d index1: %d index2: %d  index3: %d i:%d\n",
//                 //     threadIdx.x,threadIdx.y,threadIdx.y * blockDim.x + threadIdx.x,threadIdx.y * blockDim.x + i,i * blockDim.x + threadIdx.x,i);
//                 //ATX[threadIdx.y * blockDim.x + threadIdx.x] += DCTv8matrixT[threadIdx.y * blockDim.x + i] * red[blockIdx.x * blockDim.x + i * gridDim.x + threadIdx.x ];
//                 ATX[threadIdx.y * blockDim.x + threadIdx.x] += DCTv8matrix[threadIdx.y * blockDim.x + i] * dct[i * blockDim.x + threadIdx.x];
//         }       // account for the 8 rows in each block in red[]

//         __syncthreads();

//         for (i = 0; i < N; ++i)
//         {
//                  idct[threadIdx.y * blockDim.x + threadIdx.x] += ATX[threadIdx.y * blockDim.x + i] * DCTv8matrixT[i * blockDim.x + threadIdx.x];
//         }

//         // if(threadIdx.y == 0 || threadIdx.x ==0)
//         //     dct[threadIdx.y * blockDim.x + threadIdx.x] = dct[threadIdx.y * blockDim.x + threadIdx.x]/ (1.41);
//         // else
//         //      dct[threadIdx.y * blockDim.x + threadIdx.x] = dct[threadIdx.y * blockDim.x + threadIdx.x]/4;
}


int main(int argc, char *argv[])
{


        BMPData *image1 = NULL;
        int i = 0,j = 0;
        int size = 0;
        unsigned char *red = NULL,*green = NULL,*blue = NULL;
        unsigned char *d_red = NULL,*d_green = NULL,*d_blue = NULL;

        float *d_red_dct = NULL;

        unsigned char  *d_test = NULL;

        float *d_test_dct = NULL;
        float *d_test_idct = NULL;
        float *red_dct = NULL;

        float *test_dct = NULL;
        float *test_idct = NULL;


        float *d_DCTv8matrix = NULL;
        float *d_DCTv8matrixT = NULL;

        const int value = 0;

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



//          const float DCTv8matrix[N*N] = {
// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202, 

// 0.346759961331, 0.415734806151, 0.27778511651, 0.0975451610081, -0.0975451610081, -0.27778511651, -0.415734806151, -0.490392640202
// };

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



        // const float DCTv8matrixT[N*N] = { 
        // 0.245196320101, 0.245196320101, 0.245196320101, 0.245196320101, 0.245196320101, 0.245196320101, 0.245196320101, 0.245196320101, 

        // 0.146984450302, 0.146984450302, 0.146984450302, 0.146984450302, 0.146984450302, 0.146984450302, 0.146984450302, 0.146984450302, 

        // 0.0982118697984, 0.0982118697984, 0.0982118697984, 0.0982118697984, 0.0982118697984, 0.0982118697984, 0.0982118697984, 0.0982118697984, 

        // 0.0344874224104, 0.0344874224104, 0.0344874224104, 0.0344874224104, 0.0344874224104, 0.0344874224104, 0.0344874224104, 0.0344874224104, 

        // -0.0344874224104, -0.0344874224104, -0.0344874224104, -0.0344874224104, -0.0344874224104, -0.0344874224104, -0.0344874224104, -0.0344874224104, 

        // -0.0982118697984, -0.0982118697984, -0.0982118697984, -0.0982118697984, -0.0982118697984, -0.0982118697984, -0.0982118697984, -0.0982118697984, 

        // -0.146984450302, -0.146984450302, -0.146984450302, -0.146984450302, -0.146984450302, -0.146984450302, -0.146984450302, -0.146984450302, 

        // -0.173379980665, -0.173379980665, -0.173379980665, -0.173379980665, -0.173379980665, -0.173379980665, -0.173379980665, -0.173379980665
        // };

        unsigned char  test[64] = {48,39,40,68,60,38,50,121,149,82,79,101,113,106,27,62,58,63,77,69,124,107,74,125,80,97,74,54,59,71,91,66,18,34,33,46,64,61,32,37,149,108,80,106,116,61,73,92,211,233,159,88,107,158,161,109,212,104,40,44,71,136,113,66};

        dim3 grid(64,64);            // defines a grid of 256 x 1 x 1 blocks
        dim3 block(8,8); 


        // if((image1 = readBMPfile(argv[1])) == NULL)
        // {
        //         printf("Error in File 1\n");
        //         return -1;
        // }

        // size = image1->infoHeader.height * image1->infoHeader.width;
        size = 64;
        red = (unsigned char *) malloc(size * sizeof(char) );
        blue = (unsigned char *) malloc(size * sizeof(char) );
        green = (unsigned char *) malloc(size * sizeof(char) );


        red_dct = (float *) malloc(size * sizeof(float) );

        test_dct = (float *) malloc(size * sizeof(float) );
        test_idct = (float *) malloc(size * sizeof(float) );


        // for (i = 0; i < size * numColor; i+=3)
        // {
        //                 blue[i] = image1->bitMapImage[i]; 
        //                 green[i] = image1->bitMapImage[i + 1];
        //                 red[i] = image1->bitMapImage[i + 2];
        // }

        gpuErrchk( cudaMalloc( (void**)&d_red, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_green, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue, size * sizeof(char) ));

        gpuErrchk( cudaMalloc( (void**)&d_red_dct, size * sizeof(float) ));

        gpuErrchk( cudaMalloc( (void**)&d_test_dct, 64 * sizeof(float) ));

        gpuErrchk( cudaMalloc( (void**)&d_test_idct, 64 * sizeof(float) ));

        gpuErrchk( cudaMemset(d_test_dct,value, 64 * sizeof(float)));

        gpuErrchk( cudaMemset(d_test_idct,value, 64 * sizeof(float)));

        gpuErrchk( cudaMalloc( (void**)&d_test, 64 * sizeof(unsigned char) ));


        gpuErrchk( cudaMalloc( (void**)&d_red_dct, size * sizeof(float) ));

        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrix, N * N * sizeof(float) ));
        gpuErrchk( cudaMalloc( (void**)&d_DCTv8matrixT, N * N * sizeof(float) ));


        gpuErrchk(cudaMemcpy( d_red, red, size * sizeof(char) , cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_green, green, size * sizeof(char), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_blue, blue, size * sizeof(char), cudaMemcpyHostToDevice ));

        gpuErrchk(cudaMemcpy( d_test, test, size * sizeof(unsigned char), cudaMemcpyHostToDevice ));

        gpuErrchk(cudaMemcpy( d_DCTv8matrix, &DCTv8matrix, N * N * sizeof(float), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_DCTv8matrixT, &DCTv8matrixT, N * N * sizeof(float), cudaMemcpyHostToDevice ));

        // dct<<< grid, block >>>(d_red, d_red_dct, image1->infoHeader.height,
        //                         image1->infoHeader.width, d_DCTv8matrix, d_DCTv8matrixT);

        dct<<< 1, block >>>(d_test, d_test_dct, d_test_idct, 8,8, d_DCTv8matrix, d_DCTv8matrixT);

        //gpuErrchk(cudaMemcpy( &red_dct, d_red_dct, size * sizeof(float), cudaMemcpyDeviceToHost ));

        gpuErrchk(cudaMemcpy(test_dct, d_test_dct, 64 * sizeof(float), cudaMemcpyDeviceToHost ));
        gpuErrchk(cudaMemcpy(test_idct, d_test_idct, 64 * sizeof(float), cudaMemcpyDeviceToHost ));

        // for (i = 0; i < image1->infoHeader.height ; ++i)
        // {
        //         for (j = 0; j < image1->infoHeader.width ; ++j)
        //                 printf("%c ",red_dct[i*image1->infoHeader.height + j ]);
        // for (i = 0; i < 8 ; ++i)
        // {
        //         for (j = 0; j < 8 ; ++j)
        //                 printf("%d ",test[i*8 + j ]);
        //         printf("\n");
        // }

        // printf("\n\n");

        for (i = 0; i < 8 ; ++i)
        {
                for (j = 0; j < 8 ; ++j)
                        printf("%f ",test_dct[i*8 + j ]);
                printf("\n");
        }
        gpuErrchk(cudaFree( d_red ));
        gpuErrchk(cudaFree( d_green ));
        gpuErrchk(cudaFree( d_blue ));
        gpuErrchk(cudaFree( d_red_dct ));
        return 0;



}
