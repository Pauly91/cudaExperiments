#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmpReader.h"

#define numColor 3


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

__global__ void add(unsigned char *red, unsigned char *dct)
{
        printf("Kernel Code\n");
}


int main(int argc, char *argv[])
{


        BMPData *image1 = NULL;
        int i = 0,j = 0;
        int size = 0;
        unsigned char *red = NULL,*green = NULL,*blue = NULL;
        unsigned char *d_red = NULL,*d_green = NULL,*d_blue = NULL;

        unsigned char *d_red_dct = NULL;
        unsigned char *red_dct = NULL;


        if((image1 = readBMPfile(argv[1])) == NULL)
        {
                printf("Error in File 1\n");
                return -1;
        }

        size = image1->infoHeader.height * image1->infoHeader.width;

        red = (unsigned char *) malloc(size * sizeof(char) );
        blue = (unsigned char *) malloc(size * sizeof(char) );
        green = (unsigned char *) malloc(size * sizeof(char) );


        red_dct = (unsigned char *) malloc(size * sizeof(char*) );

        for (i = 0; i < size * numColor; i+=3)
        {
                        blue[i] = image1->bitMapImage[i]; 
                        green[i] = image1->bitMapImage[i + 1];
                        red[i] = image1->bitMapImage[i + 2];
        }

        gpuErrchk( cudaMalloc( (void**)&d_red, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_green, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_blue, size * sizeof(char) ));
        gpuErrchk( cudaMalloc( (void**)&d_red_dct, size * sizeof(char) ));


        gpuErrchk(cudaMemcpy( d_red, &red, size * sizeof(char) , cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_green, &green, size * sizeof(char), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemcpy( d_blue, &blue, size * sizeof(char), cudaMemcpyHostToDevice ));

        add<<< 1, 1 >>>(d_red,d_red_dct);

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
