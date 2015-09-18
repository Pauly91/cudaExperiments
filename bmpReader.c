#include <stdio.h>
#include <stdlib.h>
#include "bmpReader.h"
#define numColor 3

BMPData* readBMPfile(const char *filename)
{
	BMPData *bmpData;
	bmpData = (BMPData*) malloc(sizeof(BMPData));
	FILE *fp;
	int ERROR;
	if((fp = fopen(filename ,"rb")) == NULL)
	{
		printf("Error Reading File\n");
		fclose(fp);
		return NULL;
	}

	if(fread(&bmpData->header,sizeof(bmpData->header),1,fp) != 1)
	{
		printf("Error Reading Header\n");
		fclose(fp);
		return NULL;
	}

	if (bmpData->header.type != 19778)
	{
		printf("Not a BMP file\n");
		fclose(fp);
		return NULL;
	}
	// printf("**Header**\n");
	// printf(" type: 0x%x \n fileSize: %d \n reserved1: %d \n offset: %d \n",header.type,header.fileSize,header.reserved,header.offset);

	if(fread(&bmpData->infoHeader,sizeof(bmpData->infoHeader),1,fp) != 1)
	{
		printf("Error Reading Header\n");
		fclose(fp);
		return NULL;
	}
	// printf("\n\n**InfoHeader**\n");
	//printf(" size: %d \n width: %d \n height: %d \n planes: %d \n bits: %d \n",bmpData->infoHeader.size,bmpData->infoHeader.width,bmpData->infoHeader.height,bmpData->infoHeader.planes,bmpData->infoHeader.bits); 
	// printf(" compression: %d \n imagesize: %d \n xResolution: %d \n yResolution: %d \n",infoHeader.compression,infoHeader.imagesize,infoHeader.xResolution,infoHeader.yResolution);
	// printf(" coloursUsed: %d \n importantColours: %d \n\n",infoHeader.coloursUsed,infoHeader.importantColours);

	bmpData->bitMapImage = (unsigned char *) malloc(bmpData->infoHeader.imagesize * sizeof(char));

	fseek(fp,bmpData->header.offset,SEEK_SET);
	if((ERROR = fread(bmpData->bitMapImage,bmpData->infoHeader.imagesize,1,fp))  != 1) 
	{
		printf("Error Reading Image BitMap\n");
		printf("ERROR:%d\n",ERROR);
		free(bmpData->bitMapImage);
		fclose(fp);
		return NULL;
	}
	fclose(fp);
	return bmpData;
}

int writeBMPfile(BMPData *data,const char *filename)
{

	int ERROR;
	FILE *image;
	if((image = fopen(filename,"wb")) == NULL)
	{
		printf("Error Reading File\n");
		fclose(image);
		return -1;
	}

	if((ERROR = fwrite(&(data->header),sizeof(data->header),1,image)) != 1)
	{
		printf("Error Writing Header\n");
		printf("ERROR: %d \n",ERROR);
		fclose(image);
		return -1;
	}

	if(fwrite(&(data->infoHeader),sizeof(data->infoHeader),1,image) != 1)
	{
		printf("Error Writing InfoHeader\n");
		fclose(image);
		return -1;
	}

	fseek(image,data->header.offset,SEEK_SET);
	if((ERROR = fwrite((data->bitMapImage),data->infoHeader.imagesize,1,image))  != 1) // check if sizeof is needed ?
	{
		printf("Error Writing Image BitMap\n");
		printf("ERROR:%d\n",ERROR);
		fclose(image);
		return -1;
	}
	fclose(image);
	return 1;
}


void colourVector2MatrixConverter(unsigned char ** red,unsigned char ** green,unsigned char ** blue,unsigned char * bitMapImage,int height, int width)
{
	int i,j,k;
	for (i = 0,j = 0, k = 0; i < height * width * numColor; i+=3)
	{
			blue[j][k] = bitMapImage[i]; 
			green[j][k] = bitMapImage[i + 1];
			red[j][k] = bitMapImage[i + 2];
	 		// printf(" %u j:%d k:%d\n",red[j][k],j,k);
			k++;
			if(k == width)
			{
				j++;
				k = 0;
			}
		
	}
}

void rgb2yuv(unsigned char ** red,unsigned char ** green,unsigned char ** blue, unsigned char ** Y,unsigned char ** U,unsigned char ** V,int height, int width)
{
	int i,j;
	int value;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			value = red[i][j] * 0.2999 + green[i][j] * 0.587 + blue[i][j] * 0.114;  
			if(value > 255)
				Y[i][j] = 255;
			else if(value < 0)
				Y[i][j] = 0;
			else
				Y[i][j] = (unsigned char) value; 

			value = red[i][j] * (-0.168736) + green[i][j] * (-0.331264) + blue[i][j] * (0.500002) + 128;
			if(value > 255)
				U[i][j] = 255;
			else if(value < 0)
				U[i][j] = 0;
			else
				U[i][j] = (unsigned char) value; 

			value = red[i][j] * 0.5 + green[i][j] * (-0.418688) + blue[i][j] * (-0.081312) + 128;
			if(value > 255)
				V[i][j] = 255;
			else if(value < 0)
				V[i][j] = 0;
			else
				V[i][j] = (unsigned char) value; 			
		}
	}

}


void yuv2rgb(unsigned char ** red,unsigned char ** green,unsigned char ** blue, unsigned char ** Y,unsigned char ** U,unsigned char ** V,int height, int width)
{	
	int i,j;
	int value;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			value = Y[i][j] + (V[i][j] - 128 )* 1.4021;  
			if(value > 255)
				red[i][j] = 255;
			else if(value < 0)
				red[i][j] = 0;
			else
				red[i][j] = (unsigned char) value;

			value = Y[i][j] + (U[i][j] - 128 ) * (-0.34414) + (V[i][j] - 128) * (-0.71414);  
			if(value > 255)
				green[i][j] = 255;
			else if(value < 0)
				green[i][j] = 0;
			else
				green[i][j] = (unsigned char) value;

			value = Y[i][j] + (U[i][j] - 128) * 1.77180;
			if(value > 255)
				blue[i][j] = 255;
			else if(value < 0)
				blue[i][j] = 0;
			else
				blue[i][j] = (unsigned char) value;

		}
	}

}

void colourMatrix2VectorConverter(unsigned char ** red,unsigned char ** green,unsigned char ** blue,unsigned char * bitMapImage,int height, int width)
{
	int i,j,k;
	for (i = 0,j = 0, k = 0; i < height * width * numColor; i+=3)
	{
			bitMapImage[i] = blue[j][k]; 
			bitMapImage[i + 1] = green[j][k]; 
			bitMapImage[i + 2] = red[j][k]; 
			k++;
			if(k == width)
			{
				j++;
				k = 0;
			}
		
	}
}
